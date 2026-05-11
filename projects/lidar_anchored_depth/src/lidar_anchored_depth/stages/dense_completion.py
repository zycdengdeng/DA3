"""``complete`` stage: Layer 2 refine + Layer 4 LiDAR-priority fuse.

Inputs (paths supplied via :class:`CompleteConfig`):

* DA3 per-frame depth ``.npz`` (``run_da3_inference.py``)
* SAM dynamic masks (``run_sam_inference.py``)
* SegFormer per-pixel segments (``run_segformer_inference.py``, optional)
* AA-HAD per-camera linear calibration JSON
* Trained point-completion network checkpoint

Outputs (written into ``<output_dir>/``):

* ``refined.ply``  — Δxyz-refined static cloud (voxel-down)
* ``baseline.ply`` — Layer 1+3 prior with Δxyz=0 for comparison
* ``hybrid.ply``   — Layer 4 LiDAR-priority fused with refined fill
* ``summary.json`` — config + voxel counts + frame counts
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.pipeline.colorize import (
    build_anchor_views,
    colorize_cloud_from_views,
)
from lidar_anchored_depth.pipeline.depth_completion import (
    accumulate_static_lidar,
    refine_one_frame,
    run_refine_parallel,
)
from lidar_anchored_depth.pipeline.ground_height_grid import (
    build_ground_height_grid,
)
from lidar_anchored_depth.pipeline.lidar_completion import (
    CameraView,
    lidar_priority_fill,
    points_in_any_camera_fov,
)
from lidar_anchored_depth.reconstruction import (
    voxel_downsample_robust,
    write_ply_xyz,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


def _flatten_to_namespace(cfg: CompleteConfig) -> argparse.Namespace:
    """Synthesise the flat ``args`` namespace that the per-frame and
    worker-pool primitives in :mod:`pipeline.depth_completion` were
    originally written for.

    Phase 2 keeps the primitive signatures unchanged to minimise risk.
    A later phase will lift the primitives to a typed config and
    delete this helper.
    """
    return argparse.Namespace(
        # SceneConfig -> flat
        data_root=str(cfg.scene.data_root),
        scene=cfg.scene.scene,
        cams=list(cfg.scene.cams),
        loader_min_points=cfg.scene.loader_min_points,
        # Workers do per-frame refine on dynamic-labels-source loader.
        dynamic_labels_source=cfg.scene.dynamic_labels_source,
        # External artifact paths
        sam_mask_dir=str(cfg.sam_mask_dir) if cfg.sam_mask_dir else None,
        sam_auto_dir=str(cfg.sam_auto_dir) if cfg.sam_auto_dir else None,
        segformer_dir=str(cfg.segformer_dir) if cfg.segformer_dir else None,
        residual_checkpoint=str(cfg.residual_checkpoint),
        # Stage 2 knobs
        aahad_pixel_stride=cfg.aahad_pixel_stride,
        z_min=cfg.z_min,
        z_max=cfg.z_max,
        sam_dilate_px=cfg.sam_dilate_px,
        bbox_expand=cfg.bbox_expand,
        max_points_per_frame=cfg.max_points_per_frame,
        require_v2x_frames=cfg.require_v2x_frames,
        residual_device=cfg.residual_device,
        residual_steps=cfg.residual_steps,
        gate_distance_m=cfg.gate_distance_m,
        post_network_ground_snap=cfg.post_network_ground_snap,
        ground_snap_max_dz=cfg.ground_snap_max_dz,
    )


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


class DenseCompletionStage(Stage[CompleteConfig]):
    """Run the trained point-completion network on every (cam, ts) frame
    of one scene, then fuse with LiDAR via Layer 4 (LiDAR-priority).

    Multi-GPU sharding is opt-in via ``cfg.runtime.gpu_ids``; the serial
    fallback runs on ``cfg.residual_device`` and is bit-for-bit identical
    for the same ``(scene, cam, ts)`` triple thanks to per-frame seeding.
    """

    name = "complete"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out_dir = self.output_dir
        flat_args = _flatten_to_namespace(cfg)
        gpu_ids: list[int] = list(cfg.runtime.gpu_ids)

        # ---- inputs index ----
        calib = json.loads(Path(cfg.calib_json).read_text())
        print(f"[load] static calib for cams: {sorted(calib.keys())}")

        d_paths_index: dict[tuple[str, int, str], Path] = {}
        for p in sorted(glob.glob(cfg.d_paths_glob)):
            try:
                data = np.load(p, allow_pickle=True)
                key = (
                    str(data["scene_id"]),
                    int(data["ts_ms"]),
                    str(data["cam_id"]),
                )
            except Exception:
                continue
            d_paths_index[key] = Path(p)
        print(f"[load] {len(d_paths_index)} d_tilde frames indexed")

        # ---- dynamic-labels loader (10 Hz, used everywhere except
        #      the static-cloud accumulation step) ----
        loader = RoadsideV2XLoader(
            data_root=cfg.scene.data_root,
            scenes=[cfg.scene.scene] if "_" in cfg.scene.scene else None,
            min_num_points=cfg.scene.loader_min_points,
            labels_source=cfg.scene.dynamic_labels_source,
        )
        if "_" not in cfg.scene.scene:
            loader.scene_filter = [cfg.scene.scene]
        scene_id = _resolve_scene_id(loader, cfg.scene.scene)
        scene = next(s for s in loader.scenes if s.scene_id == scene_id)
        print(
            f"[scene] {scene_id}  ts={len(scene.timestamps_ms)}  "
            f"cams={list(cfg.scene.cams)}  "
            f"dynamic_labels={cfg.scene.dynamic_labels_source}"
        )

        # ---- static-labels loader (typically 1 Hz hand-labels) ----
        # When static and dynamic sources match, reuse the loader to
        # avoid duplicating the V2X scene index. When they differ
        # (the recommended default), build a second loader pointing at
        # the hand-label folder so accumulate_static_lidar gets a
        # cleaner bbox mask and the static cloud doesn't pick up a
        # trail of leaked car points along moving-object trajectories.
        if cfg.scene.static_labels_source == cfg.scene.dynamic_labels_source:
            static_loader = loader
        else:
            static_loader = RoadsideV2XLoader(
                data_root=cfg.scene.data_root,
                scenes=(
                    [cfg.scene.scene] if "_" in cfg.scene.scene else None
                ),
                min_num_points=cfg.scene.loader_min_points,
                labels_source=cfg.scene.static_labels_source,
            )
            if "_" not in cfg.scene.scene:
                static_loader.scene_filter = [cfg.scene.scene]
            static_scene = next(
                s for s in static_loader.scenes if s.scene_id == scene_id
            )
            print(
                f"  static_labels={cfg.scene.static_labels_source}  "
                f"({len(static_scene.timestamps_ms)} hand-label ts)"
            )

        # ---- ground grid (from hand-label-masked static LiDAR) ----
        print("[ground] aggregating static LiDAR for ground grid")
        static_lidar_full = accumulate_static_lidar(
            static_loader, scene_id, list(cfg.scene.cams),
            bbox_expand=cfg.bbox_expand,
        )
        ground_grid = build_ground_height_grid(
            static_lidar_full,
            cell_size=cfg.ground_cell_size,
            low_quantile=cfg.ground_low_quantile,
        )
        print(f"  ground grid {ground_grid.height.shape}")

        # ---- cam_views (one CameraView per cam_id) ----
        cam_views: list[CameraView] = []
        for cam_id in cfg.scene.cams:
            for ts in scene.timestamps_ms:
                try:
                    idx = loader.find_frame_idx(scene_id, ts, cam_id)
                    f = loader.get_frame(idx)
                    cam_views.append(CameraView(
                        cam_id=cam_id, K=f.K, T_wc=f.T_wc,
                        image_hw=f.image.shape[:2],
                        distortion=f.meta.get("distortion"),
                    ))
                    break
                except ValueError:
                    continue

        # ---- (cam, ts) work list ----
        work_items: list[tuple[str, int]] = []
        for cam_id in cfg.scene.cams:
            if cam_id not in calib:
                print(f"  cam{cam_id}: missing from calib, skipping")
                continue
            for ts_ms in scene.timestamps_ms:
                work_items.append((cam_id, int(ts_ms)))

        refined_chunks: list[np.ndarray] = []
        refined_rgb_chunks: list[np.ndarray] = []
        baseline_chunks: list[np.ndarray] = []
        baseline_rgb_chunks: list[np.ndarray] = []
        n_refined_frames = 0
        n_skipped = 0
        t_total = time.time()
        print()

        # ---- refine ----
        if len(gpu_ids) > 1:
            results = run_refine_parallel(
                gpu_ids,
                flat_args,
                scene_id,
                calib,
                d_paths_index,
                ground_grid,
                cam_views,
                work_items,
            )
            for res in results:
                if res is None:
                    n_skipped += 1
                    continue
                if "error" in res:
                    print(
                        f"  [worker error] cam{res['cam_id']} "
                        f"ts{res['ts_ms']}: "
                        f"{res['error'].splitlines()[0]}"
                    )
                    n_skipped += 1
                    continue
                refined_chunks.append(res["refined_xyz"])
                refined_rgb_chunks.append(res["prior_rgb"])
                baseline_chunks.append(res["prior_xyz"])
                baseline_rgb_chunks.append(res["prior_rgb"])
                n_refined_frames += 1
            print(
                f"[parallel] refined {n_refined_frames} frames "
                f"({time.time() - t_total:.1f}s wall)"
            )
        else:
            from lidar_anchored_depth.models import (
                PointResidualPredictor,
                build_prior_features,
            )
            device = cfg.residual_device
            if gpu_ids:
                device = f"cuda:{gpu_ids[0]}"
            predictor = PointResidualPredictor(
                str(cfg.residual_checkpoint),
                device=device,
                n_steps=cfg.residual_steps,
                gate_distance_m=cfg.gate_distance_m,
            )
            print(
                f"[net] loaded {cfg.residual_checkpoint}  "
                f"device={predictor.device}  steps={predictor.n_steps}"
            )
            sam_dir = Path(cfg.sam_mask_dir) if cfg.sam_mask_dir else None
            sam_auto_dir = (
                Path(cfg.sam_auto_dir) if cfg.sam_auto_dir else None
            )
            segformer_dir = (
                Path(cfg.segformer_dir) if cfg.segformer_dir else None
            )
            last_cam: str | None = None
            per_cam_count: dict[str, int] = {c: 0 for c in cfg.scene.cams}
            per_cam_t0: dict[str, float] = {
                c: time.time() for c in cfg.scene.cams
            }
            for cam_id, ts_ms in work_items:
                if cam_id != last_cam:
                    if last_cam is not None:
                        print(
                            f"  cam{last_cam}  refined "
                            f"{per_cam_count[last_cam]} frames  "
                            f"({time.time() - per_cam_t0[last_cam]:.1f}s)"
                        )
                    per_cam_t0[cam_id] = time.time()
                    last_cam = cam_id
                res = refine_one_frame(
                    scene_id, cam_id, ts_ms,
                    loader=loader,
                    predictor=predictor,
                    build_prior_features_fn=build_prior_features,
                    calib=calib,
                    d_paths_index=d_paths_index,
                    ground_grid=ground_grid,
                    cam_views=cam_views,
                    sam_dir=sam_dir,
                    sam_auto_dir=sam_auto_dir,
                    segformer_dir=segformer_dir,
                    args=flat_args,
                )
                if res is None:
                    n_skipped += 1
                    continue
                refined_chunks.append(res["refined_xyz"])
                refined_rgb_chunks.append(res["prior_rgb"])
                baseline_chunks.append(res["prior_xyz"])
                baseline_rgb_chunks.append(res["prior_rgb"])
                n_refined_frames += 1
                per_cam_count[cam_id] += 1
            if last_cam is not None:
                print(
                    f"  cam{last_cam}  refined {per_cam_count[last_cam]} "
                    f"frames  ({time.time() - per_cam_t0[last_cam]:.1f}s)"
                )

        print()
        if not refined_chunks:
            raise SystemExit("no frames refined; aborting")

        refined_xyz = np.concatenate(refined_chunks, axis=0)
        refined_rgb = np.concatenate(refined_rgb_chunks, axis=0)
        baseline_xyz = np.concatenate(baseline_chunks, axis=0)
        baseline_rgb = np.concatenate(baseline_rgb_chunks, axis=0)
        print(
            f"[accum] N refined = {refined_xyz.shape[0]}  "
            f"N baseline = {baseline_xyz.shape[0]}"
        )

        # ---- voxel-down + write refine / baseline ----
        refined_v_xyz, refined_v_rgb = voxel_downsample_robust(
            refined_xyz, cfg.voxel_size, colors=refined_rgb,
            color_method=cfg.robust_color,
        )
        baseline_v_xyz, baseline_v_rgb = voxel_downsample_robust(
            baseline_xyz, cfg.voxel_size, colors=baseline_rgb,
            color_method=cfg.robust_color,
        )
        refined_ply = out_dir / "refined.ply"
        baseline_ply = out_dir / "baseline.ply"
        write_ply_xyz(
            refined_ply, refined_v_xyz, refined_v_rgb, binary=cfg.ply_binary,
        )
        write_ply_xyz(
            baseline_ply, baseline_v_xyz, baseline_v_rgb,
            binary=cfg.ply_binary,
        )
        print(
            f"  refined  voxels = {refined_v_xyz.shape[0]}  "
            f"baseline voxels = {baseline_v_xyz.shape[0]}"
        )

        # ---- anchor cams for colorisation ----
        anchor_views: list[dict] = []
        if cfg.colorize_lidar and cfg.object_anchor_ts_ms is not None:
            anchor_views = build_anchor_views(
                loader, scene_id, int(cfg.object_anchor_ts_ms),
                list(cfg.scene.cams),
            )
            print(
                f"[colorize] anchor cams at ts={cfg.object_anchor_ts_ms}: "
                f"{len(anchor_views)} of {len(cfg.scene.cams)}"
            )

        # ---- Layer 4: LiDAR-priority fuse ----
        files: dict[str, Path] = {
            "refined": refined_ply,
            "baseline": baseline_ply,
        }
        summary: dict[str, object] = {
            "n_refined_frames": n_refined_frames,
            "n_skipped": n_skipped,
            "n_refined_points": int(refined_xyz.shape[0]),
            "n_voxels_refined": int(refined_v_xyz.shape[0]),
            "n_voxels_baseline": int(baseline_v_xyz.shape[0]),
        }
        if cfg.include_lidar_priority:
            print()
            print("[layer4] LiDAR-priority fuse on refined cloud")
            lidar_fov = points_in_any_camera_fov(
                static_lidar_full, cam_views,
                z_min=cfg.z_min, z_max=cfg.z_max,
            )
            lidar_for_fuse = static_lidar_full[lidar_fov]
            lidar_outlier_ref = lidar_for_fuse

            if (
                cfg.lidar_skip_ground
                and ground_grid is not None
                and lidar_for_fuse.size > 0
            ):
                z_g, in_grid = ground_grid.query(lidar_for_fuse[:, :2])
                band = float(cfg.lidar_skip_ground_band_m)
                is_ground = in_grid & (
                    np.abs(lidar_for_fuse[:, 2] - z_g) <= band
                )
                n_dropped = int(is_ground.sum())
                lidar_for_fuse = lidar_for_fuse[~is_ground]
                print(
                    f"  [skip-ground] dropped {n_dropped} LiDAR points "
                    f"in ground band (±{band:.2f}m); "
                    f"{lidar_for_fuse.shape[0]} non-ground LiDAR remain"
                )

            fused_xyz, fused_rgb, source = lidar_priority_fill(
                lidar_for_fuse, refined_xyz, refined_rgb,
                voxel_size=cfg.voxel_size,
                max_dist_to_lidar=cfg.max_dist_to_lidar,
                outlier_reference_lidar=lidar_outlier_ref,
            )
            n_lidar = int((source == 0).sum())
            n_aahad = int((source == 1).sum())
            print(f"  LiDAR backbone={n_lidar}  refined fill={n_aahad}")

            if cfg.colorize_lidar and anchor_views:
                if cfg.colorize_only_lidar_points:
                    lidar_mask = source == 0
                    lidar_xyz = fused_xyz[lidar_mask]
                    lidar_rgb = fused_rgb[lidar_mask]
                    lidar_rgb_col, n_recol = colorize_cloud_from_views(
                        lidar_xyz, lidar_rgb, anchor_views,
                    )
                    fused_rgb = fused_rgb.copy()
                    fused_rgb[lidar_mask] = lidar_rgb_col
                    print(
                        f"  [colorize-lidar-only] {n_recol}/"
                        f"{int(lidar_mask.sum())} LiDAR points got "
                        f"image RGB"
                    )
                else:
                    fused_rgb_col, n_recol = colorize_cloud_from_views(
                        fused_xyz, fused_rgb, anchor_views,
                    )
                    print(
                        f"  [colorize] {n_recol}/{fused_xyz.shape[0]} "
                        f"points received fresh image RGB"
                    )
                    fused_rgb = fused_rgb_col

            fused_v_xyz, fused_v_rgb = voxel_downsample_robust(
                fused_xyz, cfg.voxel_size, colors=fused_rgb,
                color_method=cfg.robust_color,
            )
            hybrid_ply = out_dir / "hybrid.ply"
            write_ply_xyz(
                hybrid_ply, fused_v_xyz, fused_v_rgb,
                binary=cfg.ply_binary,
            )
            print(f"  hybrid voxels = {fused_v_xyz.shape[0]}")
            files["hybrid"] = hybrid_ply
            summary["n_lidar_backbone"] = n_lidar
            summary["n_aahad_fill"] = n_aahad
            summary["n_voxels_hybrid"] = int(fused_v_xyz.shape[0])

        summary["wall_time_s"] = round(time.time() - t_total, 2)
        return StageArtifacts(
            output_dir=out_dir, files=files, summary=summary,
        )
