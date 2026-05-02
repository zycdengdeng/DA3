"""Apply the trained point-cloud residual flow head — Stage 7D.

For each ``(scene, ts, cam)`` frame: rebuild the same Layer 1+3 prior
cloud the data-prep script produced, run the trained
:class:`PointResidualPredictor` to predict per-point Δxyz, apply it,
and accumulate the refined points across the scene. Output:

    <output>/<scene>_completion_3d_refined.ply  — refined static cloud
    <output>/<scene>_completion_3d_baseline.ply — same prior but with
                                                  Δxyz = 0 (i.e. the
                                                  closed-form Layer 1+3
                                                  baseline) for direct
                                                  visual comparison.
    <output>/<scene>_completion_3d_summary.json

Optionally fuses with static LiDAR via Layer 4 (LiDAR-priority).

Usage
-----
    PYTHONPATH=src python scripts/run_point_completion_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --calib-json preview/scene_recon/008_..._static_calib.json \\
        --residual-checkpoint preview/point_completion_ckpt/best.pt \\
        --output preview/completion_3d_refined/
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import points_in_oriented_bbox
from lidar_anchored_depth.alignment.projection import (
    camera_to_world,
    pixel_to_camera_ray,
)
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.pipeline.ground_height_grid import (
    build_ground_height_grid,
    snap_to_ground,
)
from lidar_anchored_depth.pipeline.lidar_completion import (
    CameraView,
    lidar_priority_fill,
    points_in_any_camera_fov,
)
from lidar_anchored_depth.reconstruction import (
    voxel_downsample,
    write_ply_xyz,
)
from lidar_anchored_depth.segmentation.sam_io import load_sam_dynamic_mask


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _accumulate_static_lidar(
    loader, scene_id, cams, *, bbox_expand,
):
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    chunks = []
    for ts in scene.timestamps_ms:
        idx = None
        for c in cams:
            try:
                idx = loader.find_frame_idx(scene_id, ts, c)
                break
            except ValueError:
                continue
        if idx is None:
            continue
        frame = loader.get_frame(idx)
        is_dyn = np.zeros(frame.lidar_world.shape[0], dtype=bool)
        for obj in frame.dynamic_objects or []:
            is_dyn |= points_in_oriented_bbox(
                frame.lidar_world, obj, expand=bbox_expand,
            )
        chunks.append(frame.lidar_world[~is_dyn])
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _build_prior_for_frame(
    frame, d_image, a, b, dyn_mask,
    *, pixel_stride, z_min, z_max, ground_grid, ground_max_dz,
):
    H, W = frame.image.shape[:2]
    if d_image.shape != (H, W):
        return None

    z_cam = (a * d_image + b).astype(np.float32)
    static_mask = ~dyn_mask
    if pixel_stride > 1:
        stride = np.zeros((H, W), dtype=bool)
        stride[::pixel_stride, ::pixel_stride] = True
        static_mask &= stride
    valid = (z_cam >= z_min) & (z_cam <= z_max) & np.isfinite(z_cam) & static_mask
    if not valid.any():
        return None
    rows, cols = np.where(valid)
    z_keep = z_cam[rows, cols].astype(np.float64)
    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    rays = pixel_to_camera_ray(uv, frame.K)
    P_cam = rays * z_keep[:, None]
    P_world = camera_to_world(P_cam, frame.T_wc)
    if ground_grid is not None:
        P_world, _ = snap_to_ground(
            P_world.astype(np.float64), ground_grid, max_dz=ground_max_dz,
        )
    rgb = frame.image[rows, cols].astype(np.uint8)
    uv_int = np.stack([cols, rows], axis=1).astype(np.int32)
    return P_world.astype(np.float64), rgb, uv_int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--d-paths-glob", required=True)
    parser.add_argument("--sam-mask-dir", default=None)
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--cams", nargs="+", default=["0", "3", "6", "9"])
    parser.add_argument("--bbox-expand", type=float, default=0.10)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=300.0)
    parser.add_argument("--sam-dilate-px", type=int, default=12,
        help="dilate dynamic SAM masks before subtracting from the "
        "static prior. Larger values eat more of the moving-object "
        "edges that DA3 hallucinates as 'static'. Default 12 px.",
    )
    parser.add_argument("--require-v2x-frames", action="store_true", default=True)
    parser.add_argument(
        "--no-require-v2x-frames", action="store_false",
        dest="require_v2x_frames",
    )
    parser.add_argument("--aahad-pixel-stride", type=int, default=4)
    parser.add_argument("--ground-cell-size", type=float, default=0.30)
    parser.add_argument("--ground-low-quantile", type=float, default=0.10)
    parser.add_argument("--ground-snap-max-dz", type=float, default=0.4)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--ply-binary", action="store_true")

    parser.add_argument("--residual-device", default="cuda")
    parser.add_argument("--residual-steps", type=int, default=4)
    parser.add_argument(
        "--gate-distance-m", type=float, default=None,
        help="if set, force Δxyz = 0 for prior points whose distance "
        "to nearest LiDAR exceeds this (matches the inference-side "
        "gating discussed for the 2-D head). Default: no gating.",
    )
    parser.add_argument(
        "--post-network-ground-snap", action="store_true", default=True,
        help="re-apply ground_snap on the network output so road "
        "points that were displaced by the residual flow get pinned "
        "back to the LiDAR-derived ground grid. Default ON — without "
        "this the road becomes scattered after the network because "
        "the per-point regression target = nearest-LiDAR is itself "
        "scattered around the true ground.",
    )
    parser.add_argument(
        "--no-post-network-ground-snap", action="store_false",
        dest="post_network_ground_snap",
    )

    parser.add_argument(
        "--include-lidar-priority", action="store_true", default=True,
        help="after the network refinement, voxel-fuse with the "
        "static LiDAR backbone (Layer 4) so cm-precise LiDAR wins "
        "in voxels where it exists. Default ON.",
    )
    parser.add_argument(
        "--no-include-lidar-priority", action="store_false",
        dest="include_lidar_priority",
    )
    parser.add_argument("--max-dist-to-lidar", type=float, default=2.0)

    parser.add_argument("--max-points-per-frame", type=int, default=12000,
                        help="cap per-(ts, cam) prior cloud size before "
                        "calling the network; KNN cost is O(N²) so this "
                        "controls peak memory.")
    parser.add_argument(
        "--recon-dir", default=None,
        help="optional directory of per-object PLYs from run_object_"
        "accumulation.py. When set, dynamic objects are placed into "
        "the final hybrid via the per-object accumulated reconstruction "
        "(0.27 m chamfer) at one anchor ts pose. The diffusion network "
        "is NOT applied to dynamic objects — they bypass it entirely.",
    )
    parser.add_argument("--object-anchor-ts-ms", type=int, default=None)
    parser.add_argument("--strict-anchor", action="store_true", default=True)
    parser.add_argument(
        "--no-strict-anchor", action="store_false", dest="strict_anchor",
    )
    parser.add_argument("--object-fusion", default="lidar-priority",
                        choices=["lidar-priority", "lidar-only", "aa-had-only"])
    parser.add_argument("--object-voxel-size", type=float, default=0.05)
    parser.add_argument("--object-max-dist-to-lidar", type=float, default=0.5)
    parser.add_argument(
        "--mirror-axis", default=None,
        choices=["x", "y", "z", None],
    )
    parser.add_argument(
        "--mirror-classes", nargs="+",
        default=["Car", "Suv", "Bus", "Truck"],
    )
    parser.add_argument("--motion-drift-threshold-m", type=float, default=0.5)
    parser.add_argument("--loader-min-points", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = json.loads(Path(args.calib_json).read_text())
    print(f"[load] static calib for cams: {sorted(calib.keys())}")

    d_paths_index = {}
    for p in sorted(glob.glob(args.d_paths_glob)):
        try:
            data = np.load(p, allow_pickle=True)
            key = (str(data["scene_id"]), int(data["ts_ms"]), str(data["cam_id"]))
        except Exception:
            continue
        d_paths_index[key] = Path(p)
    print(f"[load] {len(d_paths_index)} d_tilde frames indexed")

    sam_dir = Path(args.sam_mask_dir) if args.sam_mask_dir else None

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    print(f"[scene] {scene_id}  ts={len(scene.timestamps_ms)}  cams={args.cams}")

    print("[ground] aggregating static LiDAR for ground grid")
    static_lidar_full = _accumulate_static_lidar(
        loader, scene_id, args.cams, bbox_expand=args.bbox_expand,
    )
    ground_grid = build_ground_height_grid(
        static_lidar_full, cell_size=args.ground_cell_size,
        low_quantile=args.ground_low_quantile,
    )
    print(f"  ground grid {ground_grid.height.shape}")

    cam_views = []
    for cam_id in args.cams:
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

    # Load predictor (lazy import torch).
    from lidar_anchored_depth.models import (
        PointResidualPredictor, build_prior_features,
    )
    predictor = PointResidualPredictor(
        args.residual_checkpoint,
        device=args.residual_device,
        n_steps=args.residual_steps,
        gate_distance_m=args.gate_distance_m,
    )
    print(
        f"[net] loaded {args.residual_checkpoint}  device={predictor.device}  "
        f"steps={predictor.n_steps}"
    )

    refined_chunks: list[np.ndarray] = []
    refined_rgb_chunks: list[np.ndarray] = []
    baseline_chunks: list[np.ndarray] = []
    baseline_rgb_chunks: list[np.ndarray] = []
    n_refined_frames = 0
    n_skipped = 0
    t_total = time.time()
    print()
    for cam_id in args.cams:
        if cam_id not in calib:
            print(f"  cam{cam_id}: missing from calib, skipping")
            continue
        a = float(calib[cam_id]["a"])
        b = float(calib[cam_id]["b"])
        cam_t0 = time.time()
        cam_n = 0
        for ts_ms in scene.timestamps_ms:
            d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
            if d_path is None or not d_path.is_file():
                continue
            try:
                idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
            except ValueError:
                continue
            frame = loader.get_frame(idx)
            if args.require_v2x_frames and not (frame.dynamic_objects or []):
                continue
            d_data = np.load(d_path, allow_pickle=True)
            d_image = d_data["depth"].astype(np.float32)

            dyn_mask = load_sam_dynamic_mask(
                sam_dir, scene_id, ts_ms, cam_id, frame.image.shape[:2],
                dilate_px=args.sam_dilate_px,
            )
            prior = _build_prior_for_frame(
                frame, d_image, a, b, dyn_mask,
                pixel_stride=args.aahad_pixel_stride,
                z_min=args.z_min, z_max=args.z_max,
                ground_grid=ground_grid,
                ground_max_dz=args.ground_snap_max_dz,
            )
            if prior is None:
                n_skipped += 1
                continue
            prior_xyz, prior_rgb, prior_uv = prior

            # FOV cull
            fov = np.zeros(prior_xyz.shape[0], dtype=bool)
            for v in cam_views:
                fov |= points_in_any_camera_fov(prior_xyz, [v],
                                                z_min=args.z_min, z_max=args.z_max)
            if not fov.any():
                n_skipped += 1
                continue
            prior_xyz = prior_xyz[fov]
            prior_rgb = prior_rgb[fov]
            prior_uv = prior_uv[fov]

            # Subsample if too large
            if prior_xyz.shape[0] > args.max_points_per_frame:
                sel = np.random.choice(
                    prior_xyz.shape[0], size=args.max_points_per_frame, replace=False,
                )
                prior_xyz = prior_xyz[sel]
                prior_rgb = prior_rgb[sel]
                prior_uv = prior_uv[sel]
            if prior_xyz.shape[0] < 100:
                n_skipped += 1
                continue

            # Build per-frame static LiDAR target for KDTree distance.
            is_dyn = np.zeros(frame.lidar_world.shape[0], dtype=bool)
            for obj in frame.dynamic_objects or []:
                is_dyn |= points_in_oriented_bbox(
                    frame.lidar_world, obj, expand=args.bbox_expand,
                )
            target_lidar = frame.lidar_world[~is_dyn]

            feat = build_prior_features(
                prior_xyz, prior_uv, d_image, a, b, target_lidar,
            )
            dist_to_lidar = feat[:, 3]  # 4th channel is dist_to_lidar (clipped)

            refined_xyz = predictor.refine_cloud(
                prior_xyz, prior_rgb, feat,
                dist_to_lidar=dist_to_lidar,
            )

            if args.post_network_ground_snap:
                refined_xyz_snapped, _ = snap_to_ground(
                    refined_xyz.astype(np.float64), ground_grid,
                    max_dz=args.ground_snap_max_dz,
                )
                refined_xyz = refined_xyz_snapped.astype(np.float32)

            refined_chunks.append(refined_xyz)
            refined_rgb_chunks.append(prior_rgb)
            baseline_chunks.append(prior_xyz.astype(np.float32))
            baseline_rgb_chunks.append(prior_rgb)
            n_refined_frames += 1
            cam_n += 1

        print(f"  cam{cam_id}  refined {cam_n} frames  ({time.time() - cam_t0:.1f}s)")

    print()
    if not refined_chunks:
        raise SystemExit("no frames refined; aborting")

    refined_xyz = np.concatenate(refined_chunks, axis=0)
    refined_rgb = np.concatenate(refined_rgb_chunks, axis=0)
    baseline_xyz = np.concatenate(baseline_chunks, axis=0)
    baseline_rgb = np.concatenate(baseline_rgb_chunks, axis=0)
    print(f"[accum] N refined = {refined_xyz.shape[0]}  "
          f"N baseline = {baseline_xyz.shape[0]}")

    # Voxel-downsample for visualisation parity.
    refined_v_xyz, refined_v_rgb = voxel_downsample(
        refined_xyz, args.voxel_size, colors=refined_rgb,
    )
    baseline_v_xyz, baseline_v_rgb = voxel_downsample(
        baseline_xyz, args.voxel_size, colors=baseline_rgb,
    )
    write_ply_xyz(
        out_dir / f"{scene_id}_completion_3d_refined.ply",
        refined_v_xyz, refined_v_rgb, binary=args.ply_binary,
    )
    write_ply_xyz(
        out_dir / f"{scene_id}_completion_3d_baseline.ply",
        baseline_v_xyz, baseline_v_rgb, binary=args.ply_binary,
    )
    print(
        f"  refined  voxels = {refined_v_xyz.shape[0]}  "
        f"baseline voxels = {baseline_v_xyz.shape[0]}"
    )

    # Optional Layer 4 LiDAR-priority fusion.
    if args.include_lidar_priority:
        print()
        print("[layer4] LiDAR-priority fuse on refined cloud")
        lidar_fov = points_in_any_camera_fov(
            static_lidar_full, cam_views, z_min=args.z_min, z_max=args.z_max,
        )
        lidar_for_fuse = static_lidar_full[lidar_fov]
        fused_xyz, fused_rgb, source = lidar_priority_fill(
            lidar_for_fuse, refined_xyz, refined_rgb,
            voxel_size=args.voxel_size,
            max_dist_to_lidar=args.max_dist_to_lidar,
        )
        n_lidar = int((source == 0).sum())
        n_aahad = int((source == 1).sum())
        print(f"  LiDAR backbone={n_lidar}  refined fill={n_aahad}")
        fused_v_xyz, fused_v_rgb = voxel_downsample(
            fused_xyz, args.voxel_size, colors=fused_rgb,
        )
        write_ply_xyz(
            out_dir / f"{scene_id}_completion_3d_hybrid.ply",
            fused_v_xyz, fused_v_rgb, binary=args.ply_binary,
        )

    inj_log: list[dict] = []
    n_dyn_total = 0
    if args.recon_dir:
        from lidar_anchored_depth.pipeline.object_snapshot import (
            discover_object_clouds, inject_object_snapshots,
        )
        from lidar_anchored_depth.pipeline.v2x_static_classifier import (
            classify_v2x_objects, split_static_dynamic_ids,
        )

        recon_dir = Path(args.recon_dir)
        if not recon_dir.is_dir():
            raise SystemExit(f"--recon-dir {recon_dir} not a directory")
        print()
        print(f"[layer5] dynamic snapshot injection from {recon_dir}")
        clouds = discover_object_clouds(recon_dir, prefer_icp=True)
        motion = classify_v2x_objects(
            loader, scene_id,
            cam_for_discovery=args.cams[0],
            drift_threshold_m=args.motion_drift_threshold_m,
        )
        static_ids, dyn_ids = split_static_dynamic_ids(motion)
        print(
            f"  motion: {len(static_ids)} static (always-on), "
            f"{len(dyn_ids)} dynamic (anchor only)"
        )
        dyn_xyz, dyn_rgb, inj_log = inject_object_snapshots(
            loader, scene_id, clouds,
            cam_for_discovery=args.cams[0],
            anchor_ts_ms=args.object_anchor_ts_ms,
            strict_anchor=args.strict_anchor,
            object_fusion=args.object_fusion,
            object_voxel_size=args.object_voxel_size,
            object_max_dist_to_lidar=args.object_max_dist_to_lidar,
            mirror_axis=args.mirror_axis,
            mirror_classes=frozenset(args.mirror_classes) if args.mirror_classes else None,
            static_obj_ids=static_ids,
        )
        n_dyn_total = int(dyn_xyz.shape[0])
        n_kept = sum(1 for r in inj_log if r["n_pts"] > 0)
        print(f"  injected {n_dyn_total} pts across {n_kept}/{len(inj_log)} objs")

        # Stack on top of the hybrid produced above (refined static +
        # LiDAR backbone) and write a separate "_with_dyn.ply" so the
        # user can compare static-only vs static+dynamic side-by-side.
        if args.include_lidar_priority and n_dyn_total > 0:
            fused_with_dyn_xyz = np.concatenate([fused_xyz, dyn_xyz], axis=0)
            fused_with_dyn_rgb = np.concatenate([fused_rgb, dyn_rgb], axis=0)
            full_v_xyz, full_v_rgb = voxel_downsample(
                fused_with_dyn_xyz, args.voxel_size, colors=fused_with_dyn_rgb,
            )
            write_ply_xyz(
                out_dir / f"{scene_id}_completion_3d_hybrid_with_dyn.ply",
                full_v_xyz, full_v_rgb, binary=args.ply_binary,
            )
            print(f"  ↳ hybrid_with_dyn voxels = {full_v_xyz.shape[0]}")

    summary = {
        "scene_id": scene_id,
        "n_refined_frames": n_refined_frames,
        "n_skipped": n_skipped,
        "n_refined_points": int(refined_xyz.shape[0]),
        "n_voxels_refined": int(refined_v_xyz.shape[0]),
        "n_voxels_baseline": int(baseline_v_xyz.shape[0]),
        "n_dynamic_points": n_dyn_total,
        "object_snapshots": inj_log,
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in vars(args).items()},
    }
    summary_path = out_dir / f"{scene_id}_completion_3d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"  ↳ json : {summary_path}")
    print(f"  ↳ ply  : {out_dir}/{scene_id}_completion_3d_*.ply")
    print(f"  ↳ total time: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
