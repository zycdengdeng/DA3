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

from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.pipeline.colorize import (
    build_anchor_views as _build_anchor_views,
    colorize_cloud_from_views as _colorize_cloud_from_views,
)
from lidar_anchored_depth.pipeline.depth_completion import (
    accumulate_static_lidar as _accumulate_static_lidar,
    refine_one_frame as _refine_one_frame,
    run_refine_parallel as _run_refine_parallel,
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
    voxel_downsample,
    voxel_downsample_robust,
    write_ply_xyz,
)
from lidar_anchored_depth.viz.bev import (
    auto_bev_range as _auto_bev_range,
    render_bev as _render_bev,
)


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id



_DEPRECATION_BANNER = """
================================================================================
  DEPRECATION: scripts/run_point_completion_inference.py is now legacy.

  Equivalent commands (single-stage):
    lad complete   --scene.scene <X> ...    # produces refined / baseline / hybrid PLYs
    lad inject     --scene.scene <X> ...    # adds dynamic objects at one anchor ts
    lad render-bev --scene.scene <X> ...    # per-ts BEV PNGs for video assembly

  End-to-end:
    lad pipeline --complete.scene.scene <X> ...

  Migration guide: docs/cli.md
================================================================================
""".strip()


def main() -> int:
    import sys as _sys
    print(_DEPRECATION_BANNER, file=_sys.stderr)
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
    parser.add_argument(
        "--gpu-ids", default=None,
        help="comma-separated GPU ids for multi-GPU sharding of the "
        "per-(cam, ts) refinement, e.g. '0,1,2,3'. Each id gets its "
        "own worker process; (cam, ts) work units are distributed "
        "round-robin. When unset (default), runs serially on "
        "--residual-device. Per-frame seeds are derived from "
        "(scene, cam, ts) so results are reproducible regardless of "
        "the number of workers.",
    )
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

    parser.add_argument(
        "--lidar-skip-ground", action="store_true", default=True,
        help="drop LiDAR points whose Z is within "
        "--lidar-skip-ground-band-m of the LiDAR-derived ground grid "
        "BEFORE Layer 4 LiDAR-priority fusion. The ground geometry is "
        "already locked by ground-snap (Layer 3 + post-snap), so "
        "keeping the LiDAR ground points only adds the polar scan-"
        "ring pattern visible in the BEV. With this flag on, road "
        "voxels are filled by the uniform refined cloud; verticals "
        "(poles, walls, cars) still get LiDAR-priority. Default ON.",
    )
    parser.add_argument(
        "--no-lidar-skip-ground", action="store_false",
        dest="lidar_skip_ground",
    )
    parser.add_argument(
        "--lidar-skip-ground-band-m", type=float, default=0.5,
        help="ground-band thickness around the LiDAR ground grid, in "
        "metres. Slightly larger than ground-snap-max-dz (default "
        "0.4 m) so we also catch grazing returns just above / below "
        "the snapped ground. Default 0.5 m.",
    )

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
    parser.add_argument(
        "--sam-auto-dir", default=None,
        help="optional dir of SAM Auto npz (output of run_sam_auto.py). "
        "Use --segformer-dir instead when available — SegFormer covers "
        "every pixel; SAM Auto leaves road/sky/poles unsegmented.",
    )
    parser.add_argument(
        "--segformer-dir", default=None,
        help="optional dir of SegFormer npz (output of "
        "run_segformer_inference.py). Per-pixel Cityscapes class as "
        "segment_id. Should match what was used at training time. "
        "If both --sam-auto-dir and --segformer-dir are given, "
        "--segformer-dir wins.",
    )
    parser.add_argument(
        "--video-mode", action="store_true",
        help="iterate every ts in the scene; for each ts inject the "
        "per-object snapshots at THAT ts (so dynamic objects move "
        "naturally across frames) and render a BEV PNG. The static "
        "refined+LiDAR cloud is computed once and reused. Output: "
        "<output>/bev_<i:04d>.png + optional per-frame PLY.",
    )
    parser.add_argument(
        "--video-ts-stride", type=int, default=1,
        help="render every Nth ts (default 1 = every ts).",
    )
    parser.add_argument(
        "--video-write-plys", action="store_true",
        help="also save per-frame .ply (debug; lots of disk).",
    )
    parser.add_argument(
        "--video-max-extrap-ms", type=int, default=200,
        help="for video pose interpolation: when the video ts is "
        "outside the object's annotation window by more than this, "
        "the object is not rendered. Default 200 ms — enough to "
        "absorb V2X annotation gaps but small enough to make "
        "objects truly disappear when they leave the scene.",
    )
    parser.add_argument(
        "--video-max-gap-ms", type=int, default=500,
        help="if the two annotated ts that bracket the current "
        "video ts are farther apart than this, the object is NOT "
        "rendered (the gap usually means V2X tracking failed across "
        "that span; interpolating across it teleports the object). "
        "Default 500 ms (= 5 ts at 10 Hz). Set to a very large value "
        "to disable the check.",
    )
    parser.add_argument(
        "--bev-image-size", type=int, nargs=2, default=[1024, 1024],
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--bev-x-range", type=float, nargs=2, default=None,
        metavar=("X_MIN", "X_MAX"),
        help="world X range in metres (auto-computed from data if unset)",
    )
    parser.add_argument(
        "--bev-y-range", type=float, nargs=2, default=None,
        metavar=("Y_MIN", "Y_MAX"),
    )
    parser.add_argument(
        "--bev-pad-m", type=float, default=5.0,
        help="padding around the auto-computed BEV bounds (default 5 m)",
    )
    parser.add_argument(
        "--robust-color", default="median",
        choices=["mean", "median", "mad-trim"],
        help="per-voxel colour aggregation when multiple (cam, ts) "
        "samples land in the same voxel. 'mean' = simple average "
        "(fast, but momentary occluders bleed in). 'median' = "
        "outlier-immune up to 50%% bad samples per channel "
        "(default; recommended for static accumulation where moving "
        "vehicles briefly occlude road voxels). 'mad-trim' = median "
        "+ MAD-based inlier filter, then mean of inliers.",
    )
    parser.add_argument(
        "--colorize-lidar", action="store_true", default=True,
        help="re-colour every fused-cloud point by projecting it into "
        "the anchor-ts cameras and sampling image RGB (closest camera "
        "wins). LiDAR points that started out grey now get their real "
        "photographic colour, so the final hybrid PLY is no longer "
        "dominated by grey. Default ON.",
    )
    parser.add_argument(
        "--no-colorize-lidar", action="store_false",
        dest="colorize_lidar",
    )
    parser.add_argument(
        "--colorize-only-lidar-points", action="store_true", default=True,
        help="when re-colouring, only touch LiDAR-source points "
        "(which started out fall-back grey) and leave AA-HAD points' "
        "multi-frame median colours intact. Default ON. Set "
        "--no-colorize-only-lidar-points to re-colour everything from "
        "the anchor-ts cameras (matches the old behaviour).",
    )
    parser.add_argument(
        "--no-colorize-only-lidar-points", action="store_false",
        dest="colorize_only_lidar_points",
    )
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
    sam_auto_dir = Path(args.sam_auto_dir) if args.sam_auto_dir else None
    if sam_auto_dir is not None and not sam_auto_dir.is_dir():
        raise SystemExit(f"--sam-auto-dir {sam_auto_dir} not a directory")
    segformer_dir = Path(args.segformer_dir) if args.segformer_dir else None
    if segformer_dir is not None and not segformer_dir.is_dir():
        raise SystemExit(f"--segformer-dir {segformer_dir} not a directory")

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

    # Build the (cam, ts) work list. Both serial and parallel paths
    # consume this same list; the per-frame body lives in
    # _refine_one_frame() so the two paths cannot drift.
    work_items: list[tuple[str, int]] = []
    for cam_id in args.cams:
        if cam_id not in calib:
            print(f"  cam{cam_id}: missing from calib, skipping")
            continue
        for ts_ms in scene.timestamps_ms:
            work_items.append((cam_id, int(ts_ms)))

    gpu_ids: list[int] = []
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]

    refined_chunks: list[np.ndarray] = []
    refined_rgb_chunks: list[np.ndarray] = []
    baseline_chunks: list[np.ndarray] = []
    baseline_rgb_chunks: list[np.ndarray] = []
    n_refined_frames = 0
    n_skipped = 0
    t_total = time.time()
    print()

    if len(gpu_ids) > 1:
        # ---- Multi-GPU parallel path ----
        # Parent has not imported torch yet, which is intentional:
        # workers must be the first to touch CUDA so each child sees
        # only its assigned card under CUDA_VISIBLE_DEVICES.
        results = _run_refine_parallel(
            gpu_ids,
            args,
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
                    f"  [worker error] cam{res['cam_id']} ts{res['ts_ms']}: "
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
        # ---- Serial single-GPU path (default) ----
        from lidar_anchored_depth.models import (
            PointResidualPredictor, build_prior_features,
        )
        device = args.residual_device
        if gpu_ids:
            device = f"cuda:{gpu_ids[0]}"
        predictor = PointResidualPredictor(
            args.residual_checkpoint,
            device=device,
            n_steps=args.residual_steps,
            gate_distance_m=args.gate_distance_m,
        )
        print(
            f"[net] loaded {args.residual_checkpoint}  "
            f"device={predictor.device}  steps={predictor.n_steps}"
        )

        per_cam_count: dict[str, int] = {c: 0 for c in args.cams}
        per_cam_t0: dict[str, float] = {c: time.time() for c in args.cams}
        last_cam = None
        for cam_id, ts_ms in work_items:
            if cam_id != last_cam:
                if last_cam is not None:
                    print(
                        f"  cam{last_cam}  refined {per_cam_count[last_cam]} "
                        f"frames  ({time.time() - per_cam_t0[last_cam]:.1f}s)"
                    )
                per_cam_t0[cam_id] = time.time()
                last_cam = cam_id
            res = _refine_one_frame(
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
                args=args,
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
    print(f"[accum] N refined = {refined_xyz.shape[0]}  "
          f"N baseline = {baseline_xyz.shape[0]}")

    # Voxel-downsample for visualisation parity. Robust per-voxel
    # colour aggregation (median by default) handles the common
    # multi-frame artefact of a moving vehicle briefly writing its
    # colour onto a road voxel as it passes through.
    refined_v_xyz, refined_v_rgb = voxel_downsample_robust(
        refined_xyz, args.voxel_size, colors=refined_rgb,
        color_method=args.robust_color,
    )
    baseline_v_xyz, baseline_v_rgb = voxel_downsample_robust(
        baseline_xyz, args.voxel_size, colors=baseline_rgb,
        color_method=args.robust_color,
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

    # Build anchor-ts views once for image colour transfer.
    anchor_views = []
    if args.colorize_lidar and args.object_anchor_ts_ms is not None:
        anchor_views = _build_anchor_views(
            loader, scene_id, int(args.object_anchor_ts_ms), args.cams,
        )
        print(f"[colorize] anchor cams available at ts={args.object_anchor_ts_ms}: "
              f"{len(anchor_views)} of {len(args.cams)}")

    # Optional Layer 4 LiDAR-priority fusion.
    if args.include_lidar_priority:
        print()
        print("[layer4] LiDAR-priority fuse on refined cloud")
        lidar_fov = points_in_any_camera_fov(
            static_lidar_full, cam_views, z_min=args.z_min, z_max=args.z_max,
        )
        lidar_for_fuse = static_lidar_full[lidar_fov]
        # Keep an un-thinned copy for the outlier-rejection KD-tree.
        # If we hand `lidar_priority_fill` only the post-skip-ground
        # backbone, refined points on the road surface end up with
        # "nearest LiDAR" several metres up on the verticals and get
        # falsely dropped — leaving a halo of refined points around
        # each LiDAR sensor (visible as concentric BEV rings).
        lidar_outlier_ref = lidar_for_fuse

        # Optionally drop LiDAR points whose Z falls in the ground
        # band — they only contribute the polar scan-ring artefact in
        # the BEV. The road geometry is already locked by ground-snap
        # so the cm-precision LiDAR brought to the ground voxels is
        # decorative; removing it lets the uniform refined cloud fill
        # those voxels instead.
        if args.lidar_skip_ground and ground_grid is not None and lidar_for_fuse.size > 0:
            z_g, in_grid = ground_grid.query(lidar_for_fuse[:, :2])
            band = float(args.lidar_skip_ground_band_m)
            is_ground = in_grid & (np.abs(lidar_for_fuse[:, 2] - z_g) <= band)
            n_dropped = int(is_ground.sum())
            lidar_for_fuse = lidar_for_fuse[~is_ground]
            print(
                f"  [skip-ground] dropped {n_dropped} LiDAR points in "
                f"ground band (±{band:.2f}m); {lidar_for_fuse.shape[0]} "
                f"non-ground LiDAR remain"
            )

        fused_xyz, fused_rgb, source = lidar_priority_fill(
            lidar_for_fuse, refined_xyz, refined_rgb,
            voxel_size=args.voxel_size,
            max_dist_to_lidar=args.max_dist_to_lidar,
            outlier_reference_lidar=lidar_outlier_ref,
        )
        n_lidar = int((source == 0).sum())
        n_aahad = int((source == 1).sum())
        print(f"  LiDAR backbone={n_lidar}  refined fill={n_aahad}")

        if args.colorize_lidar and anchor_views:
            if args.colorize_only_lidar_points:
                # Only re-colour LiDAR-source points (which start out
                # fall-back grey); keep the AA-HAD-fill points'
                # multi-frame-aggregated image colours as-is.
                lidar_mask = source == 0
                lidar_xyz = fused_xyz[lidar_mask]
                lidar_rgb = fused_rgb[lidar_mask]
                lidar_rgb_col, n_recol = _colorize_cloud_from_views(
                    lidar_xyz, lidar_rgb, anchor_views,
                )
                fused_rgb = fused_rgb.copy()
                fused_rgb[lidar_mask] = lidar_rgb_col
                print(f"  [colorize-lidar-only] {n_recol}/"
                      f"{int(lidar_mask.sum())} LiDAR points "
                      f"got image RGB")
            else:
                fused_rgb_col, n_recol = _colorize_cloud_from_views(
                    fused_xyz, fused_rgb, anchor_views,
                )
                print(f"  [colorize] {n_recol}/{fused_xyz.shape[0]} points "
                      f"received fresh image RGB")
                fused_rgb = fused_rgb_col

        fused_v_xyz, fused_v_rgb = voxel_downsample_robust(
            fused_xyz, args.voxel_size, colors=fused_rgb,
            color_method=args.robust_color,
        )
        write_ply_xyz(
            out_dir / f"{scene_id}_completion_3d_hybrid.ply",
            fused_v_xyz, fused_v_rgb, binary=args.ply_binary,
        )

    inj_log: list[dict] = []
    n_dyn_total = 0
    if args.recon_dir:
        from lidar_anchored_depth.pipeline.object_snapshot import (
            build_object_pose_cache, discover_object_clouds,
            inject_object_snapshots,
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

        # Build ts_cache ONCE — turns inject_object_snapshots from
        # O(n_objects × n_ts × frame_load) into O(n_objects) per call.
        # Critical for video mode (where it is called per ts) but it
        # also speeds up the single-anchor call by ~50x.
        t_cache = time.time()
        ts_cache = build_object_pose_cache(
            loader, scene_id, cam_for_discovery=args.cams[0],
        )
        print(f"  built object pose cache: {len(ts_cache)} obj_ids "
              f"({time.time() - t_cache:.1f}s)")

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
            ts_cache=ts_cache,
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

            # Colorize-with-dyn: don't touch dyn (it has per-object
            # accumulated colours which average across the object's
            # observations, useful when no single ts shows all sides);
            # only re-colour LiDAR-source points if requested.
            if args.colorize_lidar and anchor_views and not args.colorize_only_lidar_points:
                fused_with_dyn_rgb_col, n_recol = _colorize_cloud_from_views(
                    fused_with_dyn_xyz, fused_with_dyn_rgb, anchor_views,
                )
                print(f"  [colorize-with-dyn] {n_recol}/"
                      f"{fused_with_dyn_xyz.shape[0]} points coloured")
                fused_with_dyn_rgb = fused_with_dyn_rgb_col

            full_v_xyz, full_v_rgb = voxel_downsample_robust(
                fused_with_dyn_xyz, args.voxel_size, colors=fused_with_dyn_rgb,
                color_method=args.robust_color,
            )
            write_ply_xyz(
                out_dir / f"{scene_id}_completion_3d_hybrid_with_dyn.ply",
                full_v_xyz, full_v_rgb, binary=args.ply_binary,
            )
            print(f"  ↳ hybrid_with_dyn voxels = {full_v_xyz.shape[0]}")

        # ------- VIDEO MODE -----------------------------------------
        if args.video_mode:
            try:
                from PIL import Image as PILImage
            except ModuleNotFoundError as e:
                raise SystemExit("--video-mode requires Pillow") from e

            print()
            print("[video] per-ts BEV rendering")
            x_range = (
                tuple(args.bev_x_range) if args.bev_x_range else None
            )
            y_range = (
                tuple(args.bev_y_range) if args.bev_y_range else None
            )
            if x_range is None or y_range is None:
                auto_x, auto_y = _auto_bev_range(
                    fused_xyz, pad_m=args.bev_pad_m,
                )
                x_range = x_range or auto_x
                y_range = y_range or auto_y
            print(f"  BEV  x={x_range}  y={y_range}  size={tuple(args.bev_image_size)}")

            bev_dir = out_dir / "bev_frames"
            bev_dir.mkdir(parents=True, exist_ok=True)
            ply_dir = None
            if args.video_write_plys:
                ply_dir = out_dir / "frame_plys"
                ply_dir.mkdir(parents=True, exist_ok=True)

            ts_list = list(scene.timestamps_ms)[::max(1, args.video_ts_stride)]
            print(f"  rendering {len(ts_list)} frames (stride={args.video_ts_stride})")

            # Pre-voxelise the static cloud ONCE outside the per-ts
            # loop. Otherwise voxel_downsample on the full 50 M-point
            # static + dyn cloud is called per frame (~45 min/frame
            # via np.unique on a 50 M × 3 array). We pre-voxel down
            # to ~3.5 M; per-frame we only voxel-down the small dyn
            # part (~1 M -> ~100 k) and concatenate with the static
            # voxels. The BEV renderer's z-buffer correctly handles
            # any residual XY collisions between static and dyn.
            print("  pre-voxelising static cloud (one-time)...")
            t_pv = time.time()
            static_v_xyz, static_v_rgb = voxel_downsample_robust(
                fused_xyz, args.voxel_size, colors=fused_rgb,
                color_method=args.robust_color,
            )
            print(
                f"    static {fused_xyz.shape[0]} -> "
                f"{static_v_xyz.shape[0]} voxels  ({time.time() - t_pv:.1f}s)"
            )

            t_video = time.time()
            for i, ts in enumerate(ts_list):
                # Per-ts dynamic injection with linear pose
                # interpolation between the two annotated ts that
                # bracket this video frame. Objects outside their
                # annotation window (extrapolation > video_max_extrap_ms)
                # are dropped. Replaces the older "fall back to median
                # ts" behaviour which left sparsely-annotated cars
                # frozen at one position across the whole video.
                dyn_xyz_t, dyn_rgb_t, _ = inject_object_snapshots(
                    loader, scene_id, clouds,
                    cam_for_discovery=args.cams[0],
                    object_fusion=args.object_fusion,
                    object_voxel_size=args.object_voxel_size,
                    object_max_dist_to_lidar=args.object_max_dist_to_lidar,
                    mirror_axis=args.mirror_axis,
                    mirror_classes=(
                        frozenset(args.mirror_classes)
                        if args.mirror_classes else None
                    ),
                    static_obj_ids=static_ids,
                    ts_cache=ts_cache,
                    video_anchor_ts_ms=int(ts),
                    video_max_extrap_ms=args.video_max_extrap_ms,
                    video_max_gap_ms=args.video_max_gap_ms,
                )

                if dyn_xyz_t.shape[0] > 0:
                    # Voxel-down the dyn part (small, fast) so the
                    # combined cloud fed to BEV stays manageable.
                    dyn_v_xyz, dyn_v_rgb = voxel_downsample(
                        dyn_xyz_t, args.voxel_size, colors=dyn_rgb_t,
                    )
                    full_xyz = np.concatenate(
                        [static_v_xyz, dyn_v_xyz], axis=0,
                    )
                    full_rgb = np.concatenate(
                        [static_v_rgb, dyn_v_rgb], axis=0,
                    )
                else:
                    full_xyz = static_v_xyz
                    full_rgb = static_v_rgb

                bev_img = _render_bev(
                    full_xyz, full_rgb,
                    x_range=x_range, y_range=y_range,
                    image_hw=tuple(args.bev_image_size),
                )
                PILImage.fromarray(bev_img).save(
                    bev_dir / f"bev_{i:04d}.png",
                )
                if ply_dir is not None:
                    write_ply_xyz(
                        ply_dir / f"frame_{i:04d}.ply",
                        full_xyz, full_rgb, binary=True,
                    )

                if (i + 1) % 10 == 0 or i == len(ts_list) - 1:
                    elapsed = time.time() - t_video
                    print(
                        f"  frame {i + 1:>4}/{len(ts_list)}  "
                        f"ts={ts}  N={full_xyz.shape[0]}  "
                        f"({elapsed:.1f}s, {(i + 1) / elapsed:.2f} fps)"
                    )

            print()
            print(f"  ↳ {len(ts_list)} BEV PNGs in {bev_dir}")
            print(f"  ↳ to assemble video:")
            print(f"     ffmpeg -framerate 10 -i {bev_dir}/bev_%04d.png \\")
            print(f"            -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
            print(f"            {out_dir}/bev_video.mp4")

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
