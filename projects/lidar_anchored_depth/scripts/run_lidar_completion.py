"""Sparse roadside LiDAR depth completion via AA-HAD — Stage 5.

This script is the *new* orchestrator for the static-scene branch. It
replaces the "two clouds at once + voxel mean" approach in
``run_static_scene_recon.py`` with a strict **LiDAR-as-backbone +
AA-HAD-as-fill** pipeline:

  1. Per-camera static (a, b) on static-LiDAR pairs (and optionally
     per-image-cell affine via ``--region-grid``).
  2. Unproject AA-HAD on (image \\ dynamic SAM mask) for every
     (camera, ts) frame; concatenate.
  3. Cull both LiDAR and AA-HAD to **at-least-one-camera FOV**.
  4. Build a 2.5-D LiDAR ground-height grid; **snap road-like
     AA-HAD points** onto the LiDAR-measured ground (fixes the
     near-camera road curvature artefact).
  5. **LiDAR-priority fill**: AA-HAD points only enter voxels that
     LiDAR did not already occupy (cm-precision LiDAR wins). Optional
     ``--max-dist-to-lidar`` filters far-flung outliers.
  6. **Per-object snapshot injection**: for every V2X bbox.id, load
     its accumulated per-object cloud (post-ICP, ~0.27 m chamfer)
     from a recon directory and place it at one representative ts's
     pose. Replaces the per-frame static prediction of dynamic
     objects (which "scatters") with the high-quality reconstruction.

Output PLYs (in ``--output``):

    <scene>_completion.ply           # static-only (LiDAR + AA-HAD fill)
    <scene>_completion_with_dyn.ply  # + per-object snapshots
    <scene>_completion_summary.json  # per-stage point counts + metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the static-calib + unproject helpers from the old script.
sys.path.insert(0, str(Path(__file__).parent))
from run_static_scene_recon import (  # noqa: E402
    _calibrate_grid_per_camera,
    _calibrate_static_per_camera,
    _unproject_static_branch,
    _resolve_scene_id,
)

from lidar_anchored_depth.alignment.bbox_anchor import points_in_oriented_bbox  # noqa: E402
from lidar_anchored_depth.data import RoadsideV2XLoader  # noqa: E402
from lidar_anchored_depth.pipeline.ground_height_grid import (  # noqa: E402
    build_ground_height_grid, snap_to_ground,
)
from lidar_anchored_depth.pipeline.lidar_completion import (  # noqa: E402
    CameraView, lidar_priority_fill, points_in_any_camera_fov,
)
from lidar_anchored_depth.pipeline.object_snapshot import (  # noqa: E402
    discover_object_clouds, inject_object_snapshots,
)
from lidar_anchored_depth.reconstruction import (  # noqa: E402
    voxel_downsample, write_ply_xyz,
)


def _build_camera_views(
    loader: RoadsideV2XLoader, scene_id: str, cams: list[str],
) -> dict[str, CameraView]:
    """Pull intrinsics + extrinsics + image dims for each camera once
    (V2X calibration is constant across ts for a roadside rig)."""
    views: dict[str, CameraView] = {}
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    for cam_id in cams:
        # Find any frame that has this camera.
        idx = None
        for ts in scene.timestamps_ms:
            try:
                idx = loader.find_frame_idx(scene_id, ts, cam_id)
                break
            except ValueError:
                continue
        if idx is None:
            continue
        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]
        views[cam_id] = CameraView(
            cam_id=cam_id, K=frame.K, T_wc=frame.T_wc,
            image_hw=(H, W),
            distortion=frame.meta.get("distortion"),
        )
    return views


def _accumulate_lidar_static(
    loader: RoadsideV2XLoader,
    scene_id: str,
    *,
    cams: list[str],
    bbox_expand: float = 0.10,
) -> np.ndarray:
    """Accumulate static-LiDAR (= LiDAR \\ ⋃ V2X bboxes) across all ts."""
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    chunks: list[np.ndarray] = []
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d-paths-glob", required=True)
    parser.add_argument("--sam-mask-dir", default=None)
    parser.add_argument(
        "--cams", nargs="+", default=["0", "3", "6", "9"],
    )
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--aahad-pixel-stride", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=300.0)
    parser.add_argument("--bbox-expand", type=float, default=0.10)
    parser.add_argument("--ply-binary", action="store_true")
    parser.add_argument("--loader-min-points", type=int, default=0)
    parser.add_argument("--sam-dilate-px", type=int, default=5)
    parser.add_argument(
        "--require-v2x-frames", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-require-v2x-frames", action="store_false",
        dest="require_v2x_frames",
    )
    parser.add_argument(
        "--region-grid", nargs=2, type=int, default=[0, 0],
        metavar=("ROWS", "COLS"),
    )
    parser.add_argument("--region-min-lidar-per-cell", type=int, default=30)

    parser.add_argument(
        "--ground-cell-size", type=float, default=0.30,
        help="XY cell size for the 2.5-D LiDAR ground-height grid (m). "
        "Default 0.30 m. Set to 0 to disable ground snapping.",
    )
    parser.add_argument(
        "--ground-low-quantile", type=float, default=0.10,
        help="quantile of LiDAR Z per cell that defines the ground "
        "height (0.10 = 10th percentile, conservative).",
    )
    parser.add_argument(
        "--ground-snap-max-dz", type=float, default=0.6,
        help="AA-HAD points whose |z - z_ground| <= this are snapped "
        "to z_ground; others left alone (default 0.6 m).",
    )

    parser.add_argument(
        "--max-dist-to-lidar", type=float, default=2.0,
        help="drop AA-HAD fill points whose distance to nearest LiDAR "
        "neighbour exceeds this (sanity filter; default 2 m).",
    )

    parser.add_argument(
        "--recon-dir", default=None,
        help="directory holding per-object PLYs from "
        "run_object_accumulation.py (e.g. preview/recon_3I_full/). "
        "When set, dynamic objects are placed back into the scene at "
        "their representative-ts pose using the per-object cloud.",
    )
    parser.add_argument(
        "--object-anchor-ts-ms", type=int, default=None,
        help="if set, prefer this ts as the snapshot pose for any "
        "object annotated at that ts (otherwise per-object median ts).",
    )
    parser.add_argument(
        "--include-object-lidar", action="store_true",
        help="also place each per-object LiDAR cloud (in addition to "
        "AA-HAD) at the snapshot pose. Default off.",
    )

    parser.add_argument(
        "--sigma-checkpoint", default=None,
        help="optional sigma-head .pt (Stage 4A). When set, the AA-HAD "
        "fill is sigma-weighted before LiDAR-priority filling.",
    )
    parser.add_argument("--sigma-device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- index DA3 files ----------------------------------------------
    d_paths_index: dict[tuple[str, int, str], Path] = {}
    for p in sorted(glob.glob(args.d_paths_glob)):
        try:
            data = np.load(p, allow_pickle=True)
        except Exception:
            continue
        try:
            key = (str(data["scene_id"]), int(data["ts_ms"]), str(data["cam_id"]))
        except KeyError:
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
    print(f"[scene] {scene_id}")
    print()

    # ---- 1. per-camera static (a, b) ----------------------------------
    print("[1/6] per-camera static (a, b)")
    cam_calib: dict[str, dict] = {}
    for cam_id in args.cams:
        info = _calibrate_static_per_camera(
            loader, scene_id, cam_id, d_paths_index, sam_dir,
            z_min=args.z_min, z_max=args.z_max,
            sam_dilate_px=args.sam_dilate_px,
            require_v2x_frames=args.require_v2x_frames,
        )
        if info is None:
            print(f"  cam{cam_id}: too few pairs, skipping")
            continue
        cam_calib[cam_id] = info
        print(
            f"  cam{cam_id}  N_pairs={info['n_pairs']:>7}  "
            f"a={info['a']:.3f}  b={info['b']:.3f}  "
            f"rmse={info['rmse_inliers_m']:.2f}m"
        )
    if not cam_calib:
        raise SystemExit("no camera calibrated; aborting")

    # ---- 1b. optional per-image-cell grid affine ----------------------
    grid_calib_by_cam = {}
    n_rows, n_cols = args.region_grid
    if n_rows > 0 and n_cols > 0:
        print()
        print(f"[1b/6] grid affine  rows={n_rows}  cols={n_cols}")
        for cam_id in cam_calib:
            calib, _ = _calibrate_grid_per_camera(
                loader, scene_id, cam_id, d_paths_index, sam_dir,
                z_min=args.z_min, z_max=args.z_max,
                sam_dilate_px=args.sam_dilate_px,
                require_v2x_frames=args.require_v2x_frames,
                n_rows=n_rows, n_cols=n_cols,
                min_lidar_per_cell=args.region_min_lidar_per_cell,
            )
            if calib is None:
                continue
            grid_calib_by_cam[cam_id] = calib
            print(
                f"  cam{cam_id}  grid cells solved={calib.n_cells_solved}/"
                f"{calib.n_cells_total}"
            )

    # ---- 2. AA-HAD unproject ------------------------------------------
    sigma_predictor = None
    if args.sigma_checkpoint:
        from lidar_anchored_depth.models import SigmaPredictor

        sigma_predictor = SigmaPredictor(args.sigma_checkpoint, device=args.sigma_device)

    print()
    print(f"[2/6] AA-HAD unproject  stride={args.aahad_pixel_stride}")
    aahad_pts: list[np.ndarray] = []
    aahad_rgb: list[np.ndarray] = []
    aahad_sig: list[np.ndarray] = []
    for cam_id, info in cam_calib.items():
        t0 = time.time()
        pts, rgb, sig = _unproject_static_branch(
            loader, scene_id, cam_id, info["a"], info["b"],
            d_paths_index, sam_dir,
            pixel_stride=args.aahad_pixel_stride,
            z_min=args.z_min, z_max=args.z_max,
            sigma_predictor=sigma_predictor,
            sam_dilate_px=args.sam_dilate_px,
            require_v2x_frames=args.require_v2x_frames,
            grid_calib=grid_calib_by_cam.get(cam_id),
        )
        print(f"  cam{cam_id}  N={pts.shape[0]:>7}  ({time.time() - t0:.1f}s)")
        if pts.shape[0] > 0:
            aahad_pts.append(pts)
            aahad_rgb.append(rgb)
            if sig is not None:
                aahad_sig.append(sig)
    if not aahad_pts:
        raise SystemExit("no AA-HAD points produced; aborting")
    aahad_xyz = np.concatenate(aahad_pts, axis=0)
    aahad_rgb_arr = np.concatenate(aahad_rgb, axis=0)

    # ---- 3. LiDAR static accumulate -----------------------------------
    print()
    print("[3/6] LiDAR static accumulate")
    lidar_static = _accumulate_lidar_static(
        loader, scene_id, cams=args.cams, bbox_expand=args.bbox_expand,
    )
    print(f"  N_lidar_static={lidar_static.shape[0]:>8}")

    # ---- 4. camera-FOV cull -------------------------------------------
    print()
    print("[4/6] camera-FOV cull")
    views = _build_camera_views(loader, scene_id, args.cams)
    fov_lidar = points_in_any_camera_fov(lidar_static, list(views.values()))
    fov_aahad = points_in_any_camera_fov(aahad_xyz, list(views.values()))
    lidar_static = lidar_static[fov_lidar]
    aahad_xyz = aahad_xyz[fov_aahad]
    aahad_rgb_arr = aahad_rgb_arr[fov_aahad]
    print(
        f"  N_lidar_in_fov={lidar_static.shape[0]:>8}  "
        f"N_aahad_in_fov={aahad_xyz.shape[0]:>8}"
    )

    # ---- 5. ground-height snap ----------------------------------------
    n_snapped = 0
    if args.ground_cell_size > 0 and lidar_static.shape[0] > 0 and aahad_xyz.shape[0] > 0:
        print()
        print(
            f"[5/6] ground-snap  cell={args.ground_cell_size}m  "
            f"max_dz={args.ground_snap_max_dz}m"
        )
        grid = build_ground_height_grid(
            lidar_static, cell_size=args.ground_cell_size,
            low_quantile=args.ground_low_quantile,
        )
        valid_cells = int(np.isfinite(grid.height).sum())
        total_cells = int(grid.height.size)
        print(f"  grid {grid.height.shape}  valid {valid_cells}/{total_cells}")
        aahad_xyz, snapped = snap_to_ground(
            aahad_xyz, grid, max_dz=args.ground_snap_max_dz,
        )
        n_snapped = int(snapped.sum())
        print(f"  {n_snapped} / {aahad_xyz.shape[0]} AA-HAD points snapped to ground")

    # ---- 6. LiDAR-priority fill ---------------------------------------
    print()
    print("[6/6] LiDAR-priority fill")
    fused_xyz, fused_rgb, source = lidar_priority_fill(
        lidar_static, aahad_xyz, aahad_rgb_arr,
        voxel_size=args.voxel_size,
        max_dist_to_lidar=args.max_dist_to_lidar,
    )
    n_lidar_fused = int((source == 0).sum())
    n_aahad_fused = int((source == 1).sum())
    print(
        f"  LiDAR backbone : {n_lidar_fused:>8}\n"
        f"  AA-HAD fill    : {n_aahad_fused:>8}\n"
        f"  total          : {fused_xyz.shape[0]:>8}"
    )

    # Voxel-downsample the fused cloud for visualisation parity with
    # the old static_scene_recon outputs.
    fused_v_xyz, fused_v_rgb = voxel_downsample(
        fused_xyz, args.voxel_size, colors=fused_rgb,
    )
    write_ply_xyz(
        out_dir / f"{scene_id}_completion.ply",
        fused_v_xyz, fused_v_rgb, binary=args.ply_binary,
    )

    # ---- 7. per-object snapshot injection (optional) ------------------
    inj_log: list[dict] = []
    n_dyn_total = 0
    if args.recon_dir:
        recon_dir = Path(args.recon_dir)
        if not recon_dir.is_dir():
            raise SystemExit(f"--recon-dir {recon_dir} not a directory")
        print()
        print(f"[+] dynamic snapshot injection from {recon_dir}")
        clouds = discover_object_clouds(recon_dir, prefer_icp=True)
        print(f"  found {len(clouds)} per-object PLY sets")
        dyn_xyz, dyn_rgb, inj_log = inject_object_snapshots(
            loader, scene_id, clouds,
            cam_for_discovery=args.cams[0],
            anchor_ts_ms=args.object_anchor_ts_ms,
            use_lidar=args.include_object_lidar,
        )
        n_dyn_total = int(dyn_xyz.shape[0])
        print(f"  injected {n_dyn_total} dynamic points across {len(inj_log)} objects")

        full_xyz = np.concatenate([fused_xyz, dyn_xyz], axis=0)
        full_rgb = np.concatenate([fused_rgb, dyn_rgb], axis=0) if fused_rgb is not None else None
        full_v_xyz, full_v_rgb = voxel_downsample(
            full_xyz, args.voxel_size, colors=full_rgb,
        )
        write_ply_xyz(
            out_dir / f"{scene_id}_completion_with_dyn.ply",
            full_v_xyz, full_v_rgb, binary=args.ply_binary,
        )

    summary = {
        "scene_id": scene_id,
        "voxel_size_m": args.voxel_size,
        "cams": args.cams,
        "n_lidar_in_fov": int(lidar_static.shape[0]),
        "n_aahad_in_fov": int(aahad_xyz.shape[0]),
        "n_aahad_ground_snapped": int(n_snapped),
        "n_lidar_priority": n_lidar_fused,
        "n_aahad_fill": n_aahad_fused,
        "n_fused_voxels": int(fused_v_xyz.shape[0]),
        "object_snapshots": inj_log,
        "n_dynamic_points": n_dyn_total,
        "args": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in vars(args).items()
        },
    }
    summary_path = out_dir / f"{scene_id}_completion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"  ↳ json : {summary_path}")
    print(f"  ↳ ply  : {out_dir}/{scene_id}_completion*.ply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
