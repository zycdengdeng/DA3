"""Static scene reconstruction — Stage 3H deliverable.

The full intersection point cloud comes from THREE sources:

  1. LiDAR (dynamic): every LiDAR point that fell inside any V2X bbox.
     Centimeter-precise but only on objects' surfaces. We keep these
     verbatim (they're the high-fidelity skeleton of cars / pedestrians
     / etc.).
  2. LiDAR (static): every LiDAR point NOT inside any bbox. Sparse far
     away but still ~5 cm precision. The "ground truth" for static
     scene geometry.
  3. AA-HAD (static): for each (camera, ts), the part of the image
     OUTSIDE every V2X bbox SAM mask is unprojected via DA3 metric
     depth (calibrated on static LiDAR points only). This is the
     DENSE COLORED layer that fills LiDAR's static gaps.

Per-camera (a, b) for the static branch is calibrated against the
static LiDAR pairs alone (no bbox-derived geometry — the static scene
has no V2X bboxes). We use B3 RANSAC affine on the (d̃, z_lidar) pairs
sampled at static-LiDAR projections, joint over all timestamps for one
camera.

Output: one large colored PLY per scene, plus per-camera (a, b) JSON.

Usage
-----
    PYTHONPATH=src python scripts/run_static_scene_recon.py \\
        --data-root /mnt/car_road_data_TianJin --scene 008 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --output preview/scene_recon/

    # decimate AA-HAD output by stride to keep PLY size small
    ... --aahad-pixel-stride 2

    # downsample voxel size (default 5cm)
    ... --voxel-size 0.05
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
from lidar_anchored_depth.alignment.global_scale import b3_ransac_affine
from lidar_anchored_depth.alignment.projection import world_to_image
from lidar_anchored_depth.data import (
    PINHOLE_CAMERA_IDS,
    RoadsideV2XLoader,
)
from lidar_anchored_depth.pipeline.region_static_calib import (
    GridStaticCalib,
    fit_grid_static_calib,
)
from lidar_anchored_depth.reconstruction import (
    depth_to_world_points,
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


def _filter_finite_in_image(uv, z, hw):
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z = z[finite]
    if uv.size == 0:
        return uv, z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    uv_int = np.round(uv).astype(np.int64)
    H, W = hw
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    return uv[inside], z[inside], uv_int[inside, 0], uv_int[inside, 1]


def _collect_static_pairs_per_camera(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    z_min: float, z_max: float,
    sam_dilate_px: int = 0,
    require_v2x_frames: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int] | None, dict]:
    """Walk every timestamp for one camera and return the static-LiDAR
    pairs ``(d_tilde, z_cam, uv)`` plus the image dims and counters.
    Used by both the single-(a, b) and grid calibration paths."""
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)

    d_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    uv_all: list[np.ndarray] = []
    image_hw: tuple[int, int] | None = None
    n_static_total = 0
    n_dynamic_total = 0
    for ts_ms in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            continue
        d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
        if d_path is None or not d_path.is_file():
            continue
        d_data = np.load(d_path, allow_pickle=True)
        d_image = d_data["depth"]

        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]
        if d_image.shape != (H, W):
            continue
        image_hw = (H, W)

        # Skip interpolated frames that have no V2X annotations: SAM
        # didn't run for them, so the dynamic-mask subtraction is a
        # no-op and any moving object visible in the image leaks into
        # the static branch / calibration.
        if require_v2x_frames and not (frame.dynamic_objects or []):
            continue

        # Build "static LiDAR" = LiDAR \ ⋃ V2X bbox interiors.
        is_dynamic = np.zeros(frame.lidar_world.shape[0], dtype=bool)
        for obj in frame.dynamic_objects or []:
            is_dynamic |= points_in_oriented_bbox(
                frame.lidar_world, obj, expand=0.10,
            )
        static_lidar = frame.lidar_world[~is_dynamic]
        n_static_total += int((~is_dynamic).sum())
        n_dynamic_total += int(is_dynamic.sum())
        if static_lidar.shape[0] < 100:
            continue

        # Project static LiDAR to image
        dist = frame.meta.get("distortion")
        uv, z_cam, _ = world_to_image(static_lidar, frame.K, frame.T_wc, dist)
        uv, z_cam, u, v = _filter_finite_in_image(uv, z_cam, (H, W))
        if u.size < 50:
            continue

        # ALSO drop pixels covered by the dynamic SAM mask (so we don't
        # accidentally calibrate against pixels that belong to a moving
        # object even if the LiDAR point itself is "static").
        dyn_mask = load_sam_dynamic_mask(
            sam_dir, scene_id, ts_ms, cam_id, (H, W),
            dilate_px=sam_dilate_px,
        )
        keep = ~dyn_mask[v, u]
        u = u[keep]
        v = v[keep]
        z_cam = z_cam[keep]
        if u.size < 50:
            continue

        d_at = d_image[v, u].astype(np.float64)
        valid = (
            np.isfinite(d_at) & (d_at > 0)
            & (z_cam >= z_min) & (z_cam <= z_max)
        )
        if int(valid.sum()) < 50:
            continue
        d_all.append(d_at[valid])
        z_all.append(z_cam[valid].astype(np.float64))
        uv_keep = np.stack([u[valid].astype(np.float64),
                            v[valid].astype(np.float64)], axis=1)
        uv_all.append(uv_keep)

    if not d_all:
        return (
            np.zeros(0), np.zeros(0), np.zeros((0, 2)), image_hw,
            {"n_static_lidar_total": int(n_static_total),
             "n_dynamic_lidar_total": int(n_dynamic_total)},
        )
    d_full = np.concatenate(d_all)
    z_full = np.concatenate(z_all)
    uv_full = np.concatenate(uv_all)
    counters = {
        "n_static_lidar_total": int(n_static_total),
        "n_dynamic_lidar_total": int(n_dynamic_total),
    }
    return d_full, z_full, uv_full, image_hw, counters


def _calibrate_static_per_camera(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    z_min: float, z_max: float,
    sam_dilate_px: int = 0,
    require_v2x_frames: bool = True,
) -> dict | None:
    """Run B3-RANSAC on (d̃, z_lidar) pairs from this camera's STATIC
    LiDAR returns, joint over every timestamp. Returns ``None`` if
    fewer than ~100 valid pairs exist."""
    d_full, z_full, _uv, _hw, counters = _collect_static_pairs_per_camera(
        loader, scene_id, cam_id, d_paths_index, sam_dir,
        z_min=z_min, z_max=z_max,
        sam_dilate_px=sam_dilate_px,
        require_v2x_frames=require_v2x_frames,
    )
    if d_full.size < 100:
        return None
    fit = b3_ransac_affine(
        d_full, z_full,
        threshold_rel=0.10, threshold_abs=1.0, n_iterations=300, seed=0,
    )
    return {
        "a": float(fit.a), "b": float(fit.b),
        "rmse_inliers_m": float(fit.residual_rmse),
        "n_pairs": int(d_full.size),
        "n_static_lidar_total": int(counters["n_static_lidar_total"]),
        "n_dynamic_lidar_total": int(counters["n_dynamic_lidar_total"]),
    }


def _calibrate_grid_per_camera(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    z_min: float, z_max: float,
    sam_dilate_px: int,
    require_v2x_frames: bool,
    n_rows: int,
    n_cols: int,
    min_lidar_per_cell: int,
) -> tuple[GridStaticCalib | None, dict]:
    d_full, z_full, uv_full, image_hw, counters = (
        _collect_static_pairs_per_camera(
            loader, scene_id, cam_id, d_paths_index, sam_dir,
            z_min=z_min, z_max=z_max,
            sam_dilate_px=sam_dilate_px,
            require_v2x_frames=require_v2x_frames,
        )
    )
    if d_full.size < 100 or image_hw is None:
        return None, counters
    calib = fit_grid_static_calib(
        d_full, z_full, uv_full, image_hw,
        n_rows=n_rows, n_cols=n_cols,
        min_lidar_per_cell=min_lidar_per_cell,
    )
    counters["n_pairs"] = int(d_full.size)
    return calib, counters


def _unproject_static_branch(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    a: float, b: float,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    pixel_stride: int,
    z_min: float, z_max: float,
    sigma_predictor=None,
    sam_dilate_px: int = 0,
    require_v2x_frames: bool = True,
    grid_calib: GridStaticCalib | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply (a, b) to every (camera, ts) frame; unproject the STATIC
    portion (image \\ dynamic SAM mask) to world.

    Returns ``(P_world, RGB, sigma_per_point)``; ``sigma_per_point`` is
    ``None`` when ``sigma_predictor`` is not provided.
    """
    pts_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []
    sig_all: list[np.ndarray] = []
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    for ts_ms in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            continue
        d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
        if d_path is None or not d_path.is_file():
            continue
        d_data = np.load(d_path, allow_pickle=True)
        d_image = d_data["depth"]
        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]
        if d_image.shape != (H, W):
            continue
        if require_v2x_frames and not (frame.dynamic_objects or []):
            continue

        if grid_calib is not None:
            z_cam = grid_calib.apply(d_image).astype(np.float32)
        else:
            z_cam = (a * d_image + b).astype(np.float32)
        # Static mask = whole image \ dynamic SAM mask, with stride decimation
        dyn = load_sam_dynamic_mask(
            sam_dir, scene_id, ts_ms, cam_id, (H, W),
            dilate_px=sam_dilate_px,
        )
        static_mask = ~dyn
        if pixel_stride > 1:
            stride_mask = np.zeros((H, W), dtype=bool)
            stride_mask[::pixel_stride, ::pixel_stride] = True
            static_mask &= stride_mask

        if not static_mask.any():
            continue

        P_world, colors = depth_to_world_points(
            depth=z_cam, image_rgb=frame.image,
            K=frame.K, T_wc=frame.T_wc,
            mask=static_mask, z_min=z_min, z_max=z_max,
        )
        if P_world.shape[0] == 0:
            continue
        pts_all.append(P_world)
        rgb_all.append(colors)

        if sigma_predictor is not None:
            # Reproduce the same kept-pixel set the unprojector used so
            # rows align 1:1 with P_world.
            valid = (z_cam >= z_min) & (z_cam <= z_max) & np.isfinite(z_cam) & static_mask
            rows, cols = np.where(valid)
            uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
            d_at = d_image[rows, cols].astype(np.float64)
            # When a grid calib is in use, the sigma head sees the
            # camera-mean (a, b) as its 8th feature (the head was
            # trained on a single per-camera scalar). The z_pred
            # feature it computes will diverge slightly from the
            # grid-applied z_cam used for unprojection. Re-train the
            # head with grid-aware features for tighter calibration.
            sa, sb = a, b
            if grid_calib is not None and grid_calib.fallback() is not None:
                sa = float(grid_calib.fallback().a)
                sb = float(grid_calib.fallback().b)
            sigma = sigma_predictor.predict_for_pixels(
                uv, d_at, frame.K, a=sa, b=sb, image_hw=(H, W),
            )
            sig_all.append(sigma)

    if not pts_all:
        empty_sig = np.zeros(0, dtype=np.float32) if sigma_predictor is not None else None
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
            empty_sig,
        )
    pts = np.concatenate(pts_all)
    rgb = np.concatenate(rgb_all)
    sig = np.concatenate(sig_all) if sigma_predictor is not None else None
    return pts, rgb, sig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--d-paths-glob", required=True,
        help="single-quoted glob for the .npz files written by "
        "run_da3_inference.py, e.g. 'preview/da3/008_*_d.npz'",
    )
    parser.add_argument(
        "--sam-mask-dir", default=None,
        help="directory with SAM .npz files. When given, dynamic-object "
        "SAM masks are subtracted from the static branch (recommended; "
        "without it, dynamic objects leak into the static AA-HAD output).",
    )
    parser.add_argument(
        "--cams", nargs="+", default=["0", "3", "6", "9"],
        help="pinhole cam ids to use (default: all 4)",
    )
    parser.add_argument(
        "--aahad-pixel-stride", type=int, default=2,
        help="decimate AA-HAD static unprojection by this stride to keep "
        "the output PLY manageable (default 2 → 1/4 of pixels)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05,
        help="voxel-grid downsample size (m) for the final aggregate "
        "(default 5 cm)",
    )
    parser.add_argument(
        "--include-dynamic-lidar", action="store_true", default=True,
        help="include LiDAR points that fall in V2X bboxes (default ON; "
        "this is the dynamic-object skeleton in the final cloud)",
    )
    parser.add_argument(
        "--no-include-dynamic-lidar", action="store_false",
        dest="include_dynamic_lidar",
    )
    parser.add_argument(
        "--include-static-lidar", action="store_true", default=True,
        help="include LiDAR points OUTSIDE V2X bboxes (default ON)",
    )
    parser.add_argument(
        "--no-include-static-lidar", action="store_false",
        dest="include_static_lidar",
    )
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=300.0)
    parser.add_argument("--ply-binary", action="store_true")
    parser.add_argument(
        "--loader-min-points", type=int, default=0,
        help="V2X annotation filter (default 0 → permissive)",
    )
    parser.add_argument(
        "--sam-dilate-px", type=int, default=0,
        help="dilate the union of dynamic SAM masks by this many pixels "
        "before subtracting it from the static branch. DA3's depth "
        "predictions tend to bleed past the SAM mask edge by 3-10 px "
        "(soft transition vs SAM's hard cut), so a small dilation "
        "(e.g. 5-10) catches the residual ghost halo. Default 0.",
    )
    parser.add_argument(
        "--require-v2x-frames", action="store_true", default=True,
        help="skip frames that have no V2X bbox annotations (i.e. "
        "interpolated frames between V2X timestamps). Those frames "
        "have no SAM masks generated, so any moving object visible in "
        "them leaks into the static branch as a 'ghost'. Default ON.",
    )
    parser.add_argument(
        "--no-require-v2x-frames", action="store_false",
        dest="require_v2x_frames",
    )
    parser.add_argument(
        "--hybrid-include-dynamic-lidar", action="store_true", default=False,
        help="include the dynamic-LiDAR cloud (LiDAR points inside V2X "
        "bboxes) in <scene>_hybrid.ply. Default OFF — accumulating "
        "every dynamic point across the whole session smears each "
        "vehicle into a long trail and dominates the figure. The "
        "<scene>_lidar_dynamic.ply file is still written separately.",
    )
    parser.add_argument(
        "--region-grid", nargs=2, type=int, default=[0, 0],
        metavar=("ROWS", "COLS"),
        help="if set to ROWS COLS > 0, replace the single per-camera "
        "(a, b) with a per-image-cell affine fit. Each cell runs B3-"
        "RANSAC on the static-LiDAR pairs that project into it; cells "
        "with too few pairs fall back to the global per-camera (a, b). "
        "Recommended: '6 8' (6 rows × 8 cols) for 1080p frames. "
        "Default '0 0' = disabled.",
    )
    parser.add_argument(
        "--region-min-lidar-per-cell", type=int, default=30,
        help="minimum static-LiDAR samples a grid cell needs to fit "
        "its own (a, b); cells below this threshold fall back to the "
        "global fit (default 30).",
    )
    parser.add_argument(
        "--sigma-checkpoint", default=None,
        help="optional .pt produced by scripts/train_sigma_head.py. "
        "If given, the static AA-HAD branch is fused across cameras "
        "with inverse-variance weights (w = 1/sigma^2) instead of the "
        "default unweighted voxel mean. Requires torch.",
    )
    parser.add_argument(
        "--sigma-device", default="cpu",
        help="device for the sigma predictor (default cpu; recon is "
        "I/O-bound so cuda is rarely worth the transfer cost)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build d_paths index keyed by (scene_id, ts_ms, cam_id)
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
    print(f"[load] {len(d_paths_index)} d̃ frames indexed")

    sam_dir = Path(args.sam_mask_dir) if args.sam_mask_dir else None
    if sam_dir is not None and not sam_dir.is_dir():
        raise SystemExit(f"--sam-mask-dir {sam_dir} not a directory")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    print(f"[scene] {scene_id}  ts_count={len(scene.timestamps_ms)}  cams={args.cams}")
    print()

    # ---- 1. Per-camera static (a, b) calibration ----------------------
    print("[1/3] static-LiDAR calibration per camera")
    fmt = "  cam{:<3}  N pairs={:>6}  static={:>7}  dynamic={:>7}  a={:>7.3f}  b={:>7.3f}  rmse={:>5.2f}m"
    cam_calib: dict[str, dict] = {}
    for cam_id in args.cams:
        info = _calibrate_static_per_camera(
            loader, scene_id, cam_id, d_paths_index, sam_dir,
            z_min=args.z_min, z_max=args.z_max,
            sam_dilate_px=args.sam_dilate_px,
            require_v2x_frames=args.require_v2x_frames,
        )
        if info is None:
            print(f"  cam{cam_id}: too few static LiDAR pairs, skipping")
            continue
        cam_calib[cam_id] = info
        print(fmt.format(
            cam_id, info["n_pairs"],
            info["n_static_lidar_total"], info["n_dynamic_lidar_total"],
            info["a"], info["b"], info["rmse_inliers_m"],
        ))
    if not cam_calib:
        raise SystemExit("no camera could be calibrated; aborting")

    calib_json = out_dir / f"{scene_id}_static_calib.json"
    calib_json.write_text(json.dumps(cam_calib, indent=2))

    # Optional: per-image-cell calibration on top of the per-camera
    # baseline. Fixes the near-camera curvature that a single (a, b)
    # cannot express.
    n_rows, n_cols = args.region_grid
    grid_calib_by_cam: dict[str, GridStaticCalib] = {}
    if n_rows > 0 and n_cols > 0:
        print()
        print(f"[1b/3] grid calibration  rows={n_rows}  cols={n_cols}")
        gfmt = "  cam{:<3}  cells solved/total = {:>3}/{:>3}  fallback rmse={:>5.2f}m"
        grid_serial: dict[str, dict] = {}
        for cam_id in cam_calib.keys():
            calib, _ = _calibrate_grid_per_camera(
                loader, scene_id, cam_id, d_paths_index, sam_dir,
                z_min=args.z_min, z_max=args.z_max,
                sam_dilate_px=args.sam_dilate_px,
                require_v2x_frames=args.require_v2x_frames,
                n_rows=n_rows, n_cols=n_cols,
                min_lidar_per_cell=args.region_min_lidar_per_cell,
            )
            if calib is None:
                print(f"  cam{cam_id}: grid calib failed, falling back to scalar (a, b)")
                continue
            grid_calib_by_cam[cam_id] = calib
            grid_serial[cam_id] = calib.to_serialisable()
            fallback_rmse = (
                calib.fallback().residual_rmse if calib.fallback() is not None else float("nan")
            )
            print(gfmt.format(
                cam_id, calib.n_cells_solved, calib.n_cells_total, fallback_rmse,
            ))
        if grid_serial:
            grid_json = out_dir / f"{scene_id}_static_calib_grid.json"
            grid_json.write_text(json.dumps(grid_serial, indent=2))

    # ---- 2. Per-camera unproject + accumulate static AA-HAD -----------
    sigma_predictor = None
    if args.sigma_checkpoint:
        from lidar_anchored_depth.models import SigmaPredictor

        sigma_predictor = SigmaPredictor(args.sigma_checkpoint, device=args.sigma_device)
        print()
        print(f"[sigma] loaded {args.sigma_checkpoint} on {args.sigma_device}")

    print()
    print(f"[2/3] static-AA-HAD unprojection (stride={args.aahad_pixel_stride})")
    aahad_pts_all: list[np.ndarray] = []
    aahad_rgb_all: list[np.ndarray] = []
    aahad_sig_all: list[np.ndarray] = []
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
        elapsed = time.time() - t0
        if sig is not None and sig.size > 0:
            print(
                f"  cam{cam_id}  N_aahad_static={pts.shape[0]:>7}  "
                f"sigma p50={float(np.median(sig)):.2f}m  "
                f"p90={float(np.quantile(sig, 0.9)):.2f}m  ({elapsed:.1f}s)"
            )
        else:
            print(f"  cam{cam_id}  N_aahad_static={pts.shape[0]:>7}  ({elapsed:.1f}s)")
        if pts.shape[0] > 0:
            aahad_pts_all.append(pts)
            aahad_rgb_all.append(rgb)
            if sig is not None:
                aahad_sig_all.append(sig)

    aahad_pts = np.concatenate(aahad_pts_all) if aahad_pts_all else np.zeros((0, 3))
    aahad_rgb = np.concatenate(aahad_rgb_all) if aahad_rgb_all else np.zeros((0, 3), dtype=np.uint8)
    aahad_sig = np.concatenate(aahad_sig_all) if aahad_sig_all else None

    # ---- 3. LiDAR static + LiDAR dynamic accumulate -------------------
    print()
    print(f"[3/3] LiDAR static + dynamic accumulation")
    lidar_static_pts: list[np.ndarray] = []
    lidar_dynamic_pts: list[np.ndarray] = []
    for ts_ms in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, args.cams[0])
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        is_dyn = np.zeros(frame.lidar_world.shape[0], dtype=bool)
        for obj in frame.dynamic_objects or []:
            is_dyn |= points_in_oriented_bbox(
                frame.lidar_world, obj, expand=0.10,
            )
        if args.include_static_lidar:
            lidar_static_pts.append(frame.lidar_world[~is_dyn])
        if args.include_dynamic_lidar:
            lidar_dynamic_pts.append(frame.lidar_world[is_dyn])

    static_lidar = np.concatenate(lidar_static_pts) if lidar_static_pts else np.zeros((0, 3))
    dynamic_lidar = np.concatenate(lidar_dynamic_pts) if lidar_dynamic_pts else np.zeros((0, 3))
    print(f"  N_lidar_static={static_lidar.shape[0]:>8}  N_lidar_dynamic={dynamic_lidar.shape[0]:>7}")

    # ---- Merge + voxel downsample, write PLYs --------------------------
    print()
    print(f"[merge] voxel size = {args.voxel_size} m")
    aahad_voxel_sig = None
    if aahad_pts.shape[0] > 0:
        if aahad_sig is not None:
            from lidar_anchored_depth.models import sigma_weighted_voxel

            aahad_v, aahad_c, aahad_voxel_sig = sigma_weighted_voxel(
                aahad_pts, aahad_sig, args.voxel_size, colors=aahad_rgb,
            )
            print(
                f"  ↳ sigma-weighted fusion: "
                f"voxel sigma p50={float(np.median(aahad_voxel_sig)):.3f}m  "
                f"p90={float(np.quantile(aahad_voxel_sig, 0.9)):.3f}m"
            )
        else:
            aahad_v, aahad_c = voxel_downsample(
                aahad_pts, args.voxel_size, colors=aahad_rgb,
            )
    else:
        aahad_v = aahad_pts
        aahad_c = aahad_rgb
    if static_lidar.shape[0] > 0:
        static_lv, _ = voxel_downsample(static_lidar, args.voxel_size)
    else:
        static_lv = static_lidar
    if dynamic_lidar.shape[0] > 0:
        dynamic_lv, _ = voxel_downsample(dynamic_lidar, args.voxel_size)
    else:
        dynamic_lv = dynamic_lidar

    print(
        f"  ↳ N(aahad_static={aahad_v.shape[0]}, "
        f"lidar_static={static_lv.shape[0]}, "
        f"lidar_dynamic={dynamic_lv.shape[0]})"
    )

    # Write three PLYs (so reviewer can toggle layers in the viewer)
    if aahad_v.shape[0] > 0:
        write_ply_xyz(
            out_dir / f"{scene_id}_aahad_static.ply",
            aahad_v, aahad_c, binary=args.ply_binary,
        )
    if static_lv.shape[0] > 0:
        write_ply_xyz(
            out_dir / f"{scene_id}_lidar_static.ply",
            static_lv, binary=args.ply_binary,
        )
    if dynamic_lv.shape[0] > 0:
        write_ply_xyz(
            out_dir / f"{scene_id}_lidar_dynamic.ply",
            dynamic_lv, binary=args.ply_binary,
        )

    # Hybrid (AA-HAD + LiDAR) for the headline figure: prefer LiDAR
    # where it's dense (static voxels with LiDAR present already),
    # fall back to AA-HAD elsewhere. We approximate this by stacking
    # both and voxel-downsampling jointly with a slight LiDAR
    # preference (LiDAR points get weight 4× by replication).
    hybrid_pts = []
    hybrid_rgb = []
    if aahad_v.shape[0] > 0:
        hybrid_pts.append(aahad_v)
        hybrid_rgb.append(aahad_c)
    if static_lv.shape[0] > 0:
        # Fake "color" = grey for LiDAR (gets voxel-merged with AA-HAD
        # colors weighted equally per point; LiDAR-dense voxels will
        # average toward grey).
        hybrid_pts.append(static_lv)
        hybrid_rgb.append(np.full((static_lv.shape[0], 3), 180, dtype=np.uint8))
    if dynamic_lv.shape[0] > 0 and args.hybrid_include_dynamic_lidar:
        hybrid_pts.append(dynamic_lv)
        hybrid_rgb.append(np.full((dynamic_lv.shape[0], 3), 220, dtype=np.uint8))
    if hybrid_pts:
        h_pts = np.concatenate(hybrid_pts)
        h_rgb = np.concatenate(hybrid_rgb)
        h_v, h_c = voxel_downsample(h_pts, args.voxel_size, colors=h_rgb)
        write_ply_xyz(
            out_dir / f"{scene_id}_hybrid.ply",
            h_v, h_c, binary=args.ply_binary,
        )
        print(f"  ↳ N(hybrid)={h_v.shape[0]}")

    print()
    print(f"  ↳ json     : {calib_json}")
    print(f"  ↳ ply      : {out_dir}/{scene_id}_*.ply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
