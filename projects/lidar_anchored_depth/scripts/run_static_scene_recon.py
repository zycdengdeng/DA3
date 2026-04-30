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
from lidar_anchored_depth.reconstruction import (
    depth_to_world_points,
    voxel_downsample,
    write_ply_xyz,
)


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


def _load_sam_dynamic_mask(
    sam_dir: Path | None,
    scene_id: str, ts_ms: int, cam_id: str,
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Build a (H, W) bool mask = union of every per-bbox SAM mask in
    the corresponding _sam.npz. ``True`` = dynamic-object pixel."""
    H, W = image_hw
    out = np.zeros((H, W), dtype=bool)
    if sam_dir is None:
        return out
    p = sam_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
    if not p.is_file():
        return out
    data = np.load(p, allow_pickle=True)
    masks = data["masks"]
    if masks.shape[0] == 0:
        return out
    return np.any(masks, axis=0)


def _calibrate_static_per_camera(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    z_min: float, z_max: float,
) -> dict | None:
    """Run B3-RANSAC on (d̃, z_lidar) pairs from this camera's STATIC
    LiDAR returns, joint over every timestamp. Returns ``None`` if
    fewer than ~100 valid pairs exist."""
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)

    d_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
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
        dyn_mask = _load_sam_dynamic_mask(
            sam_dir, scene_id, ts_ms, cam_id, (H, W)
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

    if not d_all:
        return None
    d_full = np.concatenate(d_all)
    z_full = np.concatenate(z_all)
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
        "n_static_lidar_total": int(n_static_total),
        "n_dynamic_lidar_total": int(n_dynamic_total),
    }


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
) -> tuple[np.ndarray, np.ndarray]:
    """Apply (a, b) to every (camera, ts) frame; unproject the STATIC
    portion (image \\ dynamic SAM mask) to world. Returns (P_world, RGB).
    """
    pts_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []
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

        z_cam = (a * d_image + b).astype(np.float32)
        # Static mask = whole image \ dynamic SAM mask, with stride decimation
        dyn = _load_sam_dynamic_mask(sam_dir, scene_id, ts_ms, cam_id, (H, W))
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

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pts_all), np.concatenate(rgb_all)


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

    # ---- 2. Per-camera unproject + accumulate static AA-HAD -----------
    print()
    print(f"[2/3] static-AA-HAD unprojection (stride={args.aahad_pixel_stride})")
    aahad_pts_all: list[np.ndarray] = []
    aahad_rgb_all: list[np.ndarray] = []
    for cam_id, info in cam_calib.items():
        t0 = time.time()
        pts, rgb = _unproject_static_branch(
            loader, scene_id, cam_id, info["a"], info["b"],
            d_paths_index, sam_dir,
            pixel_stride=args.aahad_pixel_stride,
            z_min=args.z_min, z_max=args.z_max,
        )
        elapsed = time.time() - t0
        print(f"  cam{cam_id}  N_aahad_static={pts.shape[0]:>7}  ({elapsed:.1f}s)")
        if pts.shape[0] > 0:
            aahad_pts_all.append(pts)
            aahad_rgb_all.append(rgb)

    aahad_pts = np.concatenate(aahad_pts_all) if aahad_pts_all else np.zeros((0, 3))
    aahad_rgb = np.concatenate(aahad_rgb_all) if aahad_rgb_all else np.zeros((0, 3), dtype=np.uint8)

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
    if aahad_pts.shape[0] > 0:
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
    if dynamic_lv.shape[0] > 0:
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
