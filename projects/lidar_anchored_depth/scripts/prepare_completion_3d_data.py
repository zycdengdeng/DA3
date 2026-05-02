"""Build 3D training samples for the point-cloud residual head — Stage 7A.

For each ``(scene, ts, cam)`` frame produces an ``.npz`` containing:

    prior_xyz   (N_p, 3) float32   — Layer 1-5 prior cloud (FOV-clipped)
    prior_rgb   (N_p, 3) uint8     — image-sampled colour per prior point
    prior_feat  (N_p, F) float32   — per-point conditioning features
                                     [source_label, d_tilde, aahad_init_z,
                                      dist_to_lidar (clipped at 5 m),
                                      sparse_lidar_present_flag]
    target_xyz  (N_t, 3) float32   — sparse LiDAR ground-truth (FOV-clipped)
    ts_ms, cam_id, scene_id        — metadata

The "Layer 1-5 prior" is built from the existing static-recon helpers:

    Layer 1: per-camera AA-HAD (a · d̃ + b)
    Layer 2: optional region-grid affine
    Layer 3: ground-snap to LiDAR ground grid
    Layer 4: LiDAR-priority voxel fusion
    Layer 5: skipped here; per-object snapshots are not part of the
             training prior. The 3D head learns refinement on the
             static layered prior only.

At training time the residual flow head predicts a per-point
displacement ``Δxyz`` that pulls each prior point toward the nearest
LiDAR target (when one exists within ``--target-search-radius``).

Usage
-----
    PYTHONPATH=src python scripts/prepare_completion_3d_data.py \\
        --data-root /mnt/car_road_data_TianJin --scene 008 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --calib-json preview/scene_recon/008_..._static_calib.json \\
        --output preview/completion_3d_data/ \\
        --aahad-pixel-stride 4
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
    world_to_image,
)
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.pipeline.ground_height_grid import (
    build_ground_height_grid,
    snap_to_ground,
)
from lidar_anchored_depth.pipeline.lidar_completion import (
    CameraView,
    points_in_camera_fov,
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


def _accumulate_static_lidar_for_scene(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cams: list[str],
    *,
    bbox_expand: float,
) -> np.ndarray:
    """Aggregate scene-static LiDAR over all timestamps once (used for
    the ground-height grid)."""
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


def _build_prior_for_frame(
    frame,
    d_image: np.ndarray,
    a: float,
    b: float,
    dyn_mask: np.ndarray,
    *,
    pixel_stride: int,
    z_min: float,
    z_max: float,
    ground_grid,
    ground_max_dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Apply Layer 1 (AA-HAD), Layer 3 (ground-snap) and FOV culling.

    Layer 4 (LiDAR-priority) is applied later, in the joint loop, since
    it's voxel-level fusion against the FRAME's LiDAR (sparse) — and we
    want the LiDAR points themselves to NOT appear in the prior cloud
    (they are the supervision target, not the prior).

    Returns
    -------
    prior_xyz : (N_p, 3) world XYZ for prior points
    prior_rgb : (N_p, 3) uint8 RGB
    prior_uv  : (N_p, 2) int32 source pixel coords (for feature lookup)
    """
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

    # Backproject to world.
    rays = pixel_to_camera_ray(uv, frame.K)
    P_cam = rays * z_keep[:, None]
    P_world = camera_to_world(P_cam, frame.T_wc)

    # Layer 3: ground snap.
    if ground_grid is not None:
        P_world_snapped, _ = snap_to_ground(
            P_world.astype(np.float64), ground_grid, max_dz=ground_max_dz,
        )
        P_world = P_world_snapped

    rgb = frame.image[rows, cols].astype(np.uint8)
    uv_int = np.stack([cols, rows], axis=1).astype(np.int32)
    return P_world.astype(np.float64), rgb, uv_int


def _project_lidar_to_image_sparse_map(
    lidar_world: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    distortion: np.ndarray | None,
    image_hw: tuple[int, int],
    *,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sparse_z_map (H, W) float32, sparse_mask (H, W) bool)``."""
    H, W = image_hw
    sparse_z = np.zeros((H, W), dtype=np.float32)
    sparse_mask = np.zeros((H, W), dtype=bool)
    if lidar_world.size == 0:
        return sparse_z, sparse_mask
    uv, z_cam, _ = world_to_image(lidar_world, K, T_wc, distortion)
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]; z_cam = z_cam[finite]
    if uv.size == 0:
        return sparse_z, sparse_mask
    uv_int = np.round(uv).astype(np.int64)
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    uv_int = uv_int[inside]; z_cam = z_cam[inside]
    keep = (z_cam >= z_min) & (z_cam <= z_max)
    uv_int = uv_int[keep]; z_cam = z_cam[keep]
    if uv_int.shape[0] == 0:
        return sparse_z, sparse_mask
    order = np.argsort(-z_cam)
    sparse_z[uv_int[order, 1], uv_int[order, 0]] = z_cam[order].astype(np.float32)
    sparse_mask[uv_int[order, 1], uv_int[order, 0]] = True
    return sparse_z, sparse_mask


def _build_one_sample(
    loader: RoadsideV2XLoader,
    scene_id: str,
    ts_ms: int,
    cam_id: str,
    a: float,
    b: float,
    d_path: Path,
    sam_dir: Path | None,
    cam_views: list[CameraView],
    ground_grid,
    *,
    bbox_expand: float,
    z_min: float,
    z_max: float,
    sam_dilate_px: int,
    pixel_stride: int,
    require_v2x: bool,
    require_target_pts: int,
    ground_max_dz: float,
    target_search_radius: float,
    sam_auto_dir: Path | None = None,
    ground_target_band_m: float = 0.4,
) -> dict | None:
    try:
        idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
    except ValueError:
        return None
    frame = loader.get_frame(idx)
    H, W = frame.image.shape[:2]
    if require_v2x and not (frame.dynamic_objects or []):
        return None

    d_data = np.load(d_path, allow_pickle=True)
    d_image = d_data["depth"].astype(np.float32)

    is_dyn_lidar = np.zeros(frame.lidar_world.shape[0], dtype=bool)
    for obj in frame.dynamic_objects or []:
        is_dyn_lidar |= points_in_oriented_bbox(
            frame.lidar_world, obj, expand=bbox_expand,
        )
    static_lidar = frame.lidar_world[~is_dyn_lidar]
    if static_lidar.shape[0] < require_target_pts:
        return None

    dyn_mask = load_sam_dynamic_mask(
        sam_dir, scene_id, ts_ms, cam_id, (H, W), dilate_px=sam_dilate_px,
    )

    prior = _build_prior_for_frame(
        frame, d_image, a, b, dyn_mask,
        pixel_stride=pixel_stride,
        z_min=z_min, z_max=z_max,
        ground_grid=ground_grid,
        ground_max_dz=ground_max_dz,
    )
    if prior is None:
        return None
    prior_xyz, prior_rgb, prior_uv = prior

    fov = np.zeros(prior_xyz.shape[0], dtype=bool)
    for v in cam_views:
        fov |= points_in_camera_fov(prior_xyz, v, z_min=z_min, z_max=z_max)
    if not fov.any():
        return None
    prior_xyz = prior_xyz[fov]
    prior_rgb = prior_rgb[fov]
    prior_uv = prior_uv[fov]
    if prior_xyz.shape[0] < 100:
        return None

    target_fov = np.zeros(static_lidar.shape[0], dtype=bool)
    for v in cam_views:
        target_fov |= points_in_camera_fov(static_lidar, v, z_min=z_min, z_max=z_max)
    target_xyz = static_lidar[target_fov]
    if target_xyz.shape[0] < require_target_pts:
        return None

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_xyz)
        dists, nn_idx = tree.query(prior_xyz, k=1)
        valid = dists <= float(target_search_radius)
        target_disp = np.zeros((prior_xyz.shape[0], 3), dtype=np.float32)
        target_disp[valid] = (
            target_xyz[nn_idx[valid]] - prior_xyz[valid]
        ).astype(np.float32)
    except ModuleNotFoundError:
        target_disp = np.zeros((prior_xyz.shape[0], 3), dtype=np.float32)
        valid = np.zeros(prior_xyz.shape[0], dtype=bool)
        for i in range(prior_xyz.shape[0]):
            d = np.linalg.norm(target_xyz - prior_xyz[i], axis=1)
            j = int(np.argmin(d))
            if d[j] <= target_search_radius:
                target_disp[i] = (target_xyz[j] - prior_xyz[i]).astype(np.float32)
                valid[i] = True
        dists = np.full(prior_xyz.shape[0], target_search_radius + 1.0, dtype=np.float32)

    # B1: ground-aware target rule. Points whose Z is within
    # ground_target_band_m of the LiDAR-derived ground grid are
    # treated as 'road' and assigned target_disp = 0 — the network
    # learns "identify ground -> output 0", and the road geometry is
    # locked to the LiDAR ground (cm-precise) at inference. valid
    # stays True for these points (we DO supervise them, just with
    # the deterministic zero target).
    n_ground = 0
    if ground_grid is not None and ground_target_band_m > 0:
        z_g, in_grid = ground_grid.query(prior_xyz[:, :2])
        near_ground = in_grid & (np.abs(prior_xyz[:, 2] - z_g) <= ground_target_band_m)
        target_disp[near_ground] = 0.0
        valid[near_ground] = True
        n_ground = int(near_ground.sum())

    valid_count = int(valid.sum())
    if valid_count < require_target_pts:
        return None

    rows, cols = prior_uv[:, 1], prior_uv[:, 0]
    d_at = d_image[rows, cols].astype(np.float32)
    aahad_init = (a * d_at + b).astype(np.float32)
    dist_to_lidar = np.minimum(dists, 5.0).astype(np.float32)
    sparse_present = (dists <= 0.10).astype(np.float32)
    source_label = np.zeros(prior_xyz.shape[0], dtype=np.float32)
    prior_feat = np.stack([
        source_label, d_at, aahad_init, dist_to_lidar, sparse_present,
    ], axis=1).astype(np.float32)

    # A1: per-point segment id from SAM Auto. -1 means "no SAM Auto
    # data for this frame"; the dataset/network will treat -1 as a
    # singleton segment (no cross-edge masking effect).
    if sam_auto_dir is not None:
        sa_path = sam_auto_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam_auto.npz"
        if sa_path.is_file():
            with np.load(sa_path) as sa:
                seg_image = sa["segment_id"]
            if seg_image.shape == (H, W):
                segment_id = seg_image[rows, cols].astype(np.int32)
            else:
                segment_id = np.full(prior_xyz.shape[0], -1, dtype=np.int32)
        else:
            segment_id = np.full(prior_xyz.shape[0], -1, dtype=np.int32)
    else:
        segment_id = np.full(prior_xyz.shape[0], -1, dtype=np.int32)

    return {
        "prior_xyz": prior_xyz.astype(np.float32),
        "prior_rgb": prior_rgb.astype(np.uint8),
        "prior_feat": prior_feat,
        "segment_id": segment_id,
        "target_xyz": target_xyz.astype(np.float32),
        "target_disp": target_disp,
        "valid_mask": valid,
        "n_ground_targets": n_ground,
        "scene_id": scene_id,
        "ts_ms": int(ts_ms),
        "cam_id": cam_id,
        "a": float(a),
        "b": float(b),
        "n_prior": int(prior_xyz.shape[0]),
        "n_target": int(target_xyz.shape[0]),
        "n_valid": valid_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--d-paths-glob", required=True)
    parser.add_argument("--sam-mask-dir", default=None)
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cams", nargs="+", default=["0", "3", "6", "9"])
    parser.add_argument("--bbox-expand", type=float, default=0.10)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=300.0)
    parser.add_argument("--sam-dilate-px", type=int, default=5)
    parser.add_argument("--require-v2x-frames", action="store_true", default=True)
    parser.add_argument(
        "--no-require-v2x-frames", action="store_false",
        dest="require_v2x_frames",
    )
    parser.add_argument("--aahad-pixel-stride", type=int, default=4)
    parser.add_argument("--require-target-pts", type=int, default=200)
    parser.add_argument("--ground-cell-size", type=float, default=0.30)
    parser.add_argument("--ground-low-quantile", type=float, default=0.10)
    parser.add_argument("--ground-snap-max-dz", type=float, default=0.4)
    parser.add_argument("--target-search-radius", type=float, default=2.0)
    parser.add_argument(
        "--sam-auto-dir", default=None,
        help="optional directory of SAM Auto npz files "
        "(output of run_sam_auto.py). When set, each prior point gets "
        "a segment_id from the SAM Auto segmentation, written into "
        "the npz alongside the other arrays. The training-time "
        "segment-aware EdgeConv uses this to mask cross-segment "
        "neighbours. When unset, all segment_ids = -1 (no masking).",
    )
    parser.add_argument(
        "--ground-target-band-m", type=float, default=0.4,
        help="B1 rule: prior points whose Z is within this band of "
        "the LiDAR ground grid are assigned target_disp = 0 instead "
        "of 'nearest LiDAR'. Locks road geometry to the cm-precise "
        "LiDAR ground at inference. 0 disables B1. Default 0.4 m.",
    )
    parser.add_argument("--loader-min-points", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = json.loads(Path(args.calib_json).read_text())
    print(f"[load] static calib for cams: {sorted(calib.keys())}")

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
    sam_auto_dir = Path(args.sam_auto_dir) if args.sam_auto_dir else None
    if sam_auto_dir is not None and not sam_auto_dir.is_dir():
        raise SystemExit(f"--sam-auto-dir {sam_auto_dir} not a directory")

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

    # Build the scene-wide ground grid once (Layer 3 input).
    print("[ground] accumulating static LiDAR for ground grid")
    static_lidar_full = _accumulate_static_lidar_for_scene(
        loader, scene_id, args.cams, bbox_expand=args.bbox_expand,
    )
    print(f"  N static={static_lidar_full.shape[0]}")
    ground_grid = build_ground_height_grid(
        static_lidar_full, cell_size=args.ground_cell_size,
        low_quantile=args.ground_low_quantile,
    )
    print(
        f"  grid {ground_grid.height.shape}  "
        f"valid {int(np.isfinite(ground_grid.height).sum())} / {ground_grid.height.size}"
    )

    # Build per-camera CameraView for FOV culling.
    cam_views: list[CameraView] = []
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
    print(f"[fov] {len(cam_views)} CameraViews built")

    print()
    n_written = 0
    n_skipped = 0
    t0 = time.time()
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
            sample = _build_one_sample(
                loader, scene_id, ts_ms, cam_id, a, b, d_path, sam_dir,
                cam_views, ground_grid,
                bbox_expand=args.bbox_expand,
                z_min=args.z_min, z_max=args.z_max,
                sam_dilate_px=args.sam_dilate_px,
                pixel_stride=args.aahad_pixel_stride,
                require_v2x=args.require_v2x_frames,
                require_target_pts=args.require_target_pts,
                ground_max_dz=args.ground_snap_max_dz,
                target_search_radius=args.target_search_radius,
                sam_auto_dir=sam_auto_dir,
                ground_target_band_m=args.ground_target_band_m,
            )
            if sample is None:
                n_skipped += 1
                continue
            out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_3d.npz"
            np.savez_compressed(out_path, **{
                k: v for k, v in sample.items() if k in {
                    "prior_xyz", "prior_rgb", "prior_feat", "segment_id",
                    "target_xyz", "target_disp", "valid_mask",
                }
            })
            n_written += 1
            cam_n += 1
        print(
            f"  cam{cam_id}  a={a:.2f}  b={b:.2f}  "
            f"wrote {cam_n} samples  ({time.time() - cam_t0:.1f}s)"
        )

    print()
    print(f"[done] {n_written} samples ({n_skipped} skipped) in {time.time() - t0:.1f}s")
    print(f"  ↳ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
