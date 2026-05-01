"""Build training samples for the residual completion network — Stage 6C.

For each ``(scene, ts, cam)`` frame produces an ``.npz`` with the
7-channel conditioning + sparse target the U-Net needs:

    rgb               (H, W, 3) uint8
    d_tilde           (H, W)    float32
    aahad_init        (H, W)    float32     = a · d̃ + b
    sparse_lidar_z    (H, W)    float32     = projected static-LiDAR z (0 = miss)
    sparse_lidar_mask (H, W)    bool
    target_dz         (H, W)    float32     = z_lidar - aahad_init at LiDAR (0 else)
    valid_mask        (H, W)    bool

Usage
-----
    PYTHONPATH=src python scripts/prepare_completion_data.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --calib-json preview/scene_recon/008_..._static_calib.json \\
        --output preview/completion_data/

Each output file is named ``<scene>_ts<ts>_cam<cam>_train.npz``.
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
from lidar_anchored_depth.alignment.projection import world_to_image
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.segmentation.sam_io import load_sam_dynamic_mask


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _build_one_sample(
    loader: RoadsideV2XLoader,
    scene_id: str,
    ts_ms: int,
    cam_id: str,
    a: float, b: float,
    d_path: Path,
    sam_dir: Path | None,
    *,
    bbox_expand: float,
    z_min: float, z_max: float,
    sam_dilate_px: int,
    require_v2x: bool,
    require_lidar_pixels: int,
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
    if d_image.shape != (H, W):
        return None

    # Static LiDAR = LiDAR \ V2X bboxes.
    is_dyn = np.zeros(frame.lidar_world.shape[0], dtype=bool)
    for obj in frame.dynamic_objects or []:
        is_dyn |= points_in_oriented_bbox(
            frame.lidar_world, obj, expand=bbox_expand,
        )
    static_lidar = frame.lidar_world[~is_dyn]
    if static_lidar.shape[0] < 100:
        return None

    dist = frame.meta.get("distortion")
    uv, z_cam, _ = world_to_image(static_lidar, frame.K, frame.T_wc, dist)
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]; z_cam = z_cam[finite]
    if uv.size == 0:
        return None
    uv_int = np.round(uv).astype(np.int64)
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    uv_int = uv_int[inside]; z_cam = z_cam[inside]
    if uv_int.shape[0] < require_lidar_pixels:
        return None

    # Drop pixels covered by the dynamic SAM mask (we don't want to
    # train on moving objects).
    dyn_mask = load_sam_dynamic_mask(
        sam_dir, scene_id, ts_ms, cam_id, (H, W),
        dilate_px=sam_dilate_px,
    )
    keep = ~dyn_mask[uv_int[:, 1], uv_int[:, 0]]
    uv_int = uv_int[keep]; z_cam = z_cam[keep]
    in_range = (z_cam >= z_min) & (z_cam <= z_max)
    uv_int = uv_int[in_range]; z_cam = z_cam[in_range]
    if uv_int.shape[0] < require_lidar_pixels:
        return None

    # Build sparse maps. Multiple LiDAR points can fall on the same
    # pixel; we take the closest (smallest z) to be consistent with
    # an opaque-surface assumption.
    sparse_z = np.zeros((H, W), dtype=np.float32)
    sparse_mask = np.zeros((H, W), dtype=bool)
    # Sort by z descending so smaller z overwrites later (= ends up
    # in the buffer).
    order = np.argsort(-z_cam)
    uv_sorted = uv_int[order]
    z_sorted = z_cam[order]
    sparse_z[uv_sorted[:, 1], uv_sorted[:, 0]] = z_sorted.astype(np.float32)
    sparse_mask[uv_sorted[:, 1], uv_sorted[:, 0]] = True

    # AA-HAD initial z over the whole image.
    aahad_init = (a * d_image + b).astype(np.float32)

    # Per-pixel target Δz at LiDAR pixels; 0 elsewhere (network learns
    # "trust AA-HAD" outside LiDAR via the masked CFM loss).
    target_dz = np.zeros((H, W), dtype=np.float32)
    target_dz[sparse_mask] = (sparse_z[sparse_mask] - aahad_init[sparse_mask]).astype(np.float32)

    return {
        "rgb": frame.image.astype(np.uint8),
        "d_tilde": d_image,
        "aahad_init": aahad_init,
        "sparse_lidar_z": sparse_z,
        "sparse_lidar_mask": sparse_mask,
        "target_dz": target_dz,
        "valid_mask": sparse_mask,
        "scene_id": scene_id,
        "ts_ms": int(ts_ms),
        "cam_id": cam_id,
        "a": float(a),
        "b": float(b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--d-paths-glob", required=True)
    parser.add_argument("--sam-mask-dir", default=None)
    parser.add_argument(
        "--calib-json", required=True,
        help="JSON written by run_static_scene_recon.py (per-camera a, b)",
    )
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
    parser.add_argument("--require-lidar-pixels", type=int, default=200)
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
    print()

    n_written = 0
    n_skipped = 0
    t_total = time.time()
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
                bbox_expand=args.bbox_expand,
                z_min=args.z_min, z_max=args.z_max,
                sam_dilate_px=args.sam_dilate_px,
                require_v2x=args.require_v2x_frames,
                require_lidar_pixels=args.require_lidar_pixels,
            )
            if sample is None:
                n_skipped += 1
                continue
            out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_train.npz"
            np.savez_compressed(out_path, **{
                k: v for k, v in sample.items()
                if k in {
                    "rgb", "d_tilde", "aahad_init", "sparse_lidar_z",
                    "sparse_lidar_mask", "target_dz", "valid_mask",
                }
            })
            n_written += 1
            cam_n += 1
        print(
            f"  cam{cam_id}  a={a:.2f}  b={b:.2f}  "
            f"wrote {cam_n} samples  ({time.time() - cam_t0:.1f}s)"
        )

    print()
    print(f"[done] {n_written} samples ({n_skipped} skipped) in {time.time() - t_total:.1f}s")
    print(f"  ↳ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
