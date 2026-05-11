"""Per-frame depth-completion primitives + multi-GPU sharding.

These functions are the building blocks of the ``complete`` stage —
they take one ``(scene, cam, ts)`` triple and a calibration, produce
a Layer 1+3 prior cloud, and (optionally) hand it to the trained
:class:`PointResidualPredictor` for Δxyz refinement.

The :func:`run_refine_parallel` wrapper spawns one worker per GPU
listed in the runtime config; each worker pins itself to a single
card via ``CUDA_VISIBLE_DEVICES`` before its first torch import.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import points_in_oriented_bbox
from lidar_anchored_depth.alignment.projection import (
    camera_to_world,
    pixel_to_camera_ray,
)
from lidar_anchored_depth.pipeline.ground_height_grid import snap_to_ground
from lidar_anchored_depth.pipeline.lidar_completion import (
    points_in_any_camera_fov,
)
from lidar_anchored_depth.segmentation.sam_io import load_sam_dynamic_mask


def accumulate_static_lidar(
    loader, scene_id: str, cams, *, bbox_expand: float,
) -> np.ndarray:
    """Collect per-ts static (= V2X-bbox-excluded) LiDAR for ground grid."""
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


def build_prior_for_frame(
    frame, d_image, a, b, dyn_mask,
    *, pixel_stride, z_min, z_max, ground_grid, ground_max_dz,
):
    """Layer 1+3 prior: depth → world XYZ with AA-HAD linear calib +
    SAM dynamic-mask subtraction + ground-snap.

    Returns ``(P_world, rgb, uv_int)`` or ``None`` if no valid pixels.
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
    valid = (
        (z_cam >= z_min) & (z_cam <= z_max)
        & np.isfinite(z_cam) & static_mask
    )
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


def refine_one_frame(
    scene_id: str,
    cam_id: str,
    ts_ms: int,
    *,
    loader,
    predictor,
    build_prior_features_fn,
    calib: dict,
    d_paths_index: dict,
    ground_grid,
    cam_views: list,
    sam_dir: Path | None,
    sam_auto_dir: Path | None,
    segformer_dir: Path | None,
    args,
) -> dict | None:
    """Run the per-(scene, cam, ts) prior build + network refinement.

    Returns ``dict(refined_xyz, prior_rgb, prior_xyz, cam_id, ts_ms)``
    on success or ``None`` if the frame should be skipped.

    A deterministic seed derived from ``(scene_id, cam_id, ts_ms)`` is
    applied before the random subsample and the CFM noise draw so
    splitting work across GPUs gives bit-for-bit identical artefacts
    to a single-GPU run for the same triple.
    """
    import torch  # local: only after CUDA_VISIBLE_DEVICES is set in workers

    seed = abs(hash((scene_id, cam_id, int(ts_ms)))) % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cam_id not in calib:
        return None
    a = float(calib[cam_id]["a"])
    b = float(calib[cam_id]["b"])

    d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
    if d_path is None or not Path(d_path).is_file():
        return None
    try:
        idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
    except ValueError:
        return None
    frame = loader.get_frame(idx)
    if args.require_v2x_frames and not (frame.dynamic_objects or []):
        return None

    d_data = np.load(d_path, allow_pickle=True)
    d_image = d_data["depth"].astype(np.float32)

    dyn_mask = load_sam_dynamic_mask(
        sam_dir, scene_id, ts_ms, cam_id, frame.image.shape[:2],
        dilate_px=args.sam_dilate_px,
    )
    prior = build_prior_for_frame(
        frame, d_image, a, b, dyn_mask,
        pixel_stride=args.aahad_pixel_stride,
        z_min=args.z_min, z_max=args.z_max,
        ground_grid=ground_grid,
        ground_max_dz=args.ground_snap_max_dz,
    )
    if prior is None:
        return None
    prior_xyz, prior_rgb, prior_uv = prior

    fov = np.zeros(prior_xyz.shape[0], dtype=bool)
    for v in cam_views:
        fov |= points_in_any_camera_fov(
            prior_xyz, [v], z_min=args.z_min, z_max=args.z_max,
        )
    if not fov.any():
        return None
    prior_xyz = prior_xyz[fov]
    prior_rgb = prior_rgb[fov]
    prior_uv = prior_uv[fov]

    if prior_xyz.shape[0] > args.max_points_per_frame:
        sel = np.random.choice(
            prior_xyz.shape[0],
            size=args.max_points_per_frame,
            replace=False,
        )
        prior_xyz = prior_xyz[sel]
        prior_rgb = prior_rgb[sel]
        prior_uv = prior_uv[sel]
    if prior_xyz.shape[0] < 100:
        return None

    is_dyn = np.zeros(frame.lidar_world.shape[0], dtype=bool)
    for obj in frame.dynamic_objects or []:
        is_dyn |= points_in_oriented_bbox(
            frame.lidar_world, obj, expand=args.bbox_expand,
        )
    target_lidar = frame.lidar_world[~is_dyn]

    feat = build_prior_features_fn(
        prior_xyz, prior_uv, d_image, a, b, target_lidar,
    )
    dist_to_lidar = feat[:, 3]

    seg_per_pt = None
    H_img, W_img = frame.image.shape[:2]
    if segformer_dir is not None:
        sf_path = (
            segformer_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_seg.npz"
        )
        if sf_path.is_file():
            with np.load(sf_path) as sf:
                cls_image = sf["class_id"]
            if cls_image.shape == (H_img, W_img):
                seg_per_pt = cls_image[
                    prior_uv[:, 1], prior_uv[:, 0],
                ].astype(np.int64)
    elif sam_auto_dir is not None:
        sa_path = (
            sam_auto_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam_auto.npz"
        )
        if sa_path.is_file():
            with np.load(sa_path) as sa:
                seg_image = sa["segment_id"]
            if seg_image.shape == (H_img, W_img):
                seg_per_pt = seg_image[
                    prior_uv[:, 1], prior_uv[:, 0],
                ].astype(np.int64)

    refined_xyz = predictor.refine_cloud(
        prior_xyz, prior_rgb, feat,
        dist_to_lidar=dist_to_lidar,
        segment_id=seg_per_pt,
    )

    if args.post_network_ground_snap:
        refined_xyz_snapped, _ = snap_to_ground(
            refined_xyz.astype(np.float64), ground_grid,
            max_dz=args.ground_snap_max_dz,
        )
        refined_xyz = refined_xyz_snapped.astype(np.float32)

    return {
        "cam_id": cam_id,
        "ts_ms": int(ts_ms),
        "refined_xyz": refined_xyz,
        "prior_rgb": prior_rgb,
        "prior_xyz": prior_xyz.astype(np.float32),
    }


# ---- Multi-GPU worker plumbing --------------------------------------
#
# The parent process intentionally avoids importing torch / loading
# the predictor when sharding is requested, because each worker needs
# to set CUDA_VISIBLE_DEVICES *before* its first torch import so that
# ``cuda:0`` resolves to its own card. Worker state is held in a
# module-level dict because ``multiprocessing.Pool`` initialisers can
# only stash via globals.

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    gpu_q,
    args_dict,
    scene_id,
    calib,
    d_paths_index_serial,
    ground_grid,
    cam_views,
):
    """Pool initializer: pin to one GPU, load predictor + loader."""
    import argparse as _argparse
    import os

    gpu_id = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    args = _argparse.Namespace(**args_dict)

    from lidar_anchored_depth.data import RoadsideV2XLoader as _Loader
    # Workers do per-frame refine, which queries dynamic V2X bboxes
    # to cull moving-vehicle LiDAR from the prior — use the
    # dynamic-labels-source loader (10 Hz interpolation).
    loader = _Loader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
        labels_source=getattr(args, "dynamic_labels_source", "interpolation"),
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]

    from lidar_anchored_depth.models import (
        PointResidualPredictor as _Predictor,
        build_prior_features as _build_feat,
    )
    predictor = _Predictor(
        args.residual_checkpoint,
        device="cuda:0",
        n_steps=args.residual_steps,
        gate_distance_m=args.gate_distance_m,
    )

    _WORKER_STATE.update({
        "args": args,
        "scene_id": scene_id,
        "calib": calib,
        "d_paths_index": {
            tuple(k): Path(v) for k, v in d_paths_index_serial.items()
        },
        "ground_grid": ground_grid,
        "cam_views": cam_views,
        "loader": loader,
        "predictor": predictor,
        "build_prior_features": _build_feat,
        "sam_dir": Path(args.sam_mask_dir) if args.sam_mask_dir else None,
        "sam_auto_dir": (
            Path(args.sam_auto_dir) if args.sam_auto_dir else None
        ),
        "segformer_dir": (
            Path(args.segformer_dir) if args.segformer_dir else None
        ),
    })
    print(
        f"[worker pid={os.getpid()} gpu={gpu_id}] predictor loaded",
        flush=True,
    )


def _worker_refine(item):
    """Pool.map callback. ``item = (cam_id, ts_ms)``. Returns dict or None."""
    cam_id, ts_ms = item
    s = _WORKER_STATE
    try:
        return refine_one_frame(
            s["scene_id"], cam_id, ts_ms,
            loader=s["loader"],
            predictor=s["predictor"],
            build_prior_features_fn=s["build_prior_features"],
            calib=s["calib"],
            d_paths_index=s["d_paths_index"],
            ground_grid=s["ground_grid"],
            cam_views=s["cam_views"],
            sam_dir=s["sam_dir"],
            sam_auto_dir=s["sam_auto_dir"],
            segformer_dir=s["segformer_dir"],
            args=s["args"],
        )
    except Exception as e:
        import traceback
        return {
            "cam_id": cam_id,
            "ts_ms": int(ts_ms),
            "error": f"{e}\n{traceback.format_exc()}",
        }


def run_refine_parallel(
    gpu_ids,
    args,
    scene_id,
    calib,
    d_paths_index,
    ground_grid,
    cam_views,
    work_items,
):
    """Spawn ``len(gpu_ids)`` worker processes, distribute ``work_items``.

    Returns a list of result dicts (or ``None`` for skipped frames),
    in the order they completed (the caller does not depend on order).
    """
    import torch.multiprocessing as mp

    # ``spawn`` is required: ``fork`` would inherit the parent's
    # CUDA state if the parent ever touched torch.cuda, and would
    # deadlock when the child tries to re-init.
    ctx = mp.get_context("spawn")
    gpu_q = ctx.Queue()
    for gid in gpu_ids:
        gpu_q.put(int(gid))

    args_dict = vars(args)
    d_paths_index_serial = {k: str(v) for k, v in d_paths_index.items()}

    init_args = (
        gpu_q,
        args_dict,
        scene_id,
        calib,
        d_paths_index_serial,
        ground_grid,
        cam_views,
    )

    print(
        f"[parallel] spawning {len(gpu_ids)} workers on gpus={gpu_ids}",
        flush=True,
    )
    n_total = len(work_items)
    results = []
    t0 = time.time()
    with ctx.Pool(
        processes=len(gpu_ids),
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_worker_refine, work_items, chunksize=1)
        ):
            results.append(res)
            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n_total - (i + 1)) / max(rate, 1e-6)
                print(
                    f"  [parallel] {i + 1}/{n_total} frames "
                    f"({rate:.2f} fps, ETA {eta:.0f}s)",
                    flush=True,
                )
    return results


__all__ = [
    "accumulate_static_lidar",
    "build_prior_for_frame",
    "refine_one_frame",
    "run_refine_parallel",
]
