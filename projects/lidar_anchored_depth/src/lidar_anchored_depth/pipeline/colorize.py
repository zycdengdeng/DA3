"""Re-colour a world-frame point cloud from anchor-ts camera images.

For each fused-cloud point we project it into every anchor-ts camera
that can see it (in front of the principal plane, inside the image
bounds), pick the one with the smallest ``z_cam`` (= most likely to
actually see the point modulo full occlusion testing), and sample
that camera's RGB at the projected pixel. Points that no camera
sees keep the fallback colour passed in by the caller.
"""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.alignment.projection import world_to_image


def build_anchor_views(loader, scene_id: str, anchor_ts: int, cams):
    """Snapshot the camera frames at one anchor ts.

    Returns a list of dicts, each with the K, T_wc, undistorted image,
    distortion coefficients, and image (H, W) shape — exactly the
    structure consumed by :func:`colorize_cloud_from_views`.
    """
    out = []
    for cam_id in cams:
        try:
            idx = loader.find_frame_idx(scene_id, anchor_ts, cam_id)
        except ValueError:
            continue
        f = loader.get_frame(idx)
        out.append({
            "cam_id": cam_id,
            "image": f.image,
            "K": f.K,
            "T_wc": f.T_wc,
            "distortion": f.meta.get("distortion"),
            "image_hw": f.image.shape[:2],
        })
    return out


def colorize_cloud_from_views(xyz, fallback_rgb, views):
    """For every point pick the closest-in-front anchor camera and
    sample image RGB at the projected pixel.

    Returns
    -------
    rgb : (N, 3) uint8 — re-coloured cloud.
    n_recoloured : int — how many points received a fresh image colour.
    """
    N = xyz.shape[0]
    if fallback_rgb is None:
        rgb_out = np.full((N, 3), 180, dtype=np.uint8)
    else:
        rgb_out = fallback_rgb.astype(np.uint8, copy=True)
    z_best = np.full(N, np.inf, dtype=np.float64)

    for v in views:
        uv, z_cam, in_front = world_to_image(
            xyz, v["K"], v["T_wc"], v["distortion"],
        )
        if uv.size == 0:
            continue
        # world_to_image may produce NaN/inf at degenerate projections
        # (e.g. points exactly at the principal point with zero z).
        # Cast-to-int below has undefined behaviour for those — drop them.
        finite = np.isfinite(uv).all(axis=1)
        if not finite.any():
            continue
        uv = uv[finite]
        z_cam = z_cam[finite]
        idx_orig = np.flatnonzero(in_front)[finite]
        H, W = v["image_hw"]
        u_int = np.round(uv[:, 0]).astype(np.int64)
        v_int = np.round(uv[:, 1]).astype(np.int64)
        in_image = (
            (u_int >= 0) & (u_int < W)
            & (v_int >= 0) & (v_int < H)
        )
        idx_orig_keep = idx_orig[in_image]
        z_keep = z_cam[in_image]
        u_keep = u_int[in_image]
        v_keep = v_int[in_image]

        is_closer = z_keep < z_best[idx_orig_keep]
        upd = idx_orig_keep[is_closer]
        z_best[upd] = z_keep[is_closer]
        rgb_out[upd] = v["image"][v_keep[is_closer], u_keep[is_closer]]

    n_recoloured = int(np.isfinite(z_best).sum())
    return rgb_out, n_recoloured


__all__ = ["build_anchor_views", "colorize_cloud_from_views"]
