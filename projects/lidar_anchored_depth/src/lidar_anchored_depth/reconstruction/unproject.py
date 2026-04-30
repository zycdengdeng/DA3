"""Backproject a per-pixel metric depth map to a colored world-frame cloud.

Given AA-HAD's per-instance ``(a_i, b_i)`` applied to the SAM mask of an
object, every mask pixel has a metric camera-axis depth ``z_cam = a · d̃ + b``.
This module unprojects those pixels through the camera ray and the
(camera-to-world) extrinsics to produce ``(N, 3)`` world points plus
``(N, 3)`` uint8 RGB sampled from the source image.
"""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    camera_to_world,
    pixel_to_camera_ray,
)


def depth_to_world_points(
    depth: np.ndarray,
    image_rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    z_min: float = 0.5,
    z_max: float = 250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map to a colored world-frame point cloud.

    Parameters
    ----------
    depth : (H, W) float, camera-axis metric depth in meters; NaN / non-positive /
        out-of-range values are dropped.
    image_rgb : (H, W, 3) uint8, RGB. Must match ``depth`` resolution.
    K : (3, 3) intrinsics.
    T_wc : (4, 4) camera-to-world.
    mask : optional (H, W) bool — restrict to these pixels (e.g. a SAM
        instance mask). Default: every valid pixel.
    z_min, z_max : meters. Pixels with depth outside ``[z_min, z_max]`` are
        dropped (covers NaN since comparisons against NaN are False).

    Returns
    -------
    points_world : (N, 3) float64 — world-frame XYZ.
    colors_rgb  : (N, 3) uint8   — RGB color per point, sampled from
        ``image_rgb`` at each pixel.
    """
    H, W = depth.shape[:2]
    if image_rgb.shape[:2] != (H, W):
        raise ValueError(
            f"image_rgb {image_rgb.shape[:2]} must match depth {depth.shape}"
        )

    valid = (depth >= z_min) & (depth <= z_max) & np.isfinite(depth)
    if mask is not None:
        if mask.shape != (H, W):
            raise ValueError(
                f"mask {mask.shape} must match depth {depth.shape}"
            )
        valid &= mask.astype(bool)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    rows, cols = np.where(valid)
    z_cam = depth[rows, cols].astype(np.float64)
    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)

    # Camera ray with z=1 from K^-1 * [u,v,1]^T
    rays_cam = pixel_to_camera_ray(uv, K)  # (N, 3) with rays_cam[:, 2] == 1
    # Scale so the camera-frame Z component equals the requested depth
    P_cam = rays_cam * z_cam[:, None]

    P_world = camera_to_world(P_cam, T_wc)
    colors = image_rgb[rows, cols]
    if colors.dtype != np.uint8:
        colors = colors.astype(np.uint8, copy=False)
    return P_world, colors
