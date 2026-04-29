"""Synthetic-scene helper functions used across alignment tests.

These are plain functions rather than pytest fixtures so they can be
imported directly via ``from _synthetic import ...`` from any test
module. Keeping them out of ``conftest.py`` (which is special and not
importable as a regular module) avoids the relative-import dance.
"""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    bbox_3d_corners,
    bbox_uv_aabb,
    project_camera_to_image,
    world_to_camera,
    world_to_image,
)


def synthetic_relative_depth(
    lidar_world: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    image_hw: tuple[int, int],
    *,
    a_true: float = 0.20,
    b_true: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (H, W) ground-truth metric depth + relative depth pair.

    The relative depth is constructed so that ``z = a_true · d̃ + b_true``
    holds exactly at every pixel where a LiDAR sample fell — i.e. the
    "perfect DA3 model". This lets HAD-style solvers exactly recover
    ``(a_true, b_true)``.
    """
    H, W = image_hw
    z_img = np.full((H, W), np.nan, dtype=np.float32)
    uv, z_cam, _ = world_to_image(lidar_world, K, T_wc, dist=None)
    if uv.size == 0:
        return z_img, z_img.copy()
    uv_int = np.round(uv).astype(np.int64)
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    u = uv_int[inside, 0]
    v = uv_int[inside, 1]
    z_img[v, u] = z_cam[inside].astype(np.float32)
    d_img = (z_img - b_true) / a_true
    return z_img, d_img


def synthetic_d_image_consistent_with_mask(
    mask: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    Z_min: float,
    Z_max: float,
    *,
    a_true: float = 0.25,
    b_true: float = 1.0,
    fill_outside: float = 1.0,
) -> np.ndarray:
    """Build a (H, W) ``d̃`` so that the HAD solver recovers ``(a_true, b_true)``.

    Inside the mask, the pixel's world-Z is linearly interpolated from
    ``Z_max`` at the top row to ``Z_min`` at the bottom. The camera depth
    consistent with that world-Z is computed analytically, and ``d̃`` is
    set so that ``z_cam = a_true · d̃ + b_true`` holds.

    Outside the mask we fill ``fill_outside``; the HAD solver does not
    read those pixels.
    """
    from lidar_anchored_depth.alignment.projection import (
        world_z_at_pixel_unit_depth,
    )

    H, W = mask.shape
    d_img = np.full((H, W), fill_outside, dtype=np.float32)
    if not mask.any():
        return d_img

    rows, cols = np.where(mask)
    v_top = int(rows.min())
    v_bot = int(rows.max())
    span = max(1, v_bot - v_top)

    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)

    frac = (rows.astype(np.float64) - v_top) / span  # 0 at top, 1 at bot
    Z_pix = Z_max + frac * (Z_min - Z_max)            # top→Z_max, bot→Z_min

    # Avoid divide-by-zero
    safe_alpha = np.where(np.abs(alphas) < 1e-9, 1.0, alphas)
    z_cam = (Z_pix - beta) / safe_alpha
    d_pix = (z_cam - b_true) / a_true
    d_img[rows, cols] = d_pix.astype(np.float32)
    return d_img


def synthetic_mask_from_world_box(
    cx: float, cy: float, cz: float,
    length: float, width: float, height: float,
    yaw: float,
    K: np.ndarray, T_wc: np.ndarray, image_hw: tuple[int, int],
) -> np.ndarray:
    """Build a synthetic (H, W) bool mask = projected AABB of a 3D box.

    Stand-in for a SAM mask in tests; the AABB of the projected bbox
    silhouette is good enough for height-anchor unit testing.
    """
    H, W = image_hw
    corners = bbox_3d_corners(cx, cy, cz, length, width, height, yaw)
    P_cam = world_to_camera(corners, T_wc)
    if not (P_cam[:, 2] > 0.1).all():
        return np.zeros((H, W), dtype=bool)
    uv8, _ = project_camera_to_image(P_cam, K, dist=None, z_min=0.0)
    if uv8.shape[0] != 8:
        return np.zeros((H, W), dtype=bool)
    u_min, v_min, u_max, v_max = bbox_uv_aabb(uv8)
    u_lo, u_hi = max(0, int(u_min)), min(W, int(u_max) + 1)
    v_lo, v_hi = max(0, int(v_min)), min(H, int(v_max) + 1)
    mask = np.zeros((H, W), dtype=bool)
    mask[v_lo:v_hi, u_lo:u_hi] = True
    return mask
