"""LiDAR-anchored depth completion — Stage 5.

Reframes the static-scene reconstruction as **sparse LiDAR depth
completion via foundation depth + V2X bbox anchoring**:

1. LiDAR is the geometric backbone (cm-precise, ground truth).
2. AA-HAD (DA3 + per-camera affine) only fills voxels that LiDAR
   does NOT already occupy. Where LiDAR exists, it wins.
3. Only points inside at least one camera's frustum are kept (we
   cannot fill what the cameras did not see).

This module exposes the building blocks; the orchestrating script
(``scripts/run_lidar_completion.py``) calls them in order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    project_camera_to_image,
    world_to_camera,
)


@dataclass(slots=True)
class CameraView:
    """Per-camera intrinsics + extrinsics + image dims used for FOV
    culling. Pulled from ``Frame`` once per camera."""

    cam_id: str
    K: np.ndarray
    T_wc: np.ndarray
    image_hw: tuple[int, int]
    distortion: np.ndarray | None = None


def points_in_camera_fov(
    points_world: np.ndarray,
    view: CameraView,
    *,
    z_min: float = 0.1,
    z_max: float | None = None,
) -> np.ndarray:
    """``(N,)`` bool mask: True where the world point projects into
    the camera's image (after intrinsics + distortion + image-bound
    test) and lies within ``[z_min, z_max]`` in camera Z.
    """
    if points_world.size == 0:
        return np.zeros(0, dtype=bool)
    P_cam = world_to_camera(points_world, view.T_wc)
    uv, in_front = project_camera_to_image(
        P_cam, view.K, view.distortion, z_min=z_min,
    )
    H, W = view.image_hw
    in_image = (
        (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1)
        & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    )
    if z_max is not None:
        z_cam = P_cam[in_front, 2]
        in_image = in_image & (z_cam <= z_max)
    out = np.zeros(points_world.shape[0], dtype=bool)
    idx_in_front = np.flatnonzero(in_front)
    out[idx_in_front[in_image]] = True
    return out


def points_in_any_camera_fov(
    points_world: np.ndarray,
    views: list[CameraView],
    *,
    z_min: float = 0.1,
    z_max: float | None = None,
) -> np.ndarray:
    """Union of per-camera FOV masks: ``True`` if visible in at least
    one camera."""
    if not views:
        return np.zeros(points_world.shape[0], dtype=bool)
    out = np.zeros(points_world.shape[0], dtype=bool)
    for v in views:
        out |= points_in_camera_fov(points_world, v, z_min=z_min, z_max=z_max)
    return out


def _voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Integer voxel indices ``(N, 3)`` for points at ``voxel_size``."""
    return np.floor(points / float(voxel_size)).astype(np.int64)


def _voxel_hash_set(keys: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(row) for row in keys}


def lidar_priority_fill(
    lidar_xyz: np.ndarray,
    aahad_xyz: np.ndarray,
    aahad_rgb: np.ndarray | None,
    voxel_size: float,
    *,
    max_dist_to_lidar: float | None = None,
    outlier_reference_lidar: np.ndarray | None = None,
    lidar_color: tuple[int, int, int] = (180, 180, 180),
    aahad_fallback_color: tuple[int, int, int] = (220, 220, 100),
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Build the depth-completion fusion: LiDAR backbone + AA-HAD fill
    in LiDAR-empty voxels.

    Algorithm
    ---------
    1. Voxelise LiDAR; build a set of occupied voxel keys.
    2. Voxelise AA-HAD; drop AA-HAD points whose voxel is occupied
       (LiDAR wins).
    3. Optionally drop AA-HAD points whose distance to the **nearest
       LiDAR point** exceeds ``max_dist_to_lidar`` — sanity filter
       against far-flung outliers.
    4. Concatenate LiDAR + surviving AA-HAD points. The output also
       returns a ``source`` array (0 = LiDAR, 1 = AA-HAD) for diagnostics.

    Parameters
    ----------
    lidar_xyz : (N_l, 3) world XYZ.
    aahad_xyz : (N_a, 3) world XYZ.
    aahad_rgb : optional (N_a, 3) uint8 RGB. If ``None`` and
        ``aahad_xyz`` has points, AA-HAD points are coloured with
        ``aahad_fallback_color`` (so the LiDAR backbone stays grey
        instead of dragging the whole cloud into a fallback colour).
    voxel_size : metres.
    max_dist_to_lidar : optional metres; if set, drop AA-HAD points
        whose nearest LiDAR neighbour is farther than this.
    outlier_reference_lidar : optional (N, 3) cloud used **only** for
        the ``max_dist_to_lidar`` outlier check. When ``None`` (default)
        the check uses ``lidar_xyz``. Pass a *pre-filter* LiDAR cloud
        here when ``lidar_xyz`` has been thinned (e.g. ground band
        removed via ``--lidar-skip-ground``) so AA-HAD ground points
        aren't falsely flagged as outliers just because the surviving
        LiDAR backbone no longer covers the road surface.
    lidar_color : ``(R, G, B)`` uint8 for LiDAR points (default light
        grey ``(180, 180, 180)``); previously ``(0, 0, 0)`` made the
        cloud disappear in dark CloudCompare backgrounds.
    aahad_fallback_color : RGB used for AA-HAD points whose source
        cloud had no per-point colour (e.g. a per-object PLY where
        AA-HAD failed but LiDAR survived). Default yellow
        ``(220, 220, 100)``.

    Returns
    -------
    xyz : (N, 3) float32 combined cloud.
    rgb : (N, 3) uint8 — always returned when at least one of LiDAR
        or AA-HAD has points. ``None`` only when both are empty.
    source : (N,) uint8 — 0 for LiDAR, 1 for AA-HAD.
    """
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        raise ValueError(f"lidar_xyz must be (N, 3); got {lidar_xyz.shape}")
    if aahad_xyz.size and aahad_xyz.shape[1] != 3:
        raise ValueError(f"aahad_xyz must be (N, 3); got {aahad_xyz.shape}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    lidar_color_arr = np.asarray(lidar_color, dtype=np.uint8).reshape(1, 3)
    aahad_fallback_arr = np.asarray(aahad_fallback_color, dtype=np.uint8).reshape(1, 3)

    if lidar_xyz.size == 0:
        # Pure AA-HAD fallback.
        kept_xyz = aahad_xyz.astype(np.float32)
        if aahad_rgb is not None:
            kept_rgb = aahad_rgb.astype(np.uint8)
        elif kept_xyz.shape[0] > 0:
            kept_rgb = np.broadcast_to(
                aahad_fallback_arr, (kept_xyz.shape[0], 3),
            ).astype(np.uint8).copy()
        else:
            kept_rgb = None
        src = np.ones(kept_xyz.shape[0], dtype=np.uint8)
        return kept_xyz, kept_rgb, src

    if aahad_xyz.size == 0:
        kept_xyz = lidar_xyz.astype(np.float32)
        # LiDAR alone always gets coloured with lidar_color (regardless
        # of whether aahad_rgb was given) so it never silently inherits
        # a fallback colour.
        kept_rgb = np.broadcast_to(
            lidar_color_arr, (lidar_xyz.shape[0], 3),
        ).astype(np.uint8).copy()
        src = np.zeros(kept_xyz.shape[0], dtype=np.uint8)
        return kept_xyz, kept_rgb, src

    lidar_keys = _voxel_keys(lidar_xyz, voxel_size)
    aahad_keys = _voxel_keys(aahad_xyz, voxel_size)
    occupied = _voxel_hash_set(lidar_keys)
    keep_aahad = np.fromiter(
        (tuple(k) not in occupied for k in aahad_keys),
        dtype=bool,
        count=aahad_keys.shape[0],
    )

    if max_dist_to_lidar is not None and keep_aahad.any():
        try:
            from scipy.spatial import cKDTree

            # Build the outlier-rejection tree on the un-thinned LiDAR
            # if provided; otherwise fall back to the backbone LiDAR.
            tree_pts = (
                outlier_reference_lidar
                if outlier_reference_lidar is not None
                and outlier_reference_lidar.size > 0
                else lidar_xyz
            )
            tree = cKDTree(tree_pts)
            cand_idx = np.flatnonzero(keep_aahad)
            d, _ = tree.query(aahad_xyz[cand_idx], k=1)
            close = d <= float(max_dist_to_lidar)
            drop = cand_idx[~close]
            keep_aahad[drop] = False
        except ModuleNotFoundError:
            pass

    aahad_kept_xyz = aahad_xyz[keep_aahad]
    aahad_kept_rgb = aahad_rgb[keep_aahad] if aahad_rgb is not None else None

    xyz = np.concatenate([lidar_xyz, aahad_kept_xyz], axis=0).astype(np.float32)
    # LiDAR backbone is always coloured grey; AA-HAD uses its own
    # colours when given, otherwise the configurable fallback.
    lidar_rgb = np.broadcast_to(
        lidar_color_arr, (lidar_xyz.shape[0], 3),
    ).astype(np.uint8).copy()
    if aahad_kept_rgb is not None:
        aa_rgb_use = aahad_kept_rgb
    else:
        aa_rgb_use = np.broadcast_to(
            aahad_fallback_arr, (aahad_kept_xyz.shape[0], 3),
        ).astype(np.uint8).copy()
    rgb = np.concatenate([lidar_rgb, aa_rgb_use], axis=0).astype(np.uint8)

    src = np.concatenate([
        np.zeros(lidar_xyz.shape[0], dtype=np.uint8),
        np.ones(aahad_kept_xyz.shape[0], dtype=np.uint8),
    ])
    return xyz, rgb, src


__all__ = [
    "CameraView",
    "points_in_camera_fov",
    "points_in_any_camera_fov",
    "lidar_priority_fill",
]
