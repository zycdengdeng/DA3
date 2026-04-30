"""Object-local frame transforms + voxel downsampling.

For temporal accumulation of one tracked V2X object across many frames,
we need to bring the per-frame world points into a *canonical* frame
attached to the object. The bbox at frame ``i`` defines:

    rotation R_i (from euler_zyx_to_R(roll, pitch, yaw))
    translation t_i (the bbox centre)

The world ↔ object-local transform is::

    P_world = R_i · P_local + t_i
    P_local = R_iᵀ · (P_world − t_i)

After accumulating many frames in the object-local frame, voxel-grid
downsampling de-duplicates redundant returns and produces a canonical
object model that scales sublinearly with frame count.
"""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.data.calibration import euler_zyx_to_R


def world_to_object_local(
    points_world: np.ndarray, obj: DynamicObject
) -> np.ndarray:
    """Transform world-frame points into the object's local frame.

    P_local = R_objᵀ · (P_world − t_obj)
    """
    P = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        return P
    R = euler_zyx_to_R(obj.roll, obj.pitch, obj.yaw)
    centered = P - obj.xyz
    return centered @ R  # equivalent to (R.T @ centered.T).T


def object_local_to_world(
    points_local: np.ndarray, obj: DynamicObject
) -> np.ndarray:
    """Transform object-local points back to world frame."""
    P = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        return P
    R = euler_zyx_to_R(obj.roll, obj.pitch, obj.yaw)
    return P @ R.T + obj.xyz


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Voxel-grid downsample ``(N, 3)`` points.

    Each voxel's representative is the mean of all points falling into
    it. If ``colors`` is provided, each voxel's color is the mean of the
    contributing colors (rounded to ``uint8``).

    Parameters
    ----------
    points : (N, 3)
    voxel_size : meters; e.g. ``0.05`` for 5 cm voxels.
    colors : optional (N, 3) uint8 RGB.

    Returns
    -------
    centroids : (M, 3) float64
    colors    : (M, 3) uint8, or ``None`` if input ``colors`` was ``None``.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        return P, (np.zeros((0, 3), dtype=np.uint8) if colors is not None else None)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0; got {voxel_size}")

    # Grid index for each point
    keys = np.floor(P / voxel_size).astype(np.int64)
    # Unique voxel keys + inverse to aggregate
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    M = counts.shape[0]

    sums = np.zeros((M, 3), dtype=np.float64)
    np.add.at(sums, inverse, P)
    centroids = sums / counts[:, None]

    out_colors: np.ndarray | None = None
    if colors is not None:
        C = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if C.shape[0] != P.shape[0]:
            raise ValueError(
                f"colors {C.shape} must align with points {P.shape}"
            )
        sums_c = np.zeros((M, 3), dtype=np.float64)
        np.add.at(sums_c, inverse, C)
        out_colors = (sums_c / counts[:, None]).clip(0, 255).astype(np.uint8)

    return centroids, out_colors
