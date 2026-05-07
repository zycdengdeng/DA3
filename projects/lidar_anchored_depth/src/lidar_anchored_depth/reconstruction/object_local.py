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


def voxel_downsample_robust(
    points: np.ndarray,
    voxel_size: float,
    *,
    colors: np.ndarray | None = None,
    color_method: str = "median",
    mad_k: float = 2.0,
    mad_floor: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Voxel-grid downsample with **robust** per-voxel colour aggregation.

    Same XYZ behaviour as :func:`voxel_downsample` (centroid = mean of
    points falling in the voxel), but the colour is computed with an
    outlier-resistant statistic so that occasional bad samples (e.g. a
    moving car briefly occluding a road voxel and writing its colour
    onto it, or a static structure being half-occluded by a pole at
    one camera angle) cannot pull the voxel's colour off.

    Parameters
    ----------
    color_method :
        ``"mean"``     simple mean (matches :func:`voxel_downsample`).
        ``"median"``   per-channel median over all samples in the
                       voxel. Robust to up to 50% outliers per channel
                       with no parameters. **Default**.
        ``"mad-trim"`` per-channel median, then keep only samples
                       within ``mad_k * MAD + mad_floor`` of the
                       median, then mean those inliers. Tighter than
                       median when most samples agree, falls back to
                       median when all are outliers.
    mad_k :
        threshold multiplier on the median absolute deviation for
        ``"mad-trim"``. Default 2.0.
    mad_floor :
        absolute floor (in 0-255 RGB units) added to the MAD threshold
        to avoid pathological "all samples within 1 unit of median"
        cases. Default 5.0.

    Notes
    -----
    Per-voxel ranking is done via a single ``np.argsort`` over the
    inverse-index array (linear in N). The per-voxel statistics are
    computed in a Python loop over the resulting voxel chunks; for
    typical 3-5 M voxels with ~3-10 samples each this runs in tens of
    seconds — acceptable for offline post-processing.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        empty_rgb = (
            np.zeros((0, 3), dtype=np.uint8) if colors is not None else None
        )
        return P, empty_rgb
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0; got {voxel_size}")
    if color_method not in {"mean", "median", "mad-trim"}:
        raise ValueError(f"unknown color_method {color_method!r}")

    keys = np.floor(P / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True,
    )
    M = counts.shape[0]

    sums = np.zeros((M, 3), dtype=np.float64)
    np.add.at(sums, inverse, P)
    centroids = sums / counts[:, None]

    if colors is None:
        return centroids, None

    C = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if C.shape[0] != P.shape[0]:
        raise ValueError(
            f"colors {C.shape} must align with points {P.shape}"
        )

    if color_method == "mean":
        sums_c = np.zeros((M, 3), dtype=np.float64)
        np.add.at(sums_c, inverse, C)
        out_colors = (sums_c / counts[:, None]).clip(0, 255).astype(np.uint8)
        return centroids, out_colors

    # Sort samples by voxel id so each voxel's samples are contiguous.
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    C_sorted = C[order]

    # Voxel run-length boundaries.
    if inv_sorted.size > 0:
        diff = np.diff(inv_sorted) != 0
        starts = np.concatenate([[0], np.flatnonzero(diff) + 1])
    else:
        starts = np.zeros(0, dtype=np.int64)
    ends = np.concatenate([starts[1:], [inv_sorted.size]])

    out_colors = np.zeros((M, 3), dtype=np.uint8)
    if color_method == "median":
        for s, e in zip(starts, ends):
            v = int(inv_sorted[s])
            chunk = C_sorted[s:e]
            out_colors[v] = np.median(chunk, axis=0).clip(0, 255).astype(np.uint8)
    elif color_method == "mad-trim":
        for s, e in zip(starts, ends):
            v = int(inv_sorted[s])
            chunk = C_sorted[s:e]
            if chunk.shape[0] <= 2:
                # Too few samples for robust filter; fall back to mean.
                out_colors[v] = chunk.mean(axis=0).clip(0, 255).astype(np.uint8)
                continue
            med = np.median(chunk, axis=0)
            mad = np.median(np.abs(chunk - med), axis=0)
            thresh = mad_k * mad + mad_floor
            inlier = np.all(np.abs(chunk - med) <= thresh, axis=1)
            if inlier.any():
                out_colors[v] = chunk[inlier].mean(axis=0).clip(0, 255).astype(np.uint8)
            else:
                out_colors[v] = med.clip(0, 255).astype(np.uint8)

    return centroids, out_colors
