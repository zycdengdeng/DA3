"""Height-Anchored Depth (HAD) — closed-form per-instance solver.

The geometry derivation lives in ``docs/method.md`` §3. The crux is that
for a pixel ``(u, v)`` and DA3 relative depth ``d̃`` the world-frame Z
component of the back-projected 3D point is **linear** in the affine
parameters ``(a, b)`` (with ``z_cam = a · d̃ + b``), so two height
anchors produce a closed-form ``(a, b)``.

Two anchor sources are supported:

- **M1 HAD-mask** (this module): estimate ``(Z_min, Z_max)`` from the
  LiDAR points falling inside the SAM mask via percentile clipping.
- **M2 HAD-bbox / M3 AA-HAD** (:mod:`alignment.bbox_anchor`): take
  ``(Z_min, Z_max)`` directly from the V2X 3D-bbox annotation.

This module owns the shared closed-form solver
(:func:`solve_affine_from_height_anchors`) and the M1 mask-driven path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    world_to_image,
    world_z_at_pixel_unit_depth,
)


# --------------------------------------------------------------------- #
# Closed-form solver — shared by M1 / M2 / M3
# --------------------------------------------------------------------- #
def solve_affine_from_height_anchors(
    K: np.ndarray,
    T_wc: np.ndarray,
    uv_top: np.ndarray,           # (2,) pixel
    uv_bot: np.ndarray,           # (2,) pixel
    d_pred_top: float,            # DA3 relative depth at top pixel
    d_pred_bot: float,            # DA3 relative depth at bottom pixel
    Z_max: float,                 # world Z at top of object
    Z_min: float,                 # world Z at bottom of object
) -> tuple[float, float] | None:
    """Solve ``(a, b)`` from two height anchors. Closed-form, no iteration.

    Math (see ``docs/method.md`` §3, ``world_z_at_pixel_unit_depth``)::

        Z_world(u, v, z_cam) = α(u,v) · z_cam + β
        z_cam(u, v)         = a · d̃(u, v) + b
        ⇒ Z_world = α · (a · d̃ + b) + β

    Two anchors → 2×2 linear system in ``(a, b)``::

        α_top · d̃_top · a + α_top · b = Z_max - β
        α_bot · d̃_bot · a + α_bot · b = Z_min - β

    Returns
    -------
    (a, b) on success, or ``None`` when the system is singular (which
    happens iff the top and bottom anchors share the same pixel ``α``
    *and* the same ``d̃``, or any α equals 0 — typically meaning a
    horizontally-aligned camera, which the project doesn't deploy in).
    """
    uv = np.asarray([uv_top, uv_bot], dtype=np.float64)
    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)
    a_top, a_bot = float(alphas[0]), float(alphas[1])
    if abs(a_top) < 1e-9 or abs(a_bot) < 1e-9:
        return None

    A = np.array(
        [
            [a_top * d_pred_top, a_top],
            [a_bot * d_pred_bot, a_bot],
        ],
        dtype=np.float64,
    )
    rhs = np.array([Z_max - beta, Z_min - beta], dtype=np.float64)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    sol = np.linalg.solve(A, rhs)
    return float(sol[0]), float(sol[1])


# --------------------------------------------------------------------- #
# Mask geometry helpers
# --------------------------------------------------------------------- #
def mask_top_bottom_pixels(
    mask: np.ndarray, *, take_centroid_of_row: bool = True
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the topmost and bottommost pixel of a binary mask.

    The horizontal coordinate is the row centroid (averaged column
    indices) by default — robust to a ragged mask edge. Returns
    ``None`` if the mask is empty.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    v_top = int(rows.min())
    v_bot = int(rows.max())
    if take_centroid_of_row:
        u_top = float(cols[rows == v_top].mean())
        u_bot = float(cols[rows == v_bot].mean())
    else:
        u_top = float(cols[rows == v_top].min())
        u_bot = float(cols[rows == v_bot].max())
    return (
        np.array([u_top, v_top], dtype=np.float64),
        np.array([u_bot, v_bot], dtype=np.float64),
    )


# --------------------------------------------------------------------- #
# M1: HAD-mask (height anchors derived from LiDAR-in-mask)
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class HeightInterval:
    """Height interval derived from LiDAR points inside an instance mask."""

    Z_min: float
    Z_max: float
    n_points: int
    confidence: float  # in [0, 1]


def height_interval_from_lidar_in_mask(
    lidar_world: np.ndarray,
    pixel_uv: np.ndarray,
    in_front: np.ndarray,
    mask: np.ndarray,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_points: int = 5,
) -> HeightInterval | None:
    """Estimate ``(Z_min, Z_max)`` for one instance from LiDAR inside its mask.

    Parameters
    ----------
    lidar_world : (N, 3)
        Full LiDAR cloud in world frame.
    pixel_uv : (M, 2)
        Projection of the in-front LiDAR points to image (output of
        ``world_to_image``).
    in_front : (N,) bool
        Mask indicating which of the original LiDAR points were in front
        of the camera (i.e. lined up with ``pixel_uv``).
    mask : (H, W) bool
        Instance mask in the same image as ``pixel_uv``.

    Returns
    -------
    :class:`HeightInterval` or ``None`` if too few points inside the mask.

    Confidence heuristic
    --------------------
    ``confidence = clip(n_points / 50, 0, 1) * (1 - 0.5 · clip((Z_max - Z_min) / 5 - 1, 0, 1))``
    — favors masks with many points and tight intervals.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    H, W = mask.shape
    uv_int = np.round(pixel_uv).astype(np.int64)
    in_image = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    if not in_image.any():
        return None

    # Map back to original LiDAR indices (in-front + in-image)
    mask_vals = np.zeros(uv_int.shape[0], dtype=bool)
    mask_vals[in_image] = mask[uv_int[in_image, 1], uv_int[in_image, 0]]
    if mask_vals.sum() < min_points:
        return None

    # Recover the indices in the ORIGINAL lidar_world array
    front_indices = np.flatnonzero(in_front)
    selected_orig_indices = front_indices[mask_vals]
    Z = lidar_world[selected_orig_indices, 2].astype(np.float64)
    if Z.size < min_points:
        return None

    z_lo = float(np.percentile(Z, percentile_lo))
    z_hi = float(np.percentile(Z, percentile_hi))
    n_pts = int(Z.size)

    confidence_count = float(np.clip(n_pts / 50.0, 0.0, 1.0))
    height_extent = z_hi - z_lo
    looseness = float(np.clip((height_extent / 5.0) - 1.0, 0.0, 1.0))
    confidence = confidence_count * (1.0 - 0.5 * looseness)

    return HeightInterval(
        Z_min=z_lo, Z_max=z_hi, n_points=n_pts, confidence=confidence
    )


@dataclass(slots=True)
class InstanceAnchor:
    """One HAD anchor pair (top + bot) plus its closed-form ``(a, b)``."""

    instance_id: int
    uv_top: np.ndarray  # (2,)
    uv_bot: np.ndarray
    d_pred_top: float
    d_pred_bot: float
    Z_min: float
    Z_max: float
    a: float
    b: float
    confidence: float


@dataclass(slots=True)
class HADResult:
    """Per-frame HAD output (used by M1 and M2/M3)."""

    aligned_depth: np.ndarray  # (H, W) float32 metric depth, NaN elsewhere
    coverage_mask: np.ndarray  # (H, W) bool
    anchors: list[InstanceAnchor] = field(default_factory=list)
    n_masks_attempted: int = 0
    n_masks_solved: int = 0


def had_mask(
    d_pred_image: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    masks: np.ndarray,
    lidar_world: np.ndarray,
    *,
    dist: np.ndarray | None = None,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_lidar_per_mask: int = 5,
) -> HADResult:
    """M1 HAD-mask: per-instance height-anchored solve, anchors from LiDAR.

    For each mask: take its top / bottom pixels, estimate
    ``(Z_min, Z_max)`` from the LiDAR points falling inside it (5 / 95
    percentile by default), and run the shared closed-form solver.

    Parameters
    ----------
    d_pred_image : (H, W) float, DA3 relative depth.
    K, T_wc, dist : pinhole calibration (distortion may be ``None``).
    masks : (K, H, W) bool, instance masks.
    lidar_world : (N, 3) world-frame LiDAR.
    """
    H, W = d_pred_image.shape
    aligned = np.full((H, W), np.nan, dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)

    # Project all LiDAR once
    uv_all, _, in_front = world_to_image(lidar_world, K, T_wc, dist)

    anchors: list[InstanceAnchor] = []
    n_attempt = 0
    n_solved = 0

    for k in range(masks.shape[0]):
        n_attempt += 1
        mask_k = masks[k]
        if not mask_k.any():
            continue
        topbot = mask_top_bottom_pixels(mask_k)
        if topbot is None:
            continue
        uv_top, uv_bot = topbot

        height_int = height_interval_from_lidar_in_mask(
            lidar_world=lidar_world,
            pixel_uv=uv_all,
            in_front=in_front,
            mask=mask_k,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            min_points=min_lidar_per_mask,
        )
        if height_int is None:
            continue

        d_top = float(d_pred_image[int(uv_top[1]), int(uv_top[0])])
        d_bot = float(d_pred_image[int(uv_bot[1]), int(uv_bot[0])])
        if abs(d_top - d_bot) < 1e-9:
            continue

        sol = solve_affine_from_height_anchors(
            K=K, T_wc=T_wc,
            uv_top=uv_top, uv_bot=uv_bot,
            d_pred_top=d_top, d_pred_bot=d_bot,
            Z_max=height_int.Z_max, Z_min=height_int.Z_min,
        )
        if sol is None:
            continue
        a, b = sol

        # Apply to mask
        aligned[mask_k] = (a * d_pred_image[mask_k] + b).astype(np.float32)
        coverage |= mask_k
        n_solved += 1
        anchors.append(
            InstanceAnchor(
                instance_id=k,
                uv_top=uv_top, uv_bot=uv_bot,
                d_pred_top=d_top, d_pred_bot=d_bot,
                Z_min=height_int.Z_min, Z_max=height_int.Z_max,
                a=a, b=b, confidence=height_int.confidence,
            )
        )

    return HADResult(
        aligned_depth=aligned,
        coverage_mask=coverage,
        anchors=anchors,
        n_masks_attempted=n_attempt,
        n_masks_solved=n_solved,
    )
