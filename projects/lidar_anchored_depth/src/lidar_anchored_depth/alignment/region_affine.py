"""Per-mask affine baseline on ``(d̃, z_lidar)`` pairs (B4).

For each SAM instance mask, fit ``z = a_i · d̃ + b_i`` from the LiDAR
returns that fall inside the mask. This is the standard prior-art
formulation HAD will compete against. ``z`` is camera-frame depth here
(NOT world-frame Z) — that's the whole point of B4 vs HAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lidar_anchored_depth.alignment.global_scale import (
    AffineFit,
    b2_lsq_affine,
    b3_ransac_affine,
)


@dataclass(slots=True)
class RegionAffineResult:
    """Result of :func:`region_affine_align`."""

    fits: dict[int, AffineFit] = field(default_factory=dict)
    fallback_global: AffineFit | None = None
    n_regions_attempted: int = 0
    n_regions_solved: int = 0
    n_regions_fallback: int = 0


def _aggregate_lidar_in_masks(
    pixel_uv: np.ndarray,
    masks: np.ndarray,
) -> dict[int, np.ndarray]:
    """Index LiDAR sample positions by mask id.

    Returns a dict mapping mask index -> array of indices into the
    original ``pixel_uv`` / ``z_lidar`` arrays.
    """
    K = masks.shape[0]
    H, W = masks.shape[1:]
    uv = np.round(pixel_uv).astype(np.int64)
    out: dict[int, np.ndarray] = {}
    for k in range(K):
        m = masks[k]
        uvi = uv
        in_image = (
            (uvi[:, 0] >= 0) & (uvi[:, 0] < W)
            & (uvi[:, 1] >= 0) & (uvi[:, 1] < H)
        )
        if not in_image.any():
            out[k] = np.zeros(0, dtype=np.int64)
            continue
        # Sample mask values at integer pixel coords
        u_in = uvi[in_image, 0]
        v_in = uvi[in_image, 1]
        flat = np.zeros(uvi.shape[0], dtype=bool)
        flat[in_image] = m[v_in, u_in]
        out[k] = np.flatnonzero(flat)
    return out


def region_affine_align(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    pixel_uv: np.ndarray,
    masks: np.ndarray,
    *,
    min_lidar_per_region: int = 5,
    use_ransac: bool = True,
    fallback_to_global: bool = True,
    ransac_seed: int = 0,
) -> RegionAffineResult:
    """Per-mask affine fit on LiDAR-derived (d̃, z) pairs.

    Parameters
    ----------
    d_pred : (N,) DA3 relative depth at each sample's pixel.
    z_lidar : (N,) ground-truth metric depth at each sample (camera z).
    pixel_uv : (N, 2) integer-rounded pixel coords for each sample.
    masks : (K, H, W) bool instance masks.
    min_lidar_per_region : drop masks with fewer than this many LiDAR
        samples; their fit goes to ``fallback_global`` instead.
    use_ransac : True → B3 RANSAC; False → B2 LSQ.
    fallback_to_global : when True (default), masks with too few samples
        contribute to a single global RANSAC fit applied as fallback.

    Returns
    -------
    :class:`RegionAffineResult` with per-mask fits and (optionally) a
    fallback global fit.
    """
    pred = np.asarray(d_pred, dtype=np.float64).reshape(-1)
    z = np.asarray(z_lidar, dtype=np.float64).reshape(-1)
    uv = np.asarray(pixel_uv, dtype=np.float64).reshape(-1, 2)
    if pred.shape[0] != z.shape[0] or pred.shape[0] != uv.shape[0]:
        raise ValueError(
            "region_affine_align: shape mismatch among "
            f"d_pred={pred.shape}, z_lidar={z.shape}, pixel_uv={uv.shape}"
        )
    if masks.ndim != 3:
        raise ValueError(
            f"region_affine_align: masks must be (K, H, W); got {masks.shape}"
        )

    idx_per_mask = _aggregate_lidar_in_masks(uv, masks)
    fits: dict[int, AffineFit] = {}
    fallback_idx_pool: list[np.ndarray] = []
    n_attempt = 0
    n_solved = 0
    n_fallback = 0

    for k, idx in idx_per_mask.items():
        n_attempt += 1
        if idx.size < min_lidar_per_region:
            if fallback_to_global:
                fallback_idx_pool.append(idx)
            n_fallback += 1
            continue
        try:
            if use_ransac:
                fit = b3_ransac_affine(pred[idx], z[idx], seed=ransac_seed + k)
            else:
                fit = b2_lsq_affine(pred[idx], z[idx])
            fits[k] = fit
            n_solved += 1
        except (np.linalg.LinAlgError, ValueError):
            n_fallback += 1
            if fallback_to_global:
                fallback_idx_pool.append(idx)

    fallback_fit: AffineFit | None = None
    if fallback_to_global and pred.size >= 2:
        # Always compute a global fallback when requested — it covers
        # both unsolved-mask samples AND pixels lying outside every
        # mask in :func:`apply_region_fits`. Without this, ``out``
        # would be filled with ``fill_value`` everywhere outside masks.
        try:
            if use_ransac:
                fallback_fit = b3_ransac_affine(pred, z, seed=ransac_seed)
            else:
                fallback_fit = b2_lsq_affine(pred, z)
        except (np.linalg.LinAlgError, ValueError):
            fallback_fit = None

    return RegionAffineResult(
        fits=fits,
        fallback_global=fallback_fit,
        n_regions_attempted=n_attempt,
        n_regions_solved=n_solved,
        n_regions_fallback=n_fallback,
    )


def apply_region_fits(
    d_pred_image: np.ndarray,
    masks: np.ndarray,
    result: RegionAffineResult,
    *,
    fill_value: float = float("nan"),
) -> np.ndarray:
    """Apply per-mask ``(a_i, b_i)`` to a full ``d̃`` image.

    Pixels in mask ``k`` use ``fits[k]``; pixels not in any mask use the
    fallback global fit if present, otherwise ``fill_value``.
    """
    H, W = d_pred_image.shape
    if masks.shape[1:] != (H, W):
        raise ValueError(
            f"apply_region_fits: masks {masks.shape} != image {d_pred_image.shape}"
        )

    out = np.full((H, W), fill_value, dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for k, fit in result.fits.items():
        m = masks[k]
        out[m] = (fit.a * d_pred_image[m] + fit.b).astype(np.float32)
        covered |= m

    if result.fallback_global is not None:
        not_covered = ~covered
        out[not_covered] = (
            result.fallback_global.a * d_pred_image[not_covered]
            + result.fallback_global.b
        ).astype(np.float32)
    return out
