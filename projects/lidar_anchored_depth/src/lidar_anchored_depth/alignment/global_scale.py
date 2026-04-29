"""Global-alignment baselines on ``(d̃, z_lidar)`` pixel correspondences.

These map the foundation depth model's relative output ``d̃`` to metric
``z`` using a single scalar (median / LSQ scale) or affine (LSQ /
RANSAC) shared across the whole image. They serve as the floor for our
HAD ablation table.

All solvers return an :class:`AffineFit` with ``z = a · d̃ + b``. The
"pure scale" variants set ``b = 0`` and report the scale in ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AffineFit:
    """Result of a single global affine fit.

    ``apply(d_pred)`` produces metric depth, ``a · d̃ + b``.
    """

    a: float
    b: float
    inlier_mask: np.ndarray  # bool (N,) — True for samples used in the fit
    residual_mae: float  # mean abs residual on inliers, meters
    residual_rmse: float  # rms residual on inliers, meters
    n_inliers: int

    def apply(self, d_pred: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(d_pred, dtype=np.float64) + self.b


# --------------------------------------------------------------------- #
# B0 — DA3-raw (no LiDAR)
# --------------------------------------------------------------------- #
def b0_identity(d_pred: np.ndarray, z_lidar: np.ndarray) -> AffineFit:
    """B0: take DA3 output as-is. ``a=1, b=0``."""
    n = len(z_lidar)
    inliers = np.ones(n, dtype=bool)
    if n == 0:
        return AffineFit(1.0, 0.0, inliers, 0.0, 0.0, 0)
    pred = np.asarray(d_pred, dtype=np.float64)
    z = np.asarray(z_lidar, dtype=np.float64)
    res = pred - z
    return AffineFit(
        a=1.0, b=0.0, inlier_mask=inliers,
        residual_mae=float(np.mean(np.abs(res))),
        residual_rmse=float(np.sqrt(np.mean(res ** 2))),
        n_inliers=n,
    )


# --------------------------------------------------------------------- #
# B1 — Median of z / d̃
# --------------------------------------------------------------------- #
def b1_median_scale(d_pred: np.ndarray, z_lidar: np.ndarray) -> AffineFit:
    """B1: ``a = median(z / d̃), b = 0``.

    Excludes pairs with ``d̃ <= 0`` from the median to avoid divide-by-zero.
    """
    pred = np.asarray(d_pred, dtype=np.float64)
    z = np.asarray(z_lidar, dtype=np.float64)
    if pred.size == 0:
        return AffineFit(1.0, 0.0, np.zeros(0, dtype=bool), 0.0, 0.0, 0)
    valid = pred > 1e-9
    if not valid.any():
        raise ValueError("b1_median_scale: no valid (d_pred > 0) samples")
    a = float(np.median(z[valid] / pred[valid]))
    res = a * pred - z
    return AffineFit(
        a=a, b=0.0, inlier_mask=valid,
        residual_mae=float(np.mean(np.abs(res[valid]))),
        residual_rmse=float(np.sqrt(np.mean(res[valid] ** 2))),
        n_inliers=int(valid.sum()),
    )


# --------------------------------------------------------------------- #
# B2 — Closed-form LSQ affine
# --------------------------------------------------------------------- #
def b2_lsq_affine(d_pred: np.ndarray, z_lidar: np.ndarray) -> AffineFit:
    """B2: closed-form LSQ ``min_{a,b} ||a·d̃ + b - z||²``.

    Requires at least 2 samples and at least 2 distinct ``d̃`` values
    (otherwise the system is rank-deficient).
    """
    pred = np.asarray(d_pred, dtype=np.float64)
    z = np.asarray(z_lidar, dtype=np.float64)
    n = pred.size
    if n < 2:
        raise ValueError(
            f"b2_lsq_affine: need >= 2 samples, got {n}"
        )

    A = np.stack([pred, np.ones(n, dtype=np.float64)], axis=1)
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    res = a * pred + b - z
    return AffineFit(
        a=a, b=b, inlier_mask=np.ones(n, dtype=bool),
        residual_mae=float(np.mean(np.abs(res))),
        residual_rmse=float(np.sqrt(np.mean(res ** 2))),
        n_inliers=n,
    )


# --------------------------------------------------------------------- #
# B3 — RANSAC affine
# --------------------------------------------------------------------- #
def b3_ransac_affine(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    *,
    threshold_rel: float = 0.10,
    threshold_abs: float = 0.5,
    n_iterations: int = 1000,
    seed: int = 0,
    min_samples: int = 2,
) -> AffineFit:
    """B3: RANSAC affine on ``(d̃, z)`` pairs with re-fit on inliers.

    A sample is an inlier iff
    ``|a·d̃ + b - z| < max(threshold_abs, threshold_rel · z)``.
    The final fit is a closed-form LSQ over the largest inlier set found.

    Parameters
    ----------
    threshold_rel
        Relative inlier threshold. ``0.10`` ≈ 10 %.
    threshold_abs
        Floor on the inlier threshold to keep close-range numerically
        stable (default 0.5 m).
    n_iterations
        Number of random 2-sample minimal fits to try.
    seed
        Random seed for reproducibility.
    """
    pred = np.asarray(d_pred, dtype=np.float64)
    z = np.asarray(z_lidar, dtype=np.float64)
    n = pred.size
    if n < min_samples:
        raise ValueError(
            f"b3_ransac_affine: need >= {min_samples}, got {n}"
        )

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1

    for _ in range(n_iterations):
        idx = rng.choice(n, size=2, replace=False)
        d2 = pred[idx]
        z2 = z[idx]
        if abs(d2[0] - d2[1]) < 1e-9:
            continue
        a = float((z2[0] - z2[1]) / (d2[0] - d2[1]))
        b = float(z2[0] - a * d2[0])
        residual = np.abs(a * pred + b - z)
        thr = np.maximum(threshold_abs, threshold_rel * np.abs(z))
        inliers = residual < thr
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_count < min_samples:
        # Fallback: best 2 samples we haven't used; defer to global LSQ
        return b2_lsq_affine(pred, z)

    A = np.stack(
        [pred[best_inliers], np.ones(best_count, dtype=np.float64)], axis=1
    )
    sol, *_ = np.linalg.lstsq(A, z[best_inliers], rcond=None)
    a_f, b_f = float(sol[0]), float(sol[1])
    res = a_f * pred[best_inliers] + b_f - z[best_inliers]
    return AffineFit(
        a=a_f, b=b_f, inlier_mask=best_inliers,
        residual_mae=float(np.mean(np.abs(res))),
        residual_rmse=float(np.sqrt(np.mean(res ** 2))),
        n_inliers=best_count,
    )
