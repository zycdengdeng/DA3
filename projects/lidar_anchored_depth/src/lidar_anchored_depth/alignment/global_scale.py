"""Global scale / affine alignment baselines (B1, B2, B3).

Stage 1 stub. The Stage 2 implementation ports the vetted algorithm from
``utils/lidar_alignment.py`` of the prior ``roadside-reconstruction`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GlobalAlignmentResult:
    aligned_depth: np.ndarray
    scale: float
    bias: float  # 0.0 for pure-scale variants
    inlier_ratio: float
    residual_mean: float
    residual_std: float
    num_correspondences: int


def global_lsq_scale(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    pixel_uv: np.ndarray,
) -> GlobalAlignmentResult:
    raise NotImplementedError("Stage 2")


def global_ransac_scale(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    pixel_uv: np.ndarray,
    *,
    threshold_rel: float = 0.10,
    n_iterations: int = 1000,
    seed: int = 0,
) -> GlobalAlignmentResult:
    raise NotImplementedError("Stage 2")


def global_ransac_affine(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    pixel_uv: np.ndarray,
    *,
    threshold_rel: float = 0.10,
    n_iterations: int = 1000,
    seed: int = 0,
) -> GlobalAlignmentResult:
    raise NotImplementedError("Stage 2")
