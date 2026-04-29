"""Per-mask affine baseline on (d_pred, z_lidar) pairs (B4).

Stage 1 stub. Ports ``RegionWiseDepthAligner`` semantics from the prior
``utils/region_depth_alignment.py`` but slimmed down to the baseline form
(no SAM-driven HAD logic, which lives in :mod:`alignment.height_anchor`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegionAffineResult:
    aligned_depth: np.ndarray
    coverage_mask: np.ndarray
    per_region_a: dict[int, float]
    per_region_b: dict[int, float]
    per_region_inliers: dict[int, int]


def region_affine_align(
    d_pred: np.ndarray,
    z_lidar_sparse: np.ndarray,
    pixel_uv: np.ndarray,
    masks: np.ndarray,
    *,
    min_lidar_per_region: int = 5,
    use_ransac: bool = True,
) -> RegionAffineResult:
    raise NotImplementedError("Stage 2")
