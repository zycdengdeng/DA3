"""Density-aware fusion of HAD-instance depth, ground-plane depth, and
LiDAR-anchored point depth (M3 component).

Stage 1 stub.
"""

from __future__ import annotations

import numpy as np


def adaptive_fuse(
    instance_depth: np.ndarray,
    ground_depth: np.ndarray,
    lidar_density: np.ndarray | None,
) -> np.ndarray:
    """Fuse two metric depth fields covering disjoint regions.

    Resolves seams via local LiDAR density / mask boundaries. Stage 2.
    """
    raise NotImplementedError("Stage 2")
