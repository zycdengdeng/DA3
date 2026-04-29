"""Ground-plane RANSAC fit and per-pixel ground-depth (M2 component).

Stage 1 stub. Ports the vetted ``fit_ground_plane_ransac`` /
``compute_ground_depth_from_plane`` from the prior branch and adds tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GroundPlane:
    """World-frame plane ``n · X + d = 0`` with unit normal ``n``."""

    normal: np.ndarray  # (3,)
    offset: float
    inlier_ratio: float
    residual_std: float


def fit_ground_plane_ransac(
    lidar_world: np.ndarray,
    *,
    distance_threshold: float = 0.20,
    n_iterations: int = 2000,
    seed: int = 0,
) -> GroundPlane:
    raise NotImplementedError("Stage 2")


def ground_depth_map(
    plane: GroundPlane,
    K: np.ndarray,
    T_cw: np.ndarray,
    image_hw: tuple[int, int],
    *,
    non_ground_mask: np.ndarray | None = None,
    max_depth: float = 300.0,
) -> np.ndarray:
    """Per-pixel metric depth obtained by intersecting back-projection rays
    with the world-frame ground plane.

    Pixels in ``non_ground_mask == True`` are returned as NaN.
    """
    raise NotImplementedError("Stage 2")
