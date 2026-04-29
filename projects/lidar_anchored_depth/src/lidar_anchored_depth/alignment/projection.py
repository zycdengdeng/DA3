"""LiDAR ↔ image projection primitives.

Stage 1 stub: pins down the public API. Stage 2 will port the vetted
implementation from the prior ``utils/lidar_alignment.py`` (NumPy) and
``align_depth_with_lidar_torch`` (Torch, differentiable), with unit tests
against synthetic ground truth.
"""

from __future__ import annotations

import numpy as np


def project_world_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    image_hw: tuple[int, int],
    *,
    min_depth: float = 0.1,
    max_depth: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a pinhole camera image.

    Returns
    -------
    pixel_uv : (M, 2) float64
        Pixel coordinates inside image bounds.
    z_cam : (M,) float64
        Camera-axis depth of the projected points.
    indices : (M,) int64
        Indices into ``points_world`` of the points that survived all
        validity checks (in front of camera and inside image bounds).
    """
    raise NotImplementedError("Stage 2")
