"""Height-Anchored Depth (HAD) — paper main method.

This module is intentionally a stub at Stage 1: it pins down the public API
so config / pipeline / tests can be wired up. The actual numerics land in
Stage 2 alongside unit tests against known geometry.

See ``docs/method.md`` §3 for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class InstanceAnchor:
    """Two metric-depth anchors for a single instance mask.

    Attributes
    ----------
    instance_id
        Index into the originating mask stack.
    pixel_top, pixel_bot
        ``(u, v)`` pixel coordinates of the chosen anchor pixels.
    z_top, z_bot
        Camera-axis depth (meters) at the anchor pixels, derived by
        intersecting the back-projection ray with world-Z planes
        ``Z = Z_max`` and ``Z = Z_min`` respectively.
    z_min_world, z_max_world
        World-frame height interval estimated from LiDAR returns inside the
        mask.
    confidence
        Heuristic in [0, 1] reflecting (a) number of LiDAR points used,
        (b) tightness of the height interval after outlier rejection,
        (c) whether top/bot pixels touched the image border.
    """

    instance_id: int
    pixel_top: tuple[int, int]
    pixel_bot: tuple[int, int]
    z_top: float
    z_bot: float
    z_min_world: float
    z_max_world: float
    confidence: float


@dataclass(slots=True)
class HADResult:
    """Output of running HAD on one frame."""

    aligned_depth: np.ndarray  # (H, W) float32 metric depth, NaN where not covered
    anchors: list[InstanceAnchor]
    coverage_mask: np.ndarray  # (H, W) bool, True where HAD wrote a value


def extract_instance_height(
    lidar_world: np.ndarray,
    pixel_uv: np.ndarray,
    mask: np.ndarray,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_points: int = 5,
) -> tuple[float, float, float] | None:
    """Estimate ``(Z_min, Z_max, confidence)`` for one instance mask.

    Parameters
    ----------
    lidar_world
        ``(N, 3)`` LiDAR points in world frame.
    pixel_uv
        ``(N, 2)`` corresponding pixel projections (one row per LiDAR point;
        rows whose point lies outside the image should be filtered upstream
        or marked with NaN — to be specified at implementation time).
    mask
        ``(H, W)`` boolean instance mask.

    Returns
    -------
    Tuple of ``(Z_min, Z_max, confidence)`` in meters / [0, 1], or ``None``
    if there are fewer than ``min_points`` LiDAR returns inside the mask.

    Notes
    -----
    Implementation deferred to Stage 2; this stub raises ``NotImplementedError``.
    """
    raise NotImplementedError("Stage 2: implement height-interval extraction")


def solve_per_instance_affine(
    d_pred: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    masks: np.ndarray,
    lidar_world: np.ndarray,
) -> HADResult:
    """Run HAD on a frame.

    Parameters
    ----------
    d_pred
        ``(H, W)`` DA3 relative depth (positive, dimensionless).
    K, T_wc
        Camera intrinsics ``(3, 3)`` and camera-to-world ``(4, 4)``.
    masks
        ``(K, H, W)`` boolean instance masks.
    lidar_world
        ``(N, 3)`` LiDAR points in world frame.

    Returns
    -------
    :class:`HADResult` with metric depth on covered pixels and per-instance
    anchor diagnostics.

    Notes
    -----
    Stage 2 implementation. The numerical recipe follows ``docs/method.md``
    §3.2–§3.4 verbatim.
    """
    raise NotImplementedError("Stage 2: implement HAD solve")
