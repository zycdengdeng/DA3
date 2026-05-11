"""Config for the ``calib`` stage (static scene reconstruction + AA-HAD calib)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class CalibConfig:
    """Run static scene reconstruction over (scene, ts, cam), fit
    AA-HAD ``(a, b)`` per-camera against LiDAR ground anchors, and
    write the static-cloud PLYs.

    Main output that downstream stages consume:
        ``<scene>_<...>_static_calib.json`` — per-camera ``(a, b)``.

    Side outputs (for inspection / training):
        ``<scene>_static.ply``, ``<scene>_lidar_static.ply``,
        ``<scene>_hybrid.ply`` and friends.
    """

    scene: SceneConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    d_paths_glob: str | None = None
    """DA3 ``.npz`` glob. Auto: ``outputs/<scene>/depth/latest/*_d.npz``."""

    sam_mask_dir: Path | None = None
    """SAM mask dir (strongly recommended — without it dynamic pixels
    leak into the static AA-HAD output). Auto:
    ``outputs/<scene>/mask/latest/``."""

    aahad_pixel_stride: int = 2
    """Stride on depth pixels when building the static prior."""

    voxel_size: float = 0.05
    """Voxel-grid downsample size (m) for the final aggregate cloud."""

    z_min: float = 0.5
    z_max: float = 300.0
    sam_dilate_px: int = 0
    """Dilate the SAM dynamic-mask union by this many pixels before
    subtracting it. DA3 depth can bleed past the SAM edge — 5-10 px
    catches the residual halo."""

    region_grid_rows: int = 0
    region_grid_cols: int = 0
    """If both > 0, replace the single per-camera (a, b) with a per-
    image-cell affine fit. Recommended for 1080p frames: 6 8."""

    ply_binary: bool = True
    """Write PLYs as binary (faster + smaller for large clouds)."""
