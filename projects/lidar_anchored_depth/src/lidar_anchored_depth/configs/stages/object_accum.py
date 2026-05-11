"""Config for the ``object-accum`` stage (per-object 3I accumulation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class ObjectAccumConfig:
    """For every dynamic V2X object id in the scene, accumulate the
    per-frame LiDAR + AA-HAD dense reconstruction into one per-object
    PLY in the object's local frame.

    Output: ``<scene>_obj<id>_*.ply`` per dynamic object.
    """

    scene: SceneConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    d_paths_glob: str | None = None
    """DA3 ``.npz`` glob. Auto: ``outputs/<scene>/depth/latest/*_d.npz``."""

    sam_mask_dir: Path | None = None
    """SAM mask dir (output of ``lad mask``). Without this, only the
    LiDAR side is accumulated — AA-HAD prediction is skipped. Auto:
    ``outputs/<scene>/mask/latest/``."""

    voxel_size: float = 0.05
    """Voxel size (m) for per-object downsampling."""

    bbox_expand: float = 0.10
    """Expand bbox half-extents by this many metres when picking
    LiDAR points inside the bbox."""

    ground_cut_m: float = 0.10
    """Drop the bottom Nm of the bbox volume from both LiDAR and AA-HAD
    clouds — V2X bboxes usually wrap a thin road-surface band beneath
    the vehicle which would otherwise pollute the object reconstruction."""

    bbox_clip_expand: float = 0.30
    """Slack (m) when clipping AA-HAD predicted points to lie inside
    the V2X 3D bbox. Tighten to 0.10 for cleaner volumes."""

    solver_mode: Literal[
        "per-frame", "per-camera",
        "per-camera-lidar", "per-camera-lidar-offset",
    ] = "per-camera"
    """How to solve the per-object (a, b). 'per-camera' is the
    recommended balanced default. 'per-camera-lidar-offset' (M5b) is
    the recommended headline mode."""

    lidar_color: Literal["height", "red", "white", "none"] = "height"
    """How to colour the LiDAR PLY (which has no native RGB)."""

    z_min: float = 0.5
    z_max: float = 250.0
    dense_min_pixels: int = 50
    ply_binary: bool = True

    workers: int = 16
    """Per-object ProcessPoolExecutor size. Default 16; bump to 32+
    on a many-core box. 0 falls back to deterministic serial
    execution. Per-object work is independent (no shared RNG, no
    ordering requirement) so the parallel result is bit-exact."""

    skip_existing: bool = True
    """Skip bbox ids whose ``<scene>_obj<id>_lidar.ply`` AND
    ``<scene>_obj<id>_aa_had.ply`` already exist in the run dir.
    Lets a killed run resume without re-doing the finished objects."""
