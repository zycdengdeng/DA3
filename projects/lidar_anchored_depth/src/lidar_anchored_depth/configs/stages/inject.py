"""Config for the ``inject`` stage (Layer 5, single anchor snapshot).

Composes one ``SceneConfig`` + auto-discovers the upstream
``complete`` stage's ``hybrid.ply`` from
``<output.root>/<scene>/complete/latest/`` (override with
``--static-ply``).
"""

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
class InjectConfig:
    """Inject per-object accumulated snapshots into the static cloud
    at a single anchor timestamp.

    Inputs
    ------
    A static cloud (typically ``complete/latest/hybrid.ply``) and a
    directory of per-object accumulated PLYs from
    ``run_object_accumulation.py``.

    Outputs
    -------
    ``hybrid_with_dyn.ply`` and ``inject_summary.json``.
    """

    scene: SceneConfig

    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---- inputs ----
    recon_dir: Path = Path("preview/recon_3I_full")
    """Dir of per-object PLYs from run_object_accumulation.py."""

    static_ply: Path | None = None
    """Override the auto-discovered static cloud
    (default: ``<output.root>/<scene>/complete/latest/hybrid.ply``)."""

    anchor_ts_ms: int = 0
    """Anchor timestamp (ms) to place all dynamic objects at."""

    # ---- injection knobs ----
    strict_anchor: bool = True
    """Drop objects with no V2X bbox at the anchor ts."""

    object_fusion: Literal["lidar-priority", "lidar-only", "aa-had-only"] = (
        "lidar-priority"
    )
    """How to fuse per-object LiDAR with the accumulated AA-HAD reconstruction."""

    object_voxel_size: float = 0.05
    """Voxel size for per-object cloud downsampling (m)."""

    object_max_dist_to_lidar: float = 0.5
    """Drop AA-HAD points farther than this from per-object LiDAR (m)."""

    mirror_axis: Literal["x", "y", "z"] | None = None
    """If set, mirror per-object AA-HAD across this local-frame axis
    before fusing with LiDAR (useful when only one side of a vehicle
    was observed)."""

    mirror_classes: tuple[str, ...] = ("Car", "Suv", "Bus", "Truck")
    """V2X labels to apply mirror_axis to."""

    motion_drift_threshold_m: float = 0.5
    """Movement (m) between min/max ts beyond which an object is
    classified as dynamic (anchor-only) vs static (always-on)."""

    # ---- output ----
    ply_binary: bool = False
    """Write hybrid_with_dyn.ply as binary."""
