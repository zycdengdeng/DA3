"""Shared config building blocks for every stage.

Each stage config composes one ``SceneConfig`` (where the data lives),
one ``OutputConfig`` (where artifacts go), and one ``RuntimeConfig``
(workers, GPUs, seeds). Stage-specific knobs live in
``configs/stages/<stage>.py``.

These are plain ``@dataclass`` types: tyro generates the CLI from them
and stages consume them through attribute access. Avoid putting
behaviour here — configs describe *what*, stages do the *how*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SceneConfig:
    """Where the raw V2X data lives and which scene to process."""

    scene: str
    """Scene id; e.g. ``"008"`` or the full ``"008_car..._t8"``."""

    data_root: Path = Path("/mnt/car_road_data_TianJin")
    """Filesystem root of the V2X dataset."""

    cams: tuple[str, ...] = ("0", "3", "6", "9")
    """Camera ids participating in this scene."""

    loader_min_points: int = 0
    """Drop frames whose LiDAR has fewer points than this."""

    static_labels_source: Literal["interpolation", "merged_pcd"] = "merged_pcd"
    """V2X label folder for *static-cloud* bbox masking. Hand-labeled
    ``merged_pcd`` (1 Hz, ``road_labels/merged_pcd_all/``) is more
    accurate than the ``interpolation`` set (10 Hz,
    ``road_labels/interpolation_labels/``) — interpolation jitter at
    bbox edges is what causes the "trail of car points in the static
    cloud" artifact. Fall back to ``"interpolation"`` if a scene has
    no hand-labels."""

    dynamic_labels_source: Literal["interpolation", "merged_pcd"] = (
        "interpolation"
    )
    """V2X label folder for *dynamic-object* handling (per-object
    accumulation, snapshot injection, video BEV). 10 Hz interpolation
    is needed for smooth motion across video frames; the 1 Hz hand-
    labels are too sparse here. Only switch to ``"merged_pcd"`` for
    debugging."""


@dataclass
class OutputConfig:
    """Where this stage's artifacts go.

    Concrete output dir is resolved by :class:`engine.runner.OutputManager`
    to ``<root>/<scene>/<stage>/<run_id>/``; a ``latest`` symlink in the
    sibling slot points at the most recent run.
    """

    root: Path = Path("outputs")
    """Top-level outputs dir; everything below is by-convention."""

    run_id: str | None = None
    """Override the auto-generated ``YYYY-MM-DD_HH-MM-SS`` run id."""

    overwrite_latest: bool = True
    """Re-point the ``latest`` symlink at this run on success."""


@dataclass
class RuntimeConfig:
    """Parallelism + reproducibility knobs that every stage understands."""

    gpu_ids: tuple[int, ...] = field(default_factory=tuple)
    """GPU ids for sharding; empty = serial on the default device."""

    workers: int = 0
    """CPU workers for embarrassingly-parallel CPU work; 0 = serial."""

    seed: int = 0
    """Base seed; per-item seeds are derived inside each stage."""
