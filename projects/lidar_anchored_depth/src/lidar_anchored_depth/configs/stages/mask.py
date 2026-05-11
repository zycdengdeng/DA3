"""Config for the ``mask`` stage (SAM dynamic-object masks)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class MaskConfig:
    """Run SAM on each frame, prompted with the V2X bbox AABBs, to
    produce per-frame dynamic-object masks.

    Output: ``<scene>_ts<ms>_cam<X>_mask.npz`` per (scene, ts, cam).
    """

    scene: SceneConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    model_id: str = "facebook/sam-vit-huge"
    """HuggingFace SAM model id. Try sam-vit-base for fast iteration."""

    bbox_pad_px: int = 8
    """Pad each projected V2X bbox by this many pixels before sending
    to SAM — helps SAM see the full object silhouette."""

    multimask: bool = False
    """Ask SAM for 3 candidate masks per box; keep the best-score one."""

    max_boxes_per_batch: int = 64
    """SAM batch-prediction chunk size. Lower if you hit VRAM limits."""
