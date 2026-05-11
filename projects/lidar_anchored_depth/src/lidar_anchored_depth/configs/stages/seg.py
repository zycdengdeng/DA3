"""Config for the ``seg`` stage (SegFormer 19-class segmentation)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class SegConfig:
    """Run SegFormer over each frame; produces per-pixel Cityscapes
    19-class IDs that the point-completion network uses as an A1 feature
    (segment-aware EdgeConv masking).

    Output: ``<scene>_ts<ms>_cam<X>_seg.npz`` per (scene, ts, cam).
    """

    scene: SceneConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    """HF model id OR a local path. Prefer the local path when the
    network is flaky — download once via huggingface-cli, point here."""

    save_viz: bool = False
    """Also save a coloured Cityscapes-palette overlay PNG per frame."""

    skip_existing: bool = True
    """Resume by skipping (scene, ts, cam) triples whose .npz exists."""
