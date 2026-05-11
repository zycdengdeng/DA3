"""Config for the ``depth`` stage (DA3 monocular relative-depth inference)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class DepthConfig:
    """Run DA3 on every (ts, cam) of one scene.

    Produces ``<scene>_ts<ms>_cam<X>_d.npz`` files containing the
    relative-depth tensor, which downstream stages consume via the
    AA-HAD calibration to get metric depth.
    """

    scene: SceneConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    model: str = "depth-anything/DA3Mono-Large"
    """HuggingFace model id (or local path). Pick the DA3Mono-Large for
    the strongest relative depth; DA3Metric-Large for metric heads."""

    process_res: int = 504
    """DA3 internal processing resolution."""

    skip_existing: bool = True
    """Resume crashed runs by skipping (scene, ts, cam) triples whose
    .npz already lives in the output dir."""
