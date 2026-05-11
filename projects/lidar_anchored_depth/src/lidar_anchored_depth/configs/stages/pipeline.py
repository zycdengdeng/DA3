"""Config for the ``pipeline`` orchestrator (runs multiple stages back-to-back).

Composes every single-stage config the project ships, in the order
they should run end-to-end. The user sets ``scene`` / ``output`` /
``runtime`` once on the ``--complete.*`` block (the canonical block);
the orchestrator propagates those values into every other stage at
runtime. Per-stage flags can still be overridden with
``--<stage>.<flag>`` (e.g. ``--depth.model``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lidar_anchored_depth.configs.base import SceneConfig
from lidar_anchored_depth.configs.stages.calib import CalibConfig
from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.configs.stages.depth import DepthConfig
from lidar_anchored_depth.configs.stages.inject import InjectConfig
from lidar_anchored_depth.configs.stages.mask import MaskConfig
from lidar_anchored_depth.configs.stages.object_accum import ObjectAccumConfig
from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig
from lidar_anchored_depth.configs.stages.seg import SegConfig

StageName = Literal[
    "depth", "mask", "seg", "calib", "object-accum",
    "complete", "inject", "render-bev",
]


def _placeholder_scene() -> SceneConfig:
    """Default scene for non-canonical stage configs. The orchestrator
    overrides this with the complete-stage scene before running, so
    users never see the placeholder string."""
    return SceneConfig(scene="__from_complete__")


@dataclass
class FullPipelineConfig:
    """Multi-stage orchestrator: upstream depth/mask/seg/calib/object-
    accum → downstream complete → inject → render-bev.

    By default the pipeline runs only the *downstream* three stages
    (so existing users with pre-computed upstream artefacts in
    ``preview/`` keep working as-is). Add the upstream names to
    ``--stages`` (or ``--from-stage depth``) to start from raw data.
    """

    complete: CompleteConfig
    """Canonical block — its scene/output/runtime propagate to every
    other stage."""

    depth: DepthConfig = field(
        default_factory=lambda: DepthConfig(scene=_placeholder_scene()),
    )
    mask: MaskConfig = field(
        default_factory=lambda: MaskConfig(scene=_placeholder_scene()),
    )
    seg: SegConfig = field(
        default_factory=lambda: SegConfig(scene=_placeholder_scene()),
    )
    calib: CalibConfig = field(
        default_factory=lambda: CalibConfig(scene=_placeholder_scene()),
    )
    object_accum: ObjectAccumConfig = field(
        default_factory=lambda: ObjectAccumConfig(scene=_placeholder_scene()),
    )
    inject: InjectConfig = field(
        default_factory=lambda: InjectConfig(scene=_placeholder_scene()),
    )
    render_bev: BevRenderConfig = field(
        default_factory=lambda: BevRenderConfig(scene=_placeholder_scene()),
    )

    stages: tuple[StageName, ...] = ("complete", "inject", "render-bev")
    """Which stages to run, in order. Default skips the upstream stages
    (assumes preview/ products exist). Set to
    ('depth','mask','seg','calib','object-accum','complete','inject','render-bev')
    to drive the whole pipeline from raw data."""

    from_stage: StageName | None = None
    """Skip stages listed in ``stages`` before this one (resume helper)."""
