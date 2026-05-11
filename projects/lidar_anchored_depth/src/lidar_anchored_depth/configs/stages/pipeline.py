"""Config for the ``pipeline`` orchestrator (runs multiple stages back-to-back).

Composes the three single-stage configs into one. The user sets
``scene`` / ``output`` / ``runtime`` once on the ``--complete.*`` block;
the orchestrator propagates those values into ``inject`` and
``render-bev`` automatically. Per-stage flags can still be overridden
with ``--inject.<flag>`` / ``--render-bev.<flag>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lidar_anchored_depth.configs.base import SceneConfig
from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.configs.stages.inject import InjectConfig
from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig

StageName = Literal["complete", "inject", "render-bev"]


def _placeholder_scene() -> SceneConfig:
    """Used as the default for inject / render_bev's required ``scene``
    field. The orchestrator overrides it with the complete-stage scene
    before running, so users never see this string."""
    return SceneConfig(scene="__from_complete__")


@dataclass
class FullPipelineConfig:
    """Multi-stage orchestrator: ``complete`` -> ``inject`` -> ``render-bev``.

    The user typically writes only the ``--complete.*`` block; the
    orchestrator propagates that block's ``scene`` / ``output`` /
    ``runtime`` to the downstream stages. Pass ``--from-stage`` to
    resume after a failure, ``--stages`` to run a subset.
    """

    complete: CompleteConfig
    """Stage 1 config; its scene/output/runtime are the canonical ones."""

    inject: InjectConfig = field(
        default_factory=lambda: InjectConfig(scene=_placeholder_scene()),
    )
    """Stage 2 config. ``scene``/``output``/``runtime`` are auto-filled
    from ``complete`` unless explicitly overridden."""

    render_bev: BevRenderConfig = field(
        default_factory=lambda: BevRenderConfig(scene=_placeholder_scene()),
    )
    """Stage 3 config. Same auto-fill rule."""

    stages: tuple[StageName, ...] = ("complete", "inject", "render-bev")
    """Which stages to run, in order."""

    from_stage: StageName | None = None
    """Skip stages listed in ``stages`` before this one (resume helper)."""
