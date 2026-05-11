"""End-to-end orchestrator: ``complete`` -> ``inject`` -> ``render-bev``.

The orchestrator is deliberately a thin shell around the stage classes
— it does no domain logic of its own. Each stage still resolves its
own inputs (e.g. inject auto-discovers ``complete/latest/hybrid.ply``),
so a successful pipeline is equivalent to running the three CLIs in
sequence by hand.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

from lidar_anchored_depth.configs.stages.pipeline import (
    FullPipelineConfig,
    StageName,
)
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.stages.base import Stage
from lidar_anchored_depth.stages.bev_render import BevRenderStage
from lidar_anchored_depth.stages.dense_completion import DenseCompletionStage
from lidar_anchored_depth.stages.dynamic_inject import DynamicInjectStage


def _to_serialisable(obj: Any) -> Any:
    """Local copy of cli._dump to avoid a circular import from pipelines."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_serialisable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    return obj


class FullPipeline:
    """Run multiple stages back-to-back.

    Each stage is provisioned its own ``OutputManager``, so artifacts
    land in ``<output.root>/<scene>/<stage>/<run_id>/`` and the
    sibling ``latest`` symlinks are refreshed on success — downstream
    stages discover their inputs through these links.
    """

    REGISTRY: ClassVar[dict[StageName, type[Stage]]] = {
        "complete": DenseCompletionStage,
        "inject": DynamicInjectStage,
        "render-bev": BevRenderStage,
    }

    def __init__(self, cfg: FullPipelineConfig) -> None:
        self.cfg = cfg

    def _stage_cfg(self, name: StageName):
        """Return the dataclass config for ``name``, with the shared
        scene / output / runtime blocks copied in from ``complete``."""
        if name == "complete":
            return self.cfg.complete
        base = self.cfg.inject if name == "inject" else self.cfg.render_bev
        return dataclasses.replace(
            base,
            scene=self.cfg.complete.scene,
            output=self.cfg.complete.output,
            runtime=self.cfg.complete.runtime,
        )

    def _stages_to_run(self) -> list[StageName]:
        stages = list(self.cfg.stages)
        if self.cfg.from_stage is None:
            return stages
        if self.cfg.from_stage not in stages:
            raise SystemExit(
                f"--from-stage {self.cfg.from_stage!r} is not in "
                f"--stages {stages}"
            )
        idx = stages.index(self.cfg.from_stage)
        return stages[idx:]

    def run(self) -> int:
        stages = self._stages_to_run()
        print(f"[pipeline] running stages: {' -> '.join(stages)}", flush=True)

        for name in stages:
            stage_cls = self.REGISTRY[name]
            stage_cfg = self._stage_cfg(name)

            print()
            print(
                f"==[ stage: {name} ]"
                + "=" * max(0, 60 - len(name) - 12)
            )

            om = OutputManager(
                stage_cfg.output.root,
                stage_cfg.scene.scene,
                stage_cls.name,
                run_id=stage_cfg.output.run_id,
                update_latest=stage_cfg.output.overwrite_latest,
            )
            om.dump_config(stage_cfg)
            print(f"[output] run_dir = {om.run_dir}", flush=True)

            stage = stage_cls(cfg=stage_cfg, output_dir=om.run_dir)
            artifacts = stage.run()

            (om.run_dir / "summary.json").write_text(
                json.dumps(_to_serialisable(artifacts.summary), indent=2)
            )
            om.finalise()
            print(f"[done] {name} -> {om.latest_link}", flush=True)

        print()
        print("[pipeline] all stages done", flush=True)
        return 0
