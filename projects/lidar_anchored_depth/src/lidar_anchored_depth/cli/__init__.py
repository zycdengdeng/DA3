"""``lad`` top-level CLI.

This module is the entry point declared by ``pyproject.toml``:

    [project.scripts]
    lad = "lidar_anchored_depth.cli:main"

After ``pip install -e .``, the ``lad`` binary in the active env's
``bin/`` runs :func:`main` below. Subcommands are dispatched by
:mod:`tyro` from a tagged union of config dataclasses; the dispatcher
just calls the matched config's ``run()`` method, so adding a new
subcommand is one Union member + one ``run()`` method away.

Phase 1 ships a single ``lad info`` subcommand that prints the
resolved config — enough to prove the wiring works end to end.
Phase 2+ will wire real stages (``lad depth``, ``lad complete``,
``lad render-bev``, ...).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Union

import tyro

from lidar_anchored_depth import __version__ as _pkg_version
from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)
from lidar_anchored_depth.configs.stages.calib import CalibConfig
from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.configs.stages.depth import DepthConfig
from lidar_anchored_depth.configs.stages.inject import InjectConfig
from lidar_anchored_depth.configs.stages.mask import MaskConfig
from lidar_anchored_depth.configs.stages.object_accum import ObjectAccumConfig
from lidar_anchored_depth.configs.stages.pipeline import FullPipelineConfig
from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig
from lidar_anchored_depth.configs.stages.seg import SegConfig
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.pipelines.full_pipeline import FullPipeline
from lidar_anchored_depth.stages.bev_render import BevRenderStage
from lidar_anchored_depth.stages.da3_depth import DA3DepthStage
from lidar_anchored_depth.stages.dense_completion import DenseCompletionStage
from lidar_anchored_depth.stages.dynamic_inject import DynamicInjectStage
from lidar_anchored_depth.stages.object_accum import ObjectAccumStage
from lidar_anchored_depth.stages.sam_mask import SAMMaskStage
from lidar_anchored_depth.stages.segformer_seg import SegFormerStage
from lidar_anchored_depth.stages.static_calib import StaticCalibStage


# ---------------------------------------------------------------- info
#
# Stub subcommand that introspects the environment and returns 0. Used
# to smoke-test the CLI plumbing (``lad info`` resolves, ``lad info
# --help`` shows the right tree) without touching GPU / dataset.

@dataclass
class InfoCmd:
    """Print package metadata and the resolved config tree."""

    scene: SceneConfig | None = None
    """Optional scene block; when given, its values are printed."""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output dir spec — only displayed; no files are written."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    """Runtime / parallelism block — only displayed."""

    def run(self) -> int:
        payload = {
            "package": "lidar-anchored-depth",
            "version": _pkg_version,
            "python": sys.version.split()[0],
            "argv": sys.argv,
            "scene": _dump(self.scene),
            "output": _dump(self.output),
            "runtime": _dump(self.runtime),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0


@dataclass
class VersionCmd:
    """Print just the package version (for scripts that need to parse it)."""

    def run(self) -> int:
        print(_pkg_version)
        return 0


# --------------------------------------------------------------- complete
#
# Wrap CompleteConfig with a thin ``run()`` method so the CLI dispatch
# loop can call ``cmd.run()`` uniformly. Subclassing keeps the tyro CLI
# identical to bare CompleteConfig (no extra options injected).

def _execute_stage(cmd, stage_cls) -> int:
    """Common runner: provision OutputManager, dump config, run stage,
    write summary, refresh latest symlink. Used by every stage CLI."""
    om = OutputManager(
        cmd.output.root,
        cmd.scene.scene,
        stage_cls.name,
        run_id=cmd.output.run_id,
        update_latest=cmd.output.overwrite_latest,
    )
    om.dump_config(cmd)
    print(f"[output] run_dir = {om.run_dir}", flush=True)

    stage = stage_cls(cfg=cmd, output_dir=om.run_dir)
    artifacts = stage.run()

    summary_path = om.run_dir / "summary.json"
    summary_path.write_text(json.dumps(_dump(artifacts.summary), indent=2))
    om.finalise()
    print(f"[done] artifacts -> {om.latest_link}")
    return 0


# -- upstream stages --------------------------------------------------

@dataclass
class DepthCmd(DepthConfig):
    """DA3 monocular depth inference. Produces *_d.npz per (scene, ts, cam)."""

    def run(self) -> int:
        return _execute_stage(self, DA3DepthStage)


@dataclass
class MaskCmd(MaskConfig):
    """SAM dynamic-object masks. Produces *_mask.npz per (scene, ts, cam)."""

    def run(self) -> int:
        return _execute_stage(self, SAMMaskStage)


@dataclass
class SegCmd(SegConfig):
    """SegFormer Cityscapes 19-class segmentation."""

    def run(self) -> int:
        return _execute_stage(self, SegFormerStage)


@dataclass
class CalibCmd(CalibConfig):
    """Static scene reconstruction + per-camera AA-HAD ``(a, b)`` calibration."""

    def run(self) -> int:
        return _execute_stage(self, StaticCalibStage)


@dataclass
class ObjectAccumCmd(ObjectAccumConfig):
    """Per-object accumulation: per dynamic V2X object id, accumulate LiDAR +
    AA-HAD prediction across all frames into one per-object PLY in the
    object's local frame."""

    def run(self) -> int:
        return _execute_stage(self, ObjectAccumStage)


# -- downstream stages ------------------------------------------------

@dataclass
class CompleteCmd(CompleteConfig):
    """Run point-cloud completion + Layer 4 LiDAR-priority fuse on one scene."""

    def run(self) -> int:
        return _execute_stage(self, DenseCompletionStage)


@dataclass
class InjectCmd(InjectConfig):
    """Inject per-object accumulated snapshots into the static cloud
    at one anchor timestamp.

    Auto-reads ``outputs/<scene>/complete/latest/hybrid.ply`` unless
    ``--static-ply`` overrides.
    """

    def run(self) -> int:
        return _execute_stage(self, DynamicInjectStage)


@dataclass
class RenderBevCmd(BevRenderConfig):
    """Render per-ts BEV PNGs with linear pose-interpolated dynamic
    objects. Output PNGs feed straight into ``ffmpeg``."""

    def run(self) -> int:
        return _execute_stage(self, BevRenderStage)


@dataclass
class PipelineCmd(FullPipelineConfig):
    """Run complete -> inject -> render-bev back-to-back.

    Set ``--complete.scene.scene 008`` etc. once and the orchestrator
    propagates those to the downstream stages. Use ``--stages`` to
    pick a subset and ``--from-stage`` to resume after a failure.
    """

    def run(self) -> int:
        return FullPipeline(cfg=self).run()


def _dump(obj: object) -> object:
    """Lightweight recursive dataclass → dict for pretty-printing.

    A local copy of the engine helper so ``lad info`` works even
    before ``engine`` is fully imported (avoids import cycles down
    the road).
    """
    from dataclasses import fields, is_dataclass

    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dump(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dump(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------- dispatch

# tyro builds the subcommand parser from this tagged union: each member
# becomes one ``lad <name>`` subcommand. Adding a new stage in later
# phases is one ``Annotated[NewCmd, tyro.conf.subcommand(...)]`` entry.
Subcommand = Union[
    Annotated[
        InfoCmd,
        tyro.conf.subcommand(
            name="info",
            description="Print package + config info (smoke-test the CLI).",
        ),
    ],
    Annotated[
        VersionCmd,
        tyro.conf.subcommand(
            name="version",
            description="Print just the package version string.",
        ),
    ],
    Annotated[
        DepthCmd,
        tyro.conf.subcommand(
            name="depth",
            description=(
                "Upstream: DA3 monocular depth inference. Multi-GPU "
                "via --runtime.gpu-ids (--ts-shard fan-out)."
            ),
        ),
    ],
    Annotated[
        MaskCmd,
        tyro.conf.subcommand(
            name="mask",
            description=(
                "Upstream: SAM dynamic-object masks prompted by V2X "
                "bboxes. Multi-GPU shards by camera."
            ),
        ),
    ],
    Annotated[
        SegCmd,
        tyro.conf.subcommand(
            name="seg",
            description=(
                "Upstream: SegFormer Cityscapes 19-class segmentation. "
                "Multi-GPU via --ts-shard fan-out."
            ),
        ),
    ],
    Annotated[
        CalibCmd,
        tyro.conf.subcommand(
            name="calib",
            description=(
                "Upstream: static scene reconstruction + per-camera "
                "AA-HAD (a, b) calibration. Single-GPU."
            ),
        ),
    ],
    Annotated[
        ObjectAccumCmd,
        tyro.conf.subcommand(
            name="object-accum",
            description=(
                "Upstream: per-object accumulation. For every dynamic "
                "V2X object id, accumulate LiDAR + AA-HAD across all "
                "frames into one per-object local-frame PLY."
            ),
        ),
    ],
    Annotated[
        CompleteCmd,
        tyro.conf.subcommand(
            name="complete",
            description=(
                "Stage 2/4: per-frame point-completion network + Layer 4 "
                "LiDAR-priority fuse. Multi-GPU via --runtime.gpu-ids."
            ),
        ),
    ],
    Annotated[
        InjectCmd,
        tyro.conf.subcommand(
            name="inject",
            description=(
                "Stage 5: inject per-object snapshots at one anchor ts. "
                "Reads complete/latest/hybrid.ply by convention."
            ),
        ),
    ],
    Annotated[
        RenderBevCmd,
        tyro.conf.subcommand(
            name="render-bev",
            description=(
                "Render per-ts BEV PNGs with linear pose-interpolated "
                "dynamic objects. Feed into ffmpeg for video assembly."
            ),
        ),
    ],
    Annotated[
        PipelineCmd,
        tyro.conf.subcommand(
            name="pipeline",
            description=(
                "Run complete -> inject -> render-bev back-to-back. "
                "Shared scene/output/runtime propagate from --complete.* "
                "to the downstream stages."
            ),
        ),
    ],
]


def main() -> int:
    cmd = tyro.cli(
        Subcommand,
        prog="lad",
        description=(
            "LiDAR-Anchored Depth toolkit. Subcommands are listed below; "
            "try `lad <subcommand> --help` for stage-specific options."
        ),
    )
    return int(cmd.run())


if __name__ == "__main__":
    raise SystemExit(main())
