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
from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.stages.dense_completion import DenseCompletionStage


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

@dataclass
class CompleteCmd(CompleteConfig):
    """Run point-cloud completion + Layer 4 LiDAR-priority fuse on one scene."""

    def run(self) -> int:
        om = OutputManager(
            self.output.root,
            self.scene.scene,
            DenseCompletionStage.name,
            run_id=self.output.run_id,
            update_latest=self.output.overwrite_latest,
        )
        om.dump_config(self)
        print(f"[output] run_dir = {om.run_dir}", flush=True)

        stage = DenseCompletionStage(cfg=self, output_dir=om.run_dir)
        artifacts = stage.run()

        summary_path = om.run_dir / "summary.json"
        summary_path.write_text(json.dumps(_dump(artifacts.summary), indent=2))
        om.finalise()
        print(f"[done] artifacts -> {om.latest_link}")
        return 0


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
        CompleteCmd,
        tyro.conf.subcommand(
            name="complete",
            description=(
                "Stage 2/4: per-frame point-completion network + Layer 4 "
                "LiDAR-priority fuse. Multi-GPU via --runtime.gpu-ids."
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
