"""Abstract base class for pipeline stages.

A stage is a single self-contained step in the pipeline (DA3 depth,
SAM masks, point completion, BEV render, ...). The contract is:

* Construct with a frozen config dataclass.
* :meth:`Stage.run` performs the work, writes artifacts to
  ``self.output_dir`` (provisioned by the engine), and returns a
  :class:`StageArtifacts` describing what was produced. The same
  config + same input → same artifacts (modulo a deterministic seed).
* Stages must be pure with respect to filesystem state outside
  ``self.output_dir`` — no in-place mutation of inputs.

Concrete stages live in sibling modules and are wired into the CLI
in ``lidar_anchored_depth.cli``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar


ConfigT = TypeVar("ConfigT")


@dataclass
class StageArtifacts:
    """Pointers to what a stage wrote.

    Stage implementations may subclass this to expose typed accessors
    (e.g. ``DA3DepthArtifacts.depth_npz_glob``), but the base class
    holds the generic fields that the engine and downstream stages
    consume.
    """

    output_dir: Path
    """Root dir where the stage's artifacts live."""

    files: dict[str, Path] = field(default_factory=dict)
    """Named pointers to key files (``"summary_json"``, etc.)."""

    summary: dict[str, object] = field(default_factory=dict)
    """Free-form metrics / counts the stage wants reported."""


class Stage(abc.ABC, Generic[ConfigT]):
    """Base class for every pipeline stage."""

    name: str
    """Short stage id used in output paths (``"depth"``, ``"complete"``)."""

    def __init__(self, cfg: ConfigT, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir

    @abc.abstractmethod
    def run(self) -> StageArtifacts:
        """Execute the stage. Returns artifacts; raises on failure."""
