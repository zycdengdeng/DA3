"""Output directory management for stages.

Resolves a stage's output dir as
``<root>/<scene>/<stage>/<run_id>/`` and maintains a sibling
``latest`` symlink that points at the most recent run.

Downstream stages can discover their input by reading
``<root>/<scene>/<prev_stage>/latest/``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _auto_run_id() -> str:
    """``YYYY-MM-DD_HH-MM-SS`` — sortable, filesystem-safe."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _to_serialisable(obj: Any) -> Any:
    """Convert dataclasses / Paths / tuples into JSON-friendly types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_serialisable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    return obj


class OutputManager:
    """Provisions and tracks one stage's output directory.

    Parameters
    ----------
    root :
        Top-level outputs dir, e.g. ``outputs/``.
    scene :
        Scene id; first level under ``root``.
    stage :
        Stage id; second level. Use the stage's ``name`` attribute.
    run_id :
        Optional explicit run id; falls back to a current-time stamp.
    update_latest :
        If True, update the ``latest`` symlink to point at this run
        after :meth:`finalise` is called.
    """

    def __init__(
        self,
        root: Path,
        scene: str,
        stage: str,
        *,
        run_id: str | None = None,
        update_latest: bool = True,
    ) -> None:
        self.root = Path(root)
        self.scene = scene
        self.stage = stage
        self.run_id = run_id or _auto_run_id()
        self.update_latest = update_latest

        self.stage_dir = self.root / scene / stage
        self.run_dir = self.stage_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_link(self) -> Path:
        return self.stage_dir / "latest"

    def dump_config(self, cfg: Any, filename: str = "config.json") -> Path:
        """Serialise the config dataclass to JSON for reproducibility."""
        path = self.run_dir / filename
        path.write_text(json.dumps(_to_serialisable(cfg), indent=2))
        return path

    def finalise(self) -> None:
        """Mark the run as complete and refresh ``latest`` symlink."""
        if not self.update_latest:
            return
        link = self.latest_link
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
        except OSError:
            pass
        # Use a relative target so the symlink survives moving the
        # outputs/ tree (e.g. to a colleague's box).
        try:
            os.symlink(self.run_id, link, target_is_directory=True)
        except OSError as e:
            # Filesystem without symlink support (rare): leave a
            # plain text pointer instead.
            (self.stage_dir / "latest.txt").write_text(
                f"{self.run_id}\n# symlink failed: {e}\n",
            )

    @classmethod
    def resolve_latest(
        cls, root: Path, scene: str, stage: str,
    ) -> Path | None:
        """Locate the most recent run of ``<root>/<scene>/<stage>/``.

        Returns the resolved run dir, or ``None`` if nothing has been
        written yet. Used by downstream stages to discover their input.
        """
        stage_dir = Path(root) / scene / stage
        link = stage_dir / "latest"
        if link.is_symlink():
            target = link.readlink()
            run_dir = stage_dir / target if not target.is_absolute() else target
            if run_dir.is_dir():
                return run_dir
        # Fallback: pick the lexicographically-latest subdir (works
        # because run ids are ISO-style timestamps).
        if stage_dir.is_dir():
            subs = sorted(
                (p for p in stage_dir.iterdir() if p.is_dir()),
                reverse=True,
            )
            if subs:
                return subs[0]
        return None
