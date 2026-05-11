"""Resolve a stage's input artefacts from the upstream stage's
``latest`` symlink.

Each downstream stage knows which upstream stage it consumes (e.g.
``complete`` reads from ``depth`` / ``mask`` / ``seg`` / ``calib``).
When a config field is left at ``None``, the stage calls one of the
helpers in this module to fill it in. Explicit user-set paths are
left untouched.
"""

from __future__ import annotations

from pathlib import Path

from lidar_anchored_depth.engine.runner import OutputManager


def resolve_upstream_dir(
    output_root: Path,
    scene: str,
    upstream_stage: str,
    *,
    required: bool = True,
    flag_hint: str = "",
) -> Path | None:
    """Return ``<output_root>/<scene>/<upstream_stage>/latest/``.

    Parameters
    ----------
    required :
        When True (default), raise ``SystemExit`` with a migration hint
        if the upstream stage has never run. When False, return ``None``.
    flag_hint :
        CLI flag name to mention in the error message so the user knows
        what they could pass instead (e.g. ``"--sam-mask-dir"``).
    """
    latest = OutputManager.resolve_latest(output_root, scene, upstream_stage)
    if latest is None:
        if not required:
            return None
        hint = f", or pass --{flag_hint} explicitly" if flag_hint else ""
        raise SystemExit(
            f"no upstream `{upstream_stage}` run found under "
            f"{output_root}/{scene}/{upstream_stage}/. "
            f"Run `lad {upstream_stage} --scene.scene {scene}` first{hint}."
        )
    return latest


def resolve_upstream_file(
    output_root: Path,
    scene: str,
    upstream_stage: str,
    pattern: str,
    *,
    required: bool = True,
    flag_hint: str = "",
) -> Path | None:
    """Return the first file under
    ``<output_root>/<scene>/<upstream_stage>/latest/`` whose name
    matches ``pattern`` (glob). Used for one-shot artefacts like the
    calib JSON or the static hybrid PLY."""
    latest = resolve_upstream_dir(
        output_root, scene, upstream_stage,
        required=required, flag_hint=flag_hint,
    )
    if latest is None:
        return None
    matches = sorted(latest.glob(pattern))
    if not matches:
        if not required:
            return None
        hint = f", or pass --{flag_hint} explicitly" if flag_hint else ""
        raise SystemExit(
            f"upstream {latest} has no files matching {pattern!r}. "
            f"Re-run `lad {upstream_stage}`{hint}."
        )
    return matches[0]


def resolve_upstream_glob(
    output_root: Path,
    scene: str,
    upstream_stage: str,
    pattern: str,
    *,
    required: bool = True,
    flag_hint: str = "",
) -> str | None:
    """Return the **glob string** ``<latest_dir>/<pattern>``, suitable
    for handing to a script's ``--<flag>-glob`` argument or to
    ``glob.glob``. Asserts that at least one file matches the glob (so
    we don't silently feed a wrong path).

    The script that consumes this expands the glob itself; this helper
    just provides the directory + pattern string.
    """
    latest = resolve_upstream_dir(
        output_root, scene, upstream_stage,
        required=required, flag_hint=flag_hint,
    )
    if latest is None:
        return None
    # Sanity: confirm at least one match. This protects against a
    # successful upstream run that didn't write the expected files.
    if not list(latest.glob(pattern)):
        if not required:
            return None
        hint = f", or pass --{flag_hint} explicitly" if flag_hint else ""
        raise SystemExit(
            f"upstream {latest} has no files matching {pattern!r}. "
            f"Re-run `lad {upstream_stage}`{hint}."
        )
    return str(latest / pattern)


__all__ = [
    "resolve_upstream_dir",
    "resolve_upstream_file",
    "resolve_upstream_glob",
]
