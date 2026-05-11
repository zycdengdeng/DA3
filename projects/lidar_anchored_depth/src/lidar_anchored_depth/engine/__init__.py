"""Engine: stage execution support (output paths, run ids, logging,
upstream-artefact discovery)."""

from lidar_anchored_depth.engine.discovery import (
    resolve_upstream_dir,
    resolve_upstream_file,
    resolve_upstream_glob,
)
from lidar_anchored_depth.engine.runner import OutputManager

__all__ = [
    "OutputManager",
    "resolve_upstream_dir",
    "resolve_upstream_file",
    "resolve_upstream_glob",
]
