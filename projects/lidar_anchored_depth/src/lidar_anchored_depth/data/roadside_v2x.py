"""Adapter for the canonical roadside-V2X annotation format.

This loader consumes per-timestamp JSON entries with the schema documented
in ``docs/data.md`` §3 (multiple cameras per timestamp, one merged LiDAR
point cloud, a list of 3D bounding boxes with velocity). It is the
primary data path for AA-HAD.

Stage 1.1 ships only the public API (the dataclasses + the loader's
constructor and signatures); Stage 3 implements the actual parsing,
calibration loading, point-cloud / image I/O, and frame iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from lidar_anchored_depth.data.base import BaseDataset, DynamicObject, Frame


@dataclass(slots=True)
class CameraCalib:
    """Per-camera calibration loaded once per scene."""

    cam_id: str
    K: np.ndarray  # (3, 3) float64
    T_wc: np.ndarray  # (4, 4) float64, camera-to-world
    image_size_hw: tuple[int, int]
    distortion: np.ndarray | None = None  # (k,) if rectification needed


@dataclass(slots=True)
class SceneCalib:
    """Calibration bundle for a single scene (multiple cameras + LiDAR)."""

    cameras: dict[str, CameraCalib]
    lidar_T_world: np.ndarray | None = None  # if LiDAR is not already in world frame


class RoadsideV2XLoader(BaseDataset):
    """Iterate frames produced by a roadside V2X rig with 3D bbox annotations.

    Parameters
    ----------
    root
        Filesystem root containing the per-timestamp JSON entries (see
        ``docs/data.md`` §3) and the referenced ``image_file`` / ``pcd_file``
        relative paths.
    manifest
        Either the path to a single-line-JSON / JSONL manifest enumerating
        per-timestamp entries, or a directory to glob for ``*.json``.
    calib_path
        Path to the scene calibration JSON. The exact format is
        deployment-specific; the adapter reads it once and produces a
        :class:`SceneCalib`.
    cameras
        Optional whitelist of camera IDs (e.g. ``["cam0", "cam3"]``) — when
        omitted, all cameras present in the manifest are emitted as
        separate frames sharing the same LiDAR.
    occlusion_max
        Drop annotated objects with ``occlusion >= occlusion_max`` from
        :class:`DynamicObject` lists. Default ``2`` matches the canonical
        schema ("heavily occluded").
    min_num_points
        Drop objects with ``num_points`` below this threshold. Default
        ``5``; raise for stricter quality.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        calib_path: str | Path,
        cameras: list[str] | None = None,
        *,
        occlusion_max: int = 2,
        min_num_points: int = 5,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest)
        self.calib_path = Path(calib_path)
        self.cameras = cameras
        self.occlusion_max = occlusion_max
        self.min_num_points = min_num_points

        # Stage 3: actually parse manifest + calibration here.
        self._entries: list[dict] = []
        self._calib: SceneCalib | None = None

    # ------------------------------------------------------------------ #
    # BaseDataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._entries) * (
            len(self.cameras) if self.cameras else 1
        )

    def get_frame(self, idx: int) -> Frame:
        raise NotImplementedError("Stage 3: V2X loader is API-only at Stage 1.1")

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.get_frame(i)

    # ------------------------------------------------------------------ #
    # Helpers (signatures pinned now, implemented Stage 3)
    # ------------------------------------------------------------------ #
    def _parse_manifest(self) -> list[dict]:
        """Read the manifest into a list of per-timestamp JSON entries."""
        raise NotImplementedError("Stage 3")

    def _load_scene_calib(self) -> SceneCalib:
        """Read the per-scene calibration JSON into a :class:`SceneCalib`."""
        raise NotImplementedError("Stage 3")

    def _entry_to_objects(self, entry: dict) -> list[DynamicObject]:
        """Map a manifest entry's ``object`` list into :class:`DynamicObject`.

        Applies ``occlusion_max`` and ``min_num_points`` filtering. Assumes
        ``z`` is the bbox CENTER (per ``docs/data.md`` §3); bottom-z
        deployments must override this method.
        """
        raise NotImplementedError("Stage 3")
