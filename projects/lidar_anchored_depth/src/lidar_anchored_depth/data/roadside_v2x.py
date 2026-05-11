"""THICV-R2A roadside dataset adapter.

Reference: ``docs/dataset_guide.md`` (canonical format definition).

Scope (this project)
--------------------
- 4 pinhole cameras only: cam0, cam3, cam6, cam9 (§1.4)
- merged LiDAR per timestamp from ``road_labels/merged_pcd_all/`` (§1.2)
- 3D bbox annotations from ``road_labels/interpolation_labels/`` (§2.1)
- car-side data is intentionally ignored

Stage 1.2 ships the public API + dataclasses; the actual file I/O
(reading PCDs, decoding PNGs, applying static-fixture deduplication
across 89 sessions) is implemented at Stage 3 once the data is mounted
to the Linux box this loader runs on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from lidar_anchored_depth.data.base import BaseDataset, DynamicObject, Frame
from lidar_anchored_depth.data.calibration import (
    PINHOLE_CAMERA_IDS,
    PINHOLE_FOLDER_TO_CAMID,
    SceneCalibration,
    euler_zyx_to_R,
    load_scene_calibration,
)
from lidar_anchored_depth.data.carid_lookup import CaridLookup, load_carid_lookup


STATIC_FIXTURE_CLASSES: frozenset[str] = frozenset({
    "Bollards",
    "Crash_bucket",
    "Cone",
})
"""V2X annotation classes treated as scene-static (§2.2 of dataset guide).

For these classes the loader anchors the bbox once per ``(scene_id,
instance_id)`` and reuses that anchor for all subsequent frames in the
scene; the per-frame entry is omitted from ``Frame.dynamic_objects``.
"""

PCD_TIMESTAMP_TOLERANCE_MS: int = 500
"""Maximum allowed gap between an annotation timestamp and a PCD timestamp."""


@dataclass(slots=True)
class SceneIndex:
    """Index of a single scene directory (one of 89)."""

    scene_id: str  # e.g. "008_car0325_road0327_t8"
    root: Path  # absolute path to the scene directory
    label_jsons: list[Path] = field(default_factory=list)  # sorted by ts ms
    timestamps_ms: list[int] = field(default_factory=list)
    ego_id: int | None = None  # from carid.json if available

    def __len__(self) -> int:
        return len(self.label_jsons)


@dataclass(slots=True)
class StaticFixture:
    """A scene-static annotated object (Bollards / Crash_bucket / Cone)."""

    scene_id: str
    instance_id: int
    label: str
    obj: DynamicObject  # with vx=vy=0; xyz/lwh/yaw frozen at first sighting


class RoadsideV2XLoader(BaseDataset):
    """Iterate per-(scene, timestamp, camera) frames over THICV-R2A.

    Parameters
    ----------
    data_root
        Path to ``/mnt/car_road_data_TianJin/`` (or equivalent mount).
    scenes
        Optional whitelist of scene-id prefixes (e.g. ``["008", "010"]``).
        ``None`` means all 89 scenes.
    cameras
        Optional whitelist of pinhole camera-ids; default uses all 4
        (``"0", "3", "6", "9"``). Fisheye cameras are never emitted.
    occlusion_max
        Drop dynamic objects with ``occlusion >= occlusion_max``.
    min_num_points
        Drop dynamic objects with ``num_points`` below this threshold.
    static_fixture_classes
        Override the default scene-static class set. Pass an empty
        frozenset to disable the optimization (treat fixtures as
        per-frame dynamic).
    pcd_tolerance_ms
        Max ms gap between annotation ts and PCD ts when matching
        ``merged_pcd`` files (only relevant if ``merged_pcd_all`` is
        absent and we fall back to ``merged_pcd``).
    """

    def __init__(
        self,
        data_root: str | Path,
        scenes: list[str] | None = None,
        cameras: list[str] | None = None,
        *,
        occlusion_max: int = 2,
        min_num_points: int = 5,
        static_fixture_classes: frozenset[str] = STATIC_FIXTURE_CLASSES,
        pcd_tolerance_ms: int = PCD_TIMESTAMP_TOLERANCE_MS,
        labels_source: str = "interpolation",
    ) -> None:
        """
        Parameters
        ----------
        labels_source : "interpolation" or "merged_pcd"
            Which V2X bbox folder to read.

            * ``"interpolation"`` (default): 10 Hz interpolated bboxes
              from ``road_labels/interpolation_labels/``. Smooth for
              video, but the interpolation jitter can leak per-object
              LiDAR points into the static cloud.
            * ``"merged_pcd"``: 1 Hz hand-labeled bboxes from
              ``road_labels/merged_pcd_all/``. Much more accurate
              (no interpolation jitter); used by the static-cloud
              accumulator to avoid leaving a trail of car points
              along the trajectory. Sparser timestamp list — not
              suitable for video.
        """
        self.data_root = Path(data_root)
        self.scene_filter = list(scenes) if scenes else None

        if cameras is None:
            cameras = list(PINHOLE_CAMERA_IDS)
        for c in cameras:
            if c not in PINHOLE_CAMERA_IDS:
                raise ValueError(
                    f"camera id {c!r} is not a pinhole; allowed: {PINHOLE_CAMERA_IDS}"
                )
        self.cameras = cameras

        if labels_source not in ("interpolation", "merged_pcd"):
            raise ValueError(
                f"labels_source must be 'interpolation' or 'merged_pcd', "
                f"got {labels_source!r}"
            )
        self.labels_source = labels_source
        self.occlusion_max = occlusion_max
        self.min_num_points = min_num_points
        self.static_fixture_classes = static_fixture_classes
        self.pcd_tolerance_ms = pcd_tolerance_ms

        self._calib: SceneCalibration | None = None
        self._carid: CaridLookup | None = None
        self._scenes: list[SceneIndex] = []
        self._scene_fixtures: dict[str, dict[int, StaticFixture]] = {}
        self._index_built: bool = False

    # ------------------------------------------------------------------ #
    # Lazy initialization
    # ------------------------------------------------------------------ #
    @property
    def calibration(self) -> SceneCalibration:
        """The scene calibration (loaded once, shared across all 89 sessions)."""
        if self._calib is None:
            self._calib = load_scene_calibration(
                self.data_root / "support_info" / "calib.json"
            )
        return self._calib

    @property
    def carid_lookup(self) -> CaridLookup:
        """The per-scene ego-id mapping."""
        if self._carid is None:
            self._carid = load_carid_lookup(
                self.data_root / "support_info" / "carid.json"
            )
        return self._carid

    @property
    def scene_fixtures(self) -> dict[str, dict[int, StaticFixture]]:
        """Static fixtures per scene, keyed by ``scene_id`` then ``instance_id``.

        Populated lazily during scene indexing. Each fixture's bbox is
        captured at its first sighting in the scene and reused for all
        subsequent frames of that scene.
        """
        if not self._index_built:
            self._build_index()
        return self._scene_fixtures

    def _build_index(self) -> None:
        """Walk ``data_root`` to enumerate scenes and per-frame timestamps.

        Cheap pass: only directory listing and ``int(stem)`` parsing of
        annotation JSON filenames. Static-fixture deduplication is
        deferred to ``get_frame`` (lazy) to keep this O(scenes + frames)
        without reading any JSON contents.
        """
        if self._index_built:
            return

        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"data_root does not exist: {self.data_root}"
            )

        scene_dirs = []
        for d in sorted(self.data_root.iterdir()):
            if d.name == "support_info":
                continue
            try:
                if d.is_dir():
                    scene_dirs.append(d)
            except PermissionError:
                continue
        if self.scene_filter is not None:
            allowed = set(self.scene_filter)
            scene_dirs = [
                d
                for d in scene_dirs
                if d.name in allowed
                or any(d.name.startswith(prefix) for prefix in allowed)
            ]

        self._scenes = []
        self._scene_fixtures = {}

        labels_subdir = (
            "interpolation_labels"
            if self.labels_source == "interpolation"
            else "merged_pcd_all"
        )

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            label_dir = scene_dir / "road_labels" / labels_subdir
            try:
                if not label_dir.is_dir():
                    continue
                json_files = sorted(label_dir.glob("*.json"))
            except PermissionError:
                continue
            if not json_files:
                continue

            timestamps: list[int] = []
            kept_jsons: list[Path] = []
            for jp in json_files:
                try:
                    ts = int(jp.stem)
                except ValueError:
                    continue
                timestamps.append(ts)
                kept_jsons.append(jp)

            ego_id: int | None = None
            try:
                ego_id = self.carid_lookup.get_ego_id(scene_id)
            except FileNotFoundError:
                ego_id = None

            self._scenes.append(
                SceneIndex(
                    scene_id=scene_id,
                    root=scene_dir,
                    label_jsons=kept_jsons,
                    timestamps_ms=timestamps,
                    ego_id=ego_id,
                )
            )
            self._scene_fixtures[scene_id] = {}

        self._index_built = True

    def _resolve_idx(self, idx: int) -> tuple[SceneIndex, int, str]:
        """Map a flat ``idx`` to ``(scene, ts_index, camera_id)``.

        Layout: outer loop over scenes, then over timestamps, inner loop
        over the configured cameras.
        """
        if idx < 0:
            idx += len(self)
        n_cams = len(self.cameras)
        running = 0
        for scene in self._scenes:
            n_ts = len(scene)
            block = n_ts * n_cams
            if idx < running + block:
                local = idx - running
                ts_i = local // n_cams
                cam_i = local % n_cams
                return scene, ts_i, self.cameras[cam_i]
            running += block
        raise IndexError(idx)

    @property
    def scenes(self) -> list[SceneIndex]:
        """Lazily-built list of indexed scenes."""
        if not self._index_built:
            self._build_index()
        return self._scenes

    def find_frame_idx(
        self, scene_id: str, ts_ms: int, cam_id: str
    ) -> int:
        """Map ``(scene_id, ts_ms, cam_id)`` to the flat index used by
        :meth:`get_frame`. Raises ``ValueError`` when no match is found.

        Accepts either a full scene name or a numeric prefix like
        ``"008"`` for ``scene_id``.
        """
        if not self._index_built:
            self._build_index()
        if cam_id not in self.cameras:
            raise ValueError(
                f"camera {cam_id!r} not in loader.cameras = {self.cameras}"
            )
        cam_i = self.cameras.index(cam_id)
        n_cams = len(self.cameras)

        running = 0
        for s in self._scenes:
            match = (
                s.scene_id == scene_id
                or s.scene_id.startswith(scene_id + "_")
            )
            if not match:
                running += len(s) * n_cams
                continue
            try:
                ts_i = s.timestamps_ms.index(int(ts_ms))
            except ValueError as exc:
                first = s.timestamps_ms[:3]
                raise ValueError(
                    f"timestamp {ts_ms} not in scene {s.scene_id} "
                    f"(first 3: {first})"
                ) from exc
            return running + ts_i * n_cams + cam_i
        raise ValueError(f"scene matching {scene_id!r} not found")

    # ------------------------------------------------------------------ #
    # BaseDataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if not self._index_built:
            self._build_index()
        n_ts = sum(len(s) for s in self._scenes)
        return n_ts * len(self.cameras)

    def get_frame(self, idx: int) -> Frame:
        """Load one frame ``(scene, timestamp, camera)``.

        Reads the annotation JSON, the corresponding pinhole image, and
        the merged LiDAR PCD; returns a populated :class:`Frame`. The
        first time a scene is touched we also scan its annotations for
        static fixtures (Bollards / Crash_bucket / Cone) and cache them
        in :attr:`scene_fixtures`.
        """
        if not self._index_built:
            self._build_index()
        scene, ts_i, cam_id = self._resolve_idx(idx)
        ts_ms = scene.timestamps_ms[ts_i]
        json_path = scene.label_jsons[ts_i]

        # Lazy static-fixture scan for the scene
        if scene.scene_id not in self._scene_fixtures or not self._scene_fixtures[
            scene.scene_id
        ]:
            self._scan_scene_fixtures(scene)

        # Load image
        img_path = self.pinhole_image_path(scene.root, cam_id, ts_ms)
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenCV (cv2) is required for image I/O; install with "
                "`pip install opencv-python` or `opencv-python-headless`"
            ) from exc
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Load LiDAR
        from lidar_anchored_depth.data.pcd_io import read_pcd_xyz

        pcd_path = self.merged_pcd_path(scene.root, ts_ms)
        if not pcd_path.is_file():
            pcd_path = self.merged_pcd_path(
                scene.root, ts_ms, prefer_aligned=False
            )
        lidar_world = read_pcd_xyz(pcd_path)

        # Read annotation
        entry = self._read_annotation_entry(json_path)
        dynamic_objects: list[DynamicObject] = []
        for raw_obj in entry.get("object", []):
            obj, is_static = self._parse_object(raw_obj, scene.scene_id)
            if is_static:
                continue  # absorbed into scene_fixtures
            if not self._filter_dynamic(obj):
                continue
            dynamic_objects.append(obj)

        # Calibration for this camera
        cam_calib = self.calibration.cameras[cam_id]

        return Frame(
            frame_id=f"{scene.scene_id}/{ts_ms}/cam{cam_id}",
            image=rgb,
            K=cam_calib.K,
            T_wc=cam_calib.T_wc,
            lidar_world=lidar_world,
            dynamic_objects=dynamic_objects,
            image_ts_ms=ts_ms,
            lidar_ts_ms=ts_ms,
            meta={
                "scene_id": scene.scene_id,
                "cam_id": cam_id,
                "is_fisheye": cam_calib.is_fisheye,
                "image_size_hw": cam_calib.image_size_hw,
                "distortion": cam_calib.dist,
                "ego_id": scene.ego_id,
                "image_path": str(img_path),
                "pcd_path": str(pcd_path),
                "json_path": str(json_path),
            },
        )

    def _scan_scene_fixtures(self, scene: SceneIndex) -> None:
        """Populate ``scene_fixtures[scene_id]`` by scanning every JSON.

        Static fixtures are recorded at their first sighting (lowest
        timestamp). Subsequent occurrences with the same instance id are
        ignored — they're assumed identical (rig is fixed across the 89
        sessions).
        """
        bucket = self._scene_fixtures.setdefault(scene.scene_id, {})
        for json_path in scene.label_jsons:
            entry = self._read_annotation_entry(json_path)
            for raw_obj in entry.get("object", []):
                obj, is_static = self._parse_object(raw_obj, scene.scene_id)
                if not is_static:
                    continue
                key = int(obj.id)
                if key in bucket:
                    continue
                bucket[key] = StaticFixture(
                    scene_id=scene.scene_id,
                    instance_id=key,
                    label=obj.label,
                    obj=obj,
                )

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.get_frame(i)

    # ------------------------------------------------------------------ #
    # Per-frame helpers
    # ------------------------------------------------------------------ #
    def _parse_object(
        self, raw_obj: dict, scene_id: str
    ) -> tuple[DynamicObject, bool]:
        """Map one annotation entry to :class:`DynamicObject`.

        Returns
        -------
        (obj, is_static_fixture)
            ``obj`` is a populated :class:`DynamicObject`. The boolean
            indicates whether this annotation belongs to the static
            fixture set (caller decides whether to emit it per-frame).

        Notes
        -----
        - ``z`` is the bbox CENTER (THICV-R2A convention; §2.1).
        - ``roll/pitch/yaw`` are ZYX Euler angles in radians; we expose
          the ``yaw`` field as-is on :class:`DynamicObject` (consumers
          that need the full rotation should call
          :func:`euler_zyx_to_R(roll, pitch, yaw)` from ``calibration``).
        """
        label = str(raw_obj["label"])
        is_static = label in self.static_fixture_classes

        xyz = np.array(
            [float(raw_obj["x"]), float(raw_obj["y"]), float(raw_obj["z"])],
            dtype=np.float64,
        )
        lwh = np.array(
            [
                float(raw_obj["length"]),
                float(raw_obj["width"]),
                float(raw_obj["height"]),
            ],
            dtype=np.float64,
        )
        velocity_xy = np.array(
            [float(raw_obj.get("vx", 0.0)), float(raw_obj.get("vy", 0.0))],
            dtype=np.float64,
        )

        obj = DynamicObject(
            id=int(raw_obj["id"]),
            label=label,
            xyz=xyz,
            lwh=lwh,
            yaw=float(raw_obj["yaw"]),
            roll=float(raw_obj.get("roll", 0.0)),
            pitch=float(raw_obj.get("pitch", 0.0)),
            velocity_xy=velocity_xy,
            occlusion=int(raw_obj.get("occlusion", 0)),
            num_points=int(raw_obj.get("num_points", 0)),
        )
        return obj, is_static

    def _filter_dynamic(self, obj: DynamicObject) -> bool:
        """Return True if this object should appear in ``Frame.dynamic_objects``."""
        if obj.occlusion >= self.occlusion_max:
            return False
        if obj.num_points < self.min_num_points:
            return False
        return True

    def _global_key(self, scene_id: str, instance_id: int) -> tuple[str, int]:
        """The composite key needed for cross-scene unique identity.

        ``id`` resets per scene; pair with ``scene_id`` for global
        uniqueness (§5 of dataset guide).
        """
        return (scene_id, instance_id)

    @staticmethod
    def pinhole_image_path(
        scene_root: Path, cam_id: str, ts_ms: int | str
    ) -> Path:
        """Resolve the pinhole image path for ``(scene, cam, timestamp)``.

        Honours the ``pinhole{N}/`` folder ↔ ``cam{3,6,9,0}_*.png`` file
        prefix mismatch (§1.4 of dataset guide).
        """
        if cam_id not in PINHOLE_CAMERA_IDS:
            raise ValueError(f"cam_id {cam_id!r} is not a pinhole")
        # invert PINHOLE_FOLDER_TO_CAMID
        folder = next(
            f for f, c in PINHOLE_FOLDER_TO_CAMID.items() if c == cam_id
        )
        return (
            scene_root
            / "road"
            / "cameras"
            / folder
            / f"cam{cam_id}_{ts_ms}.png"
        )

    @staticmethod
    def merged_pcd_path(
        scene_root: Path, ts_ms: int | str, *, prefer_aligned: bool = True
    ) -> Path:
        """Resolve the merged-LiDAR PCD path for a given timestamp.

        When ``prefer_aligned`` is True (recommended), use
        ``road_labels/merged_pcd_all/<ts>.pcd`` which is timestamp-aligned
        to the annotation file. Falls back to ``road/lidar/merged_pcd/``
        when the aligned variant is missing.
        """
        primary = (
            scene_root / "road_labels" / "merged_pcd_all" / f"{ts_ms}.pcd"
        )
        if prefer_aligned:
            return primary
        return scene_root / "road" / "lidar" / "merged_pcd" / f"{ts_ms}.pcd"

    @staticmethod
    def annotation_json_path(scene_root: Path, ts_ms: int | str) -> Path:
        """Resolve the annotation JSON path for a given timestamp."""
        return (
            scene_root
            / "road_labels"
            / "interpolation_labels"
            / f"{ts_ms}.json"
        )

    def _read_annotation_entry(self, json_path: Path) -> dict:
        """Read a single annotation JSON file."""
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
