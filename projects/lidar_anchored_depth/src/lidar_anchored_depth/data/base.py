"""Unified per-frame schema and abstract dataset.

All algorithms in this project consume a single :class:`Frame`. To add a new
roadside / V2X dataset, implement a :class:`BaseDataset` subclass that yields
:class:`Frame` instances. See ``docs/data.md`` for the full conventions
contract — in particular: world is right-handed Z-up, the camera frame is
OpenCV (X right, Y down, Z forward), ``T_wc`` is camera→world, and LiDAR is
in world frame meters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass(slots=True)
class DynamicObject:
    """A V2X 3D-detection bounding box used by AA-HAD.

    See ``docs/data.md`` §1.1 for the canonical schema and §3 for the
    JSON ingest format. Defaults are chosen so ground vehicles need only
    populate the geometric fields.
    """

    id: int
    label: str
    xyz: np.ndarray  # (3,) world-frame bbox CENTER, meters
    lwh: np.ndarray  # (3,) length, width, height (meters)
    yaw: float
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    roll: float = 0.0
    pitch: float = 0.0
    occlusion: int = 0
    num_points: int = 0

    @property
    def Z_min(self) -> float:
        """Bottom of the bbox in world Z, meters."""
        return float(self.xyz[2] - self.lwh[2] / 2.0)

    @property
    def Z_max(self) -> float:
        """Top of the bbox in world Z, meters."""
        return float(self.xyz[2] + self.lwh[2] / 2.0)

    def predict_xyz(self, dt_seconds: float) -> np.ndarray:
        """Linear motion-compensated centre at ``t + dt`` (assumes ``v_z = 0``)."""
        out = self.xyz.copy()
        out[0] += float(self.velocity_xy[0]) * dt_seconds
        out[1] += float(self.velocity_xy[1]) * dt_seconds
        return out


@dataclass(slots=True)
class Frame:
    """A single roadside / V2X frame.

    Attributes
    ----------
    frame_id
        Unique identifier within the originating dataset.
    image
        ``(H, W, 3)`` uint8 RGB.
    K
        ``(3, 3)`` float64 OpenCV pinhole intrinsics in pixels (assumes the
        image is undistorted upstream).
    T_wc
        ``(4, 4)`` float64 SE(3) camera-to-world transform. Convention:
        ``P_world = T_wc @ [P_camera; 1]``. World is Z-up.
    lidar_world
        ``(N, 3)`` float32 LiDAR points in the world frame, meters.
    dynamic_objects
        Optional list of :class:`DynamicObject` (V2X 3D-detection bboxes).
        When present, AA-HAD uses bbox heights as primary anchors instead
        of estimating ``[Z_min, Z_max]`` from points-in-mask.
    image_ts_ms, lidar_ts_ms
        Optional millisecond UTC timestamps. Required only when their
        difference is non-zero and ``dynamic_objects`` carry non-zero
        velocity (i.e., when AA-HAD must motion-compensate the bbox).
    gt_depth
        Optional ``(H, W)`` float32 metric depth in meters; ``0`` or ``NaN``
        marks invalid pixels. Used only for evaluation.
    sam_masks
        Optional ``(K, H, W)`` boolean instance masks. If absent, the
        pipeline computes them on demand.
    meta
        Free-form dataset-specific metadata (timestamp, scene id, ...).
    """

    frame_id: str
    image: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    lidar_world: np.ndarray
    dynamic_objects: list[DynamicObject] | None = None
    image_ts_ms: int | None = None
    lidar_ts_ms: int | None = None
    gt_depth: np.ndarray | None = None
    sam_masks: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got shape {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {self.image.dtype}")

        if self.K.shape != (3, 3):
            raise ValueError(f"K must be (3, 3), got {self.K.shape}")

        if self.T_wc.shape != (4, 4):
            raise ValueError(f"T_wc must be (4, 4), got {self.T_wc.shape}")

        if self.lidar_world.ndim != 2 or self.lidar_world.shape[1] != 3:
            raise ValueError(
                f"lidar_world must be (N, 3), got shape {self.lidar_world.shape}"
            )

        if self.gt_depth is not None and self.gt_depth.shape != self.image.shape[:2]:
            raise ValueError(
                f"gt_depth shape {self.gt_depth.shape} does not match image "
                f"{self.image.shape[:2]}"
            )

        if self.sam_masks is not None:
            if self.sam_masks.ndim != 3 or self.sam_masks.shape[1:] != self.image.shape[:2]:
                raise ValueError(
                    f"sam_masks must be (K, H, W) matching image; got "
                    f"{self.sam_masks.shape}"
                )

        if self.dynamic_objects is not None:
            for o in self.dynamic_objects:
                if o.xyz.shape != (3,):
                    raise ValueError(f"DynamicObject.xyz must be (3,); got {o.xyz.shape}")
                if o.lwh.shape != (3,):
                    raise ValueError(f"DynamicObject.lwh must be (3,); got {o.lwh.shape}")
                if o.velocity_xy.shape != (2,):
                    raise ValueError(
                        f"DynamicObject.velocity_xy must be (2,); got "
                        f"{o.velocity_xy.shape}"
                    )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def hw(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    @property
    def T_cw(self) -> np.ndarray:
        """World-to-camera transform (inverse of :attr:`T_wc`)."""
        return np.linalg.inv(self.T_wc)


class BaseDataset(ABC):
    """Abstract iterable of :class:`Frame` instances.

    Subclasses must implement :meth:`__len__` and :meth:`get_frame`. The
    :meth:`__iter__` default just walks indices in order.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_frame(self, idx: int) -> Frame: ...

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.get_frame(i)

    def __getitem__(self, idx: int) -> Frame:
        return self.get_frame(idx)
