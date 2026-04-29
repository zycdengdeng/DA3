"""Parse the THICV-R2A ``calib.json`` into typed calibration objects.

Reference: ``docs/dataset_guide.md`` §3.1.

The world frame is the **VirtualLidar** frame. The schema is a single
calibration shared across all 89 sessions of the intersection (the rig is
fixed). For each pinhole camera we want to expose:

- intrinsics ``K`` (3×3)
- distortion (5-coef plumb-bob)
- camera-to-world transform ``T_wc`` (4×4) such that
  ``P_world = T_wc @ [P_cam; 1]``
- image size

For each LiDAR we want to expose:

- LiDAR-to-world transform ``T_wL`` (4×4) such that
  ``P_world = T_wL @ [P_lidar; 1]``

Implementation note: the ``calib.json`` stores ``virtualLidarToCam``
(world→camera), so ``T_wc`` is its inverse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PINHOLE_CAMERA_IDS: tuple[str, ...] = ("0", "3", "6", "9")
"""The 4 pinhole cameras used by this project (§1.4 of dataset guide)."""

PINHOLE_FOLDER_TO_CAMID: dict[str, str] = {
    "pinhole0": "3",
    "pinhole1": "6",
    "pinhole2": "9",
    "pinhole3": "0",
}
"""Folder ``pinhole{0..3}`` ↔ camera-id ``{3, 6, 9, 0}`` mapping."""

PINHOLE_CAMID_TO_FOLDER: dict[str, str] = {
    v: k for k, v in PINHOLE_FOLDER_TO_CAMID.items()
}


@dataclass(slots=True)
class CameraCalib:
    """Per-camera calibration in world frame."""

    cam_id: str
    is_fisheye: bool
    K: np.ndarray  # (3, 3) float64
    dist: np.ndarray  # plumb-bob (5,) for pinhole, (4,) for fisheye
    T_wc: np.ndarray  # (4, 4) float64 — camera-to-world
    image_size_hw: tuple[int, int]  # (height, width) in pixels


@dataclass(slots=True)
class LidarCalib:
    """Per-LiDAR calibration in world frame."""

    lidar_id: str
    T_wL: np.ndarray  # (4, 4) float64 — LiDAR-to-world


@dataclass(slots=True)
class SceneCalibration:
    """All calibration for the THICV-R2A intersection."""

    cameras: dict[str, CameraCalib]  # keyed by cam_id, e.g. "3"
    lidars: dict[str, LidarCalib]  # keyed by lidar_id, e.g. "0"
    img_size_pinhole_hw: tuple[int, int]
    img_size_fisheye_hw: tuple[int, int]

    @property
    def pinhole_cameras(self) -> dict[str, CameraCalib]:
        """Subset of ``cameras`` containing only the pinholes."""
        return {k: v for k, v in self.cameras.items() if not v.is_fisheye}


# --------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------- #
def _rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Convert a 3-vector axis-angle rotation to a 3×3 matrix.

    This is the standard Rodrigues formula. The calibration's
    ``virtualLidarToCam.rotate`` field is in this convention.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 3×3 rotation and a 3-vector translation into a 4×4 SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    """SE(3) inverse: ``T^{-1}`` for a rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def euler_zyx_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a rotation matrix from ZYX intrinsic Euler angles.

    Convention from THICV-R2A annotations (``roll, pitch, yaw`` fields):
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. Do NOT use ``cv2.Rodrigues``
    on these triples — that treats them as axis-angle and is wrong.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------- #
def load_scene_calibration(calib_path: str | Path) -> SceneCalibration:
    """Parse ``calib.json`` into a :class:`SceneCalibration`.

    Parameters
    ----------
    calib_path
        Path to ``support_info/calib.json``.
    """
    path = Path(calib_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    img_size = raw["imgSize"]
    fish_size = (int(img_size["fish"][1]), int(img_size["fish"][0]))
    pin_size = (int(img_size["notFish"][1]), int(img_size["notFish"][0]))

    cameras: dict[str, CameraCalib] = {}
    for cam_id, entry in raw["camera"].items():
        is_fish = bool(entry.get("isFish", 0))
        K = np.asarray(entry["intri"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(entry.get("distor", []), dtype=np.float64).reshape(-1)

        rvec = np.asarray(
            entry["virtualLidarToCam"]["rotate"], dtype=np.float64
        ).reshape(3)
        tvec = np.asarray(
            entry["virtualLidarToCam"]["trans"], dtype=np.float64
        ).reshape(3)
        R_v2c = _rodrigues_to_R(rvec)
        T_v2c = _rt_to_T(R_v2c, tvec)
        T_wc = _invert_T(T_v2c)

        cameras[str(cam_id)] = CameraCalib(
            cam_id=str(cam_id),
            is_fisheye=is_fish,
            K=K,
            dist=dist,
            T_wc=T_wc,
            image_size_hw=fish_size if is_fish else pin_size,
        )

    lidars: dict[str, LidarCalib] = {}
    for lidar_id, entry in raw["lidar"].items():
        R_l2v = np.asarray(
            entry["lidarToVirtualLidar"]["rotateMatrix"], dtype=np.float64
        ).reshape(3, 3)
        t_l2v = np.asarray(
            entry["lidarToVirtualLidar"]["trans"], dtype=np.float64
        ).reshape(3)
        T_wL = _rt_to_T(R_l2v, t_l2v)
        lidars[str(lidar_id)] = LidarCalib(lidar_id=str(lidar_id), T_wL=T_wL)

    return SceneCalibration(
        cameras=cameras,
        lidars=lidars,
        img_size_pinhole_hw=pin_size,
        img_size_fisheye_hw=fish_size,
    )
