"""Dataset adapters and the unified Frame schema."""

from lidar_anchored_depth.data.base import BaseDataset, DynamicObject, Frame
from lidar_anchored_depth.data.calibration import (
    PINHOLE_CAMERA_IDS,
    PINHOLE_FOLDER_TO_CAMID,
    CameraCalib,
    LidarCalib,
    SceneCalibration,
    euler_zyx_to_R,
    load_scene_calibration,
)
from lidar_anchored_depth.data.carid_lookup import (
    CaridLookup,
    EgoEntry,
    load_carid_lookup,
)
from lidar_anchored_depth.data.roadside_v2x import (
    PCD_TIMESTAMP_TOLERANCE_MS,
    STATIC_FIXTURE_CLASSES,
    RoadsideV2XLoader,
    SceneIndex,
    StaticFixture,
)

__all__ = [
    "BaseDataset",
    "CameraCalib",
    "CaridLookup",
    "DynamicObject",
    "EgoEntry",
    "Frame",
    "LidarCalib",
    "PCD_TIMESTAMP_TOLERANCE_MS",
    "PINHOLE_CAMERA_IDS",
    "PINHOLE_FOLDER_TO_CAMID",
    "RoadsideV2XLoader",
    "SceneCalibration",
    "SceneIndex",
    "STATIC_FIXTURE_CLASSES",
    "StaticFixture",
    "euler_zyx_to_R",
    "load_carid_lookup",
    "load_scene_calibration",
]
