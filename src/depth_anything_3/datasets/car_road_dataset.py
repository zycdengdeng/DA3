# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data loader for Car-Road Cooperative Dataset.

This module provides utilities to load and process the car-road cooperative
perception dataset for roadside reconstruction.

Dataset Structure:
    /mnt/car_road_data_fix/
    ├── 001_car0325_road0327_t1/
    │   ├── road/
    │   │   ├── cameras/
    │   │   │   ├── pinhole0/  (cam0)
    │   │   │   ├── pinhole1/  (cam3)
    │   │   │   ├── pinhole2/  (cam6)
    │   │   │   ├── pinhole3/  (cam9)
    │   │   │   ├── fisheye0/  (cam2)
    │   │   │   ├── fisheye1/  (cam5)
    │   │   │   ├── fisheye2/  (cam8)
    │   │   │   └── fisheye3/  (cam11)
    │   │   └── lidar/
    │   │       ├── lidar0/
    │   │       ├── lidar1/
    │   │       ├── lidar2/
    │   │       ├── lidar3/
    │   │       └── merged_pcd/
    │   └── ...
    └── support_info/
        └── calib.json

Calibration Format:
    - All cameras have extrinsics in VirtualLidar coordinate system
    - Pinhole cameras: 5 distortion coefficients
    - Fisheye cameras: 4 distortion coefficients (equidistant model)
    - LiDAR to VirtualLidar transforms provided
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""

    camera_id: str
    is_fisheye: bool
    intrinsics: np.ndarray  # (3, 3)
    distortion: np.ndarray  # (4,) for fisheye, (5,) for pinhole
    rotation: np.ndarray  # (3, 3) rotation matrix (VirtualLidar to Camera)
    translation: np.ndarray  # (3,) translation vector
    image_size: Tuple[int, int]  # (width, height)

    @property
    def extrinsics_w2c(self) -> np.ndarray:
        """Get 4x4 world-to-camera (VirtualLidar to Camera) matrix."""
        ext = np.eye(4)
        ext[:3, :3] = self.rotation
        ext[:3, 3] = self.translation
        return ext

    @property
    def extrinsics_c2w(self) -> np.ndarray:
        """Get 4x4 camera-to-world matrix."""
        return np.linalg.inv(self.extrinsics_w2c)


@dataclass
class LiDARCalibration:
    """LiDAR calibration parameters."""

    lidar_id: str
    rotation: np.ndarray  # (3, 3) rotation matrix (LiDAR to VirtualLidar)
    translation: np.ndarray  # (3,) translation vector

    @property
    def transform_to_virtual(self) -> np.ndarray:
        """Get 4x4 LiDAR to VirtualLidar transform matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass
class SceneData:
    """Data for a single timestamp in a scene."""

    timestamp: str  # milliseconds as string
    images: Dict[str, np.ndarray]  # camera_id -> image
    image_paths: Dict[str, str]  # camera_id -> path
    lidar_points: Dict[str, np.ndarray]  # lidar_id -> points (N, 3)
    merged_points: Optional[np.ndarray] = None  # merged point cloud in VirtualLidar frame


class CarRoadDatasetLoader:
    """
    Loader for the Car-Road Cooperative Dataset.

    This loader handles:
    - Calibration parsing (cameras + LiDARs)
    - Image and point cloud loading
    - Coordinate system transformations
    - Timestamp synchronization
    """

    # Camera ID to folder name mapping
    PINHOLE_CAMERAS = {
        "0": "pinhole0",   # cam0
        "3": "pinhole1",   # cam3
        "6": "pinhole2",   # cam6
        "9": "pinhole3",   # cam9
    }

    FISHEYE_CAMERAS = {
        "2": "fisheye0",   # cam2
        "5": "fisheye1",   # cam5
        "8": "fisheye2",   # cam8
        "11": "fisheye3",  # cam11
    }

    # LiDAR to Camera pairing (from the example code)
    LIDAR_CAMERA_PAIRS = {
        "0": "9",  # lidar0 -> cam9
        "1": "0",  # lidar1 -> cam0
        "2": "3",  # lidar2 -> cam3
        "3": "6",  # lidar3 -> cam6
    }

    def __init__(
        self,
        data_root: str = "/mnt/car_road_data_fix",
        calib_path: Optional[str] = None,
        use_fisheye: bool = False,
    ):
        """
        Initialize the dataset loader.

        Args:
            data_root: Root directory of the dataset
            calib_path: Path to calib.json. If None, uses default location.
            use_fisheye: Whether to include fisheye cameras
        """
        self.data_root = data_root
        self.use_fisheye = use_fisheye

        # Load calibration
        if calib_path is None:
            calib_path = os.path.join(data_root, "support_info", "calib.json")

        self.calib_path = calib_path
        self._load_calibration()

    def _load_calibration(self):
        """Load and parse calibration file."""
        with open(self.calib_path, 'r') as f:
            calib_data = json.load(f)

        self.image_sizes = {
            "fisheye": tuple(calib_data["imgSize"]["fish"]),  # (1280, 1280)
            "pinhole": tuple(calib_data["imgSize"]["notFish"]),  # (1280, 720)
        }

        # Parse camera calibrations
        self.cameras: Dict[str, CameraCalibration] = {}
        for cam_id, cam_data in calib_data["camera"].items():
            is_fisheye = cam_data["isFish"] == 1

            # Skip fisheye if not requested
            if is_fisheye and not self.use_fisheye:
                continue

            # Parse intrinsics (3x3 matrix flattened row-major)
            intrinsics = np.array(cam_data["intri"]).reshape(3, 3)

            # Parse distortion
            distortion = np.array(cam_data["distor"])

            # Parse extrinsics (VirtualLidar to Camera)
            # rotate is rodrigues vector
            rvec = np.array(cam_data["virtualLidarToCam"]["rotate"])
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.array(cam_data["virtualLidarToCam"]["trans"])

            # Image size
            img_size = self.image_sizes["fisheye" if is_fisheye else "pinhole"]

            self.cameras[cam_id] = CameraCalibration(
                camera_id=cam_id,
                is_fisheye=is_fisheye,
                intrinsics=intrinsics,
                distortion=distortion,
                rotation=rotation,
                translation=translation,
                image_size=img_size,
            )

        # Parse LiDAR calibrations
        self.lidars: Dict[str, LiDARCalibration] = {}
        for lid_id, lid_data in calib_data["lidar"].items():
            # rotateMatrix is 3x3 matrix flattened row-major
            rotation = np.array(lid_data["lidarToVirtualLidar"]["rotateMatrix"]).reshape(3, 3)
            translation = np.array(lid_data["lidarToVirtualLidar"]["trans"])

            self.lidars[lid_id] = LiDARCalibration(
                lidar_id=lid_id,
                rotation=rotation,
                translation=translation,
            )

        print(f"Loaded calibration: {len(self.cameras)} cameras, {len(self.lidars)} LiDARs")
        print(f"  Pinhole cameras: {[c for c in self.cameras if not self.cameras[c].is_fisheye]}")
        print(f"  Fisheye cameras: {[c for c in self.cameras if self.cameras[c].is_fisheye]}")

    def get_scene_path(self, scene_name: str) -> str:
        """Get full path to a scene directory."""
        return os.path.join(self.data_root, scene_name)

    def list_scenes(self) -> List[str]:
        """List all available scenes."""
        scenes = []
        for name in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, name)
            if os.path.isdir(scene_path) and os.path.exists(os.path.join(scene_path, "road")):
                scenes.append(name)
        return scenes

    def get_timestamps(self, scene_name: str, camera_id: str = "0") -> List[str]:
        """
        Get available timestamps for a scene.

        Args:
            scene_name: Name of the scene folder
            camera_id: Camera ID to use for finding timestamps

        Returns:
            List of timestamps (milliseconds as strings)
        """
        scene_path = self.get_scene_path(scene_name)

        # Determine camera folder
        if camera_id in self.PINHOLE_CAMERAS:
            cam_folder = self.PINHOLE_CAMERAS[camera_id]
        elif camera_id in self.FISHEYE_CAMERAS:
            cam_folder = self.FISHEYE_CAMERAS[camera_id]
        else:
            raise ValueError(f"Unknown camera ID: {camera_id}")

        cam_path = os.path.join(scene_path, "road", "cameras", cam_folder)

        if not os.path.exists(cam_path):
            return []

        # Extract timestamps from filenames: cam{N}_{timestamp}.png
        timestamps = []
        for fname in os.listdir(cam_path):
            match = re.match(r'cam\d+_(\d+)\.png$', fname)
            if match:
                timestamps.append(match.group(1))

        return sorted(timestamps)

    def load_image(
        self,
        scene_name: str,
        camera_id: str,
        timestamp: str,
        undistort: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """
        Load an image for a specific camera and timestamp.

        Args:
            scene_name: Scene folder name
            camera_id: Camera ID (e.g., "0", "3", "6", "9")
            timestamp: Timestamp in milliseconds
            undistort: Whether to undistort the image

        Returns:
            image: BGR image array
            path: Full path to the image
        """
        scene_path = self.get_scene_path(scene_name)

        # Determine camera folder
        if camera_id in self.PINHOLE_CAMERAS:
            cam_folder = self.PINHOLE_CAMERAS[camera_id]
        elif camera_id in self.FISHEYE_CAMERAS:
            cam_folder = self.FISHEYE_CAMERAS[camera_id]
        else:
            raise ValueError(f"Unknown camera ID: {camera_id}")

        img_path = os.path.join(
            scene_path, "road", "cameras", cam_folder,
            f"cam{camera_id}_{timestamp}.png"
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(img_path)

        if undistort and camera_id in self.cameras:
            image = self.undistort_image(image, camera_id)

        return image, img_path

    def undistort_image(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """Undistort an image using camera calibration."""
        if camera_id not in self.cameras:
            return image

        cam = self.cameras[camera_id]
        K = cam.intrinsics
        dist = cam.distortion

        if cam.is_fisheye:
            # Fisheye undistortion
            h, w = image.shape[:2]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, dist, (w, h), np.eye(3), balance=0.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        else:
            # Pinhole undistortion
            undistorted = cv2.undistort(image, K, dist)

        return undistorted

    def load_lidar_points(
        self,
        scene_name: str,
        lidar_id: str,
        timestamp: str,
        transform_to_virtual: bool = True,
    ) -> np.ndarray:
        """
        Load LiDAR point cloud.

        Args:
            scene_name: Scene folder name
            lidar_id: LiDAR ID (e.g., "0", "1", "2", "3")
            timestamp: Timestamp in milliseconds
            transform_to_virtual: Whether to transform to VirtualLidar frame

        Returns:
            points: (N, 3) point cloud array
        """
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for loading point clouds")

        scene_path = self.get_scene_path(scene_name)
        pcd_path = os.path.join(
            scene_path, "road", "lidar", f"lidar{lidar_id}",
            f"{timestamp}.pcd"
        )

        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float64)

        if transform_to_virtual and lidar_id in self.lidars:
            # Transform to VirtualLidar coordinate system
            T = self.lidars[lidar_id].transform_to_virtual
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            points = (T @ points_homo.T).T[:, :3]

        return points

    def load_merged_points(
        self,
        scene_name: str,
        timestamp: str,
    ) -> np.ndarray:
        """
        Load merged (fused) point cloud.

        Args:
            scene_name: Scene folder name
            timestamp: Timestamp in milliseconds

        Returns:
            points: (N, 3) point cloud array in VirtualLidar frame
        """
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for loading point clouds")

        scene_path = self.get_scene_path(scene_name)
        pcd_path = os.path.join(
            scene_path, "road", "lidar", "merged_pcd",
            f"{timestamp}.pcd"
        )

        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Merged point cloud not found: {pcd_path}")

        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float64)

        return points

    def load_scene_data(
        self,
        scene_name: str,
        timestamp: str,
        camera_ids: Optional[List[str]] = None,
        load_merged_lidar: bool = True,
        load_individual_lidars: bool = False,
        undistort_images: bool = False,
    ) -> SceneData:
        """
        Load all data for a single timestamp.

        Args:
            scene_name: Scene folder name
            timestamp: Timestamp in milliseconds
            camera_ids: List of camera IDs to load. If None, loads all pinhole cameras.
            load_merged_lidar: Whether to load merged point cloud
            load_individual_lidars: Whether to load individual LiDAR point clouds
            undistort_images: Whether to undistort images

        Returns:
            SceneData object containing images and point clouds
        """
        if camera_ids is None:
            # Default to pinhole cameras only
            camera_ids = list(self.PINHOLE_CAMERAS.keys())

        # Filter to cameras we have calibration for
        camera_ids = [c for c in camera_ids if c in self.cameras]

        # Load images
        images = {}
        image_paths = {}
        for cam_id in camera_ids:
            try:
                img, path = self.load_image(scene_name, cam_id, timestamp, undistort_images)
                images[cam_id] = img
                image_paths[cam_id] = path
            except FileNotFoundError:
                print(f"Warning: Image not found for camera {cam_id} at {timestamp}")

        # Load LiDAR data
        lidar_points = {}
        if load_individual_lidars:
            for lid_id in self.lidars:
                try:
                    points = self.load_lidar_points(scene_name, lid_id, timestamp)
                    lidar_points[lid_id] = points
                except FileNotFoundError:
                    print(f"Warning: LiDAR {lid_id} not found at {timestamp}")

        # Load merged point cloud
        merged_points = None
        if load_merged_lidar:
            try:
                merged_points = self.load_merged_points(scene_name, timestamp)
            except FileNotFoundError:
                print(f"Warning: Merged point cloud not found at {timestamp}")

        return SceneData(
            timestamp=timestamp,
            images=images,
            image_paths=image_paths,
            lidar_points=lidar_points,
            merged_points=merged_points,
        )

    def get_camera_arrays(
        self,
        camera_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get intrinsics and extrinsics as numpy arrays for reconstruction.

        Args:
            camera_ids: List of camera IDs. If None, uses all pinhole cameras.

        Returns:
            intrinsics: (N, 3, 3) array
            extrinsics: (N, 4, 4) array (world-to-camera)
        """
        if camera_ids is None:
            camera_ids = sorted([c for c in self.cameras if not self.cameras[c].is_fisheye])

        intrinsics = []
        extrinsics = []

        for cam_id in camera_ids:
            if cam_id not in self.cameras:
                raise ValueError(f"Camera {cam_id} not found in calibration")

            cam = self.cameras[cam_id]
            intrinsics.append(cam.intrinsics)
            extrinsics.append(cam.extrinsics_w2c)

        return np.array(intrinsics), np.array(extrinsics)

    def project_points_to_camera(
        self,
        points: np.ndarray,  # (N, 3) in VirtualLidar frame
        camera_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to camera image plane.

        Args:
            points: (N, 3) points in VirtualLidar coordinate system
            camera_id: Camera ID

        Returns:
            uv: (M, 2) pixel coordinates of valid points
            depths: (M,) depth values of valid points
            valid_mask: (N,) boolean mask of valid points
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")

        cam = self.cameras[camera_id]

        # Transform to camera frame
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_cam = (cam.extrinsics_w2c @ points_homo.T).T[:, :3]

        # Filter points in front of camera
        valid = points_cam[:, 2] > 0.1

        if not np.any(valid):
            return np.zeros((0, 2)), np.zeros(0), valid

        # Project using OpenCV
        rvec, _ = cv2.Rodrigues(cam.rotation)
        tvec = cam.translation.reshape(3, 1)

        obj_points = points[valid].reshape(-1, 1, 3).astype(np.float64)

        if cam.is_fisheye:
            uv, _ = cv2.fisheye.projectPoints(
                obj_points, rvec, tvec,
                cam.intrinsics.astype(np.float64),
                cam.distortion.astype(np.float64)
            )
        else:
            uv, _ = cv2.projectPoints(
                obj_points, rvec, tvec,
                cam.intrinsics.astype(np.float64),
                cam.distortion.astype(np.float64)
            )

        uv = uv.reshape(-1, 2)
        depths = points_cam[valid, 2]

        # Filter points outside image bounds
        w, h = cam.image_size
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

        # Update valid mask
        valid_indices = np.where(valid)[0]
        final_valid = np.zeros(len(points), dtype=bool)
        final_valid[valid_indices[in_bounds]] = True

        return uv[in_bounds], depths[in_bounds], final_valid


def get_lidar_for_camera(camera_id: str) -> Optional[str]:
    """Get the paired LiDAR ID for a camera."""
    # Reverse lookup from LIDAR_CAMERA_PAIRS
    for lid_id, cam_id in CarRoadDatasetLoader.LIDAR_CAMERA_PAIRS.items():
        if cam_id == camera_id:
            return lid_id
    return None
