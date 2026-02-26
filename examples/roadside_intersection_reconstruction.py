#!/usr/bin/env python3
"""
Example: Roadside Intersection Reconstruction with DA3 + LiDAR Alignment

This example demonstrates how to reconstruct a traffic intersection scene
using DA3 depth estimation with LiDAR-guided metric scale alignment.

Scene Setup:
- 4 cameras at intersection corners (elevated positions, ~6-10m height)
- 4 LiDARs co-located with cameras
- Scene size: ~200m x 200m
- Each camera-LiDAR pair is calibrated

Usage:
    python examples/roadside_intersection_reconstruction.py \
        --image_dir /path/to/images \
        --calib_dir /path/to/calibration \
        --lidar_dir /path/to/lidar \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.roadside_reconstruction import (
    RoadsideReconstructor,
    reconstruct_intersection,
)


def load_calibration(calib_dir: str, num_cameras: int = 4):
    """
    Load camera calibration from files.

    Expected files:
        calib_dir/
            cam_0_intrinsic.txt   # fx fy cx cy (or 3x3 matrix)
            cam_0_extrinsic.txt   # 4x4 world-to-camera matrix
            cam_1_intrinsic.txt
            cam_1_extrinsic.txt
            ...

    Or single files:
        calib_dir/
            intrinsics.npy   # (N, 3, 3)
            extrinsics.npy   # (N, 4, 4)
    """
    calib_dir = Path(calib_dir)

    # Try loading single files first
    intrinsics_file = calib_dir / "intrinsics.npy"
    extrinsics_file = calib_dir / "extrinsics.npy"

    if intrinsics_file.exists() and extrinsics_file.exists():
        intrinsics = np.load(intrinsics_file)
        extrinsics = np.load(extrinsics_file)
        return intrinsics, extrinsics

    # Load per-camera calibration files
    intrinsics = []
    extrinsics = []

    for i in range(num_cameras):
        # Try different file formats
        int_file = calib_dir / f"cam_{i}_intrinsic.txt"
        ext_file = calib_dir / f"cam_{i}_extrinsic.txt"

        if int_file.exists():
            int_data = np.loadtxt(int_file)
            if int_data.shape == (4,):  # fx fy cx cy format
                fx, fy, cx, cy = int_data
                int_mat = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            else:
                int_mat = int_data.reshape(3, 3)
            intrinsics.append(int_mat)

        if ext_file.exists():
            ext_data = np.loadtxt(ext_file)
            ext_mat = ext_data.reshape(4, 4)
            extrinsics.append(ext_mat)

    return np.array(intrinsics), np.array(extrinsics)


def load_lidar_points(lidar_dir: str, num_lidars: int = 4):
    """
    Load LiDAR point clouds from files.

    Supported formats:
        - .bin (KITTI format): (N, 4) float32 [x, y, z, intensity]
        - .npy: (N, 3) or (N, 4)
        - .pcd: Point Cloud Data format
        - .txt: Space-separated x y z [intensity]
    """
    lidar_dir = Path(lidar_dir)
    lidar_points = []

    for i in range(num_lidars):
        # Try different file formats
        for ext in ['.bin', '.npy', '.pcd', '.txt']:
            lidar_file = lidar_dir / f"lidar_{i}{ext}"
            if not lidar_file.exists():
                lidar_file = lidar_dir / f"velodyne_{i}{ext}"
            if not lidar_file.exists():
                continue

            if ext == '.bin':
                points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
                points = points[:, :3]  # Remove intensity
            elif ext == '.npy':
                points = np.load(lidar_file)
                if points.shape[1] == 4:
                    points = points[:, :3]
            elif ext == '.txt':
                points = np.loadtxt(lidar_file)
                if points.shape[1] == 4:
                    points = points[:, :3]
            else:
                raise ValueError(f"Unsupported format: {ext}")

            lidar_points.append(points)
            print(f"Loaded LiDAR {i}: {len(points):,} points")
            break
        else:
            print(f"Warning: No LiDAR file found for sensor {i}")
            lidar_points.append(None)

    return lidar_points


def load_images(image_dir: str, num_cameras: int = 4):
    """Load images for each camera."""
    image_dir = Path(image_dir)
    images = []

    for i in range(num_cameras):
        # Try different naming conventions
        for pattern in [f"cam_{i}", f"camera_{i}", f"image_{i}", str(i)]:
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = image_dir / f"{pattern}{ext}"
                if img_file.exists():
                    images.append(str(img_file))
                    print(f"Loaded image: {img_file}")
                    break
            else:
                continue
            break
        else:
            raise FileNotFoundError(f"No image found for camera {i}")

    return images


def create_example_calibration(output_dir: str, num_cameras: int = 4):
    """
    Create example calibration files for a typical intersection setup.

    This creates a sample configuration with:
    - 4 cameras at corners of a 200m x 200m intersection
    - Cameras mounted at 8m height, looking toward center
    - 60-degree field of view
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intersection parameters
    intersection_size = 200.0  # meters
    camera_height = 8.0  # meters

    # Camera positions at corners (in world coordinates)
    corner_positions = np.array([
        [-intersection_size/2, -intersection_size/2, camera_height],  # SW corner
        [intersection_size/2, -intersection_size/2, camera_height],   # SE corner
        [intersection_size/2, intersection_size/2, camera_height],    # NE corner
        [-intersection_size/2, intersection_size/2, camera_height],   # NW corner
    ])

    # Image size
    image_width = 1920
    image_height = 1080

    # Camera intrinsics (60 degree FOV)
    fov_h = np.radians(60)
    fx = image_width / (2 * np.tan(fov_h / 2))
    fy = fx  # Square pixels
    cx = image_width / 2
    cy = image_height / 2

    intrinsics = []
    extrinsics = []

    for i, pos in enumerate(corner_positions):
        # Intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics.append(K)

        # Camera looks toward center of intersection
        target = np.array([0, 0, 0])  # Center of intersection

        # Compute rotation matrix (camera looking at target)
        forward = target - pos
        forward = forward / np.linalg.norm(forward)

        # Up vector (world Z)
        up = np.array([0, 0, 1])

        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Recompute up
        up = np.cross(right, forward)

        # Rotation matrix (world to camera)
        R = np.array([right, -up, forward])  # OpenCV convention

        # Translation (world to camera)
        t = -R @ pos

        # Extrinsic matrix (4x4 world-to-camera)
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = t
        extrinsics.append(ext)

        # Save individual calibration files
        np.savetxt(output_dir / f"cam_{i}_intrinsic.txt", K)
        np.savetxt(output_dir / f"cam_{i}_extrinsic.txt", ext)

    # Save combined calibration
    np.save(output_dir / "intrinsics.npy", np.array(intrinsics))
    np.save(output_dir / "extrinsics.npy", np.array(extrinsics))

    # Save metadata
    metadata = {
        "num_cameras": num_cameras,
        "intersection_size_m": intersection_size,
        "camera_height_m": camera_height,
        "image_size": [image_width, image_height],
        "fov_degrees": np.degrees(fov_h),
        "camera_positions": corner_positions.tolist(),
    }
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Example calibration saved to {output_dir}")
    print(f"Camera positions:\n{corner_positions}")

    return np.array(intrinsics), np.array(extrinsics)


def main():
    parser = argparse.ArgumentParser(
        description="Roadside Intersection Reconstruction with DA3 + LiDAR"
    )
    parser.add_argument("--image_dir", type=str, help="Directory with camera images")
    parser.add_argument("--calib_dir", type=str, help="Directory with calibration files")
    parser.add_argument("--lidar_dir", type=str, help="Directory with LiDAR point clouds")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="da3-giant",
                        help="DA3 model name (default: da3-giant)")
    parser.add_argument("--num_cameras", type=int, default=4,
                        help="Number of cameras (default: 4)")
    parser.add_argument("--conf_threshold", type=float, default=30.0,
                        help="Confidence threshold percentile (default: 30)")
    parser.add_argument("--no_ransac", action="store_true",
                        help="Disable RANSAC for LiDAR alignment")
    parser.add_argument("--max_view_angle", type=float, default=70.0,
                        help="Max angle (degrees) from optical axis. Filters peripheral noise. (default: 70)")
    parser.add_argument("--max_depth", type=float, default=150.0,
                        help="Max depth in meters. Filters distant points. (default: 150)")
    parser.add_argument("--create_example_calib", action="store_true",
                        help="Create example calibration files and exit")

    args = parser.parse_args()

    # Create example calibration if requested
    if args.create_example_calib:
        create_example_calibration(args.output_dir, args.num_cameras)
        return

    # Validate inputs
    if not args.image_dir:
        parser.error("--image_dir is required for reconstruction")
    if not args.calib_dir:
        parser.error("--calib_dir is required for reconstruction")

    print("=" * 60)
    print("Roadside Intersection Reconstruction")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading calibration...")
    intrinsics, extrinsics = load_calibration(args.calib_dir, args.num_cameras)
    print(f"Loaded {len(intrinsics)} camera calibrations")

    print("\n[2/4] Loading images...")
    images = load_images(args.image_dir, args.num_cameras)
    print(f"Loaded {len(images)} images")

    print("\n[3/4] Loading LiDAR data...")
    if args.lidar_dir:
        lidar_points = load_lidar_points(args.lidar_dir, args.num_cameras)
        lidar_available = [p is not None for p in lidar_points]
        print(f"LiDAR available: {sum(lidar_available)}/{args.num_cameras}")
    else:
        print("No LiDAR directory specified, reconstruction will be in relative scale")
        lidar_points = None

    print("\n[4/4] Running reconstruction...")
    print(f"Model: {args.model}")
    print(f"RANSAC: {'disabled' if args.no_ransac else 'enabled'}")

    # Run reconstruction
    print(f"Max view angle: {args.max_view_angle}°")
    print(f"Max depth: {args.max_depth}m")

    result = reconstruct_intersection(
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        lidar_points=lidar_points,
        model_name=args.model,
        export_dir=args.output_dir,
        conf_threshold_percentile=args.conf_threshold,
        use_ransac=not args.no_ransac,
        max_view_angle=args.max_view_angle,
        max_depth=args.max_depth,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Reconstruction Complete!")
    print("=" * 60)
    print(f"\nTotal fused points: {len(result.fused_points):,}")
    print(f"\nPer-camera statistics:")
    for cam_id, n_points in result.num_points_per_camera.items():
        scale = result.scale_factors.get(cam_id, 1.0)
        print(f"  {cam_id}: {n_points:,} points, scale={scale:.4f}")

    if result.alignment_results:
        print(f"\nLiDAR alignment statistics:")
        for cam_id, align_result in result.alignment_results.items():
            print(f"  {cam_id}:")
            print(f"    Scale: {align_result.scale_factor:.4f}")
            print(f"    Inlier ratio: {align_result.inlier_ratio:.1%}")
            print(f"    Residual: {align_result.residual_mean:.3f}m +/- {align_result.residual_std:.3f}m")

    print(f"\nResults saved to: {args.output_dir}")
    print("  - fused_pointcloud.ply: Combined point cloud")
    print("  - cam_*/pointcloud.ply: Per-camera point clouds")
    print("  - cam_*/depth_conf.npz: Depth and confidence maps")
    print("  - alignment_stats.txt: LiDAR alignment statistics")


if __name__ == "__main__":
    main()
