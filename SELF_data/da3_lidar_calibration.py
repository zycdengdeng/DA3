# -*- coding: utf-8 -*-
"""
DA3 Depth Calibration with LiDAR Points

This script calibrates DA3 depth estimation using sparse but accurate LiDAR points.
It computes a global scale factor to convert relative depth to metric depth.

Usage:
    python da3_lidar_calibration.py --data_dir ./SELF_data --output_dir ./output
"""

import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch
from PIL import Image

# DA3 imports
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import least_squares_scale_scalar


def rodrigues_to_rotation_matrix(rvec):
    """Convert Rodrigues vector to rotation matrix."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R


def load_pcd(path):
    """Load point cloud from PCD file."""
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    return xyz


def load_calibration(calib_path):
    """Load calibration data from JSON file."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    return calib


def get_camera_params(calib, cam_id):
    """
    Extract camera parameters from calibration.

    Returns:
        K: (3, 3) intrinsic matrix
        dist: distortion coefficients
        R_V2C: (3, 3) rotation from VirtualLidar to Camera
        t_V2C: (3,) translation from VirtualLidar to Camera
        is_fisheye: bool
    """
    cam = calib["camera"][str(cam_id)]

    # Intrinsic matrix
    K = np.asarray(cam["intri"], dtype=np.float64).reshape(3, 3)

    # Distortion coefficients
    dist = np.asarray(cam.get("distor", []), dtype=np.float64) if "distor" in cam else None

    # VirtualLidar to Camera transform
    R_V2C = rodrigues_to_rotation_matrix(cam["virtualLidarToCam"]["rotate"])
    t_V2C = np.asarray(cam["virtualLidarToCam"]["trans"], dtype=np.float64)

    is_fisheye = bool(cam.get("isFish", 0))

    return K, dist, R_V2C, t_V2C, is_fisheye


def project_points_to_camera(xyz_virtual, R_V2C, t_V2C, K, dist, is_fisheye, img_size):
    """
    Project 3D points from VirtualLidar frame to camera image plane.

    Args:
        xyz_virtual: (N, 3) points in VirtualLidar frame
        R_V2C: (3, 3) rotation matrix
        t_V2C: (3,) translation vector
        K: (3, 3) intrinsic matrix
        dist: distortion coefficients
        is_fisheye: whether fisheye camera model
        img_size: (H, W) image size

    Returns:
        uv: (M, 2) valid pixel coordinates
        depth: (M,) depth values (Z in camera frame)
        valid_mask: (N,) boolean mask of valid points
    """
    H, W = img_size

    # Transform to camera frame: Xc = R * Xv + t
    Xc = (R_V2C @ xyz_virtual.T).T + t_V2C.reshape(1, 3)

    # Filter points in front of camera
    front_mask = Xc[:, 2] > 0.1  # z > 0.1m

    if not np.any(front_mask):
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros(len(xyz_virtual), dtype=bool)

    # Project to image plane
    xyz_front = xyz_virtual[front_mask].astype(np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(R_V2C.astype(np.float64))
    tvec = t_V2C.astype(np.float64).reshape(3, 1)

    if is_fisheye and dist is not None and len(dist) == 4:
        uv, _ = cv2.fisheye.projectPoints(xyz_front, rvec, tvec, K, dist)
    else:
        uv, _ = cv2.projectPoints(xyz_front, rvec, tvec, K, dist)

    uv = uv.reshape(-1, 2)
    depth = Xc[front_mask, 2]

    # Filter points within image bounds
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    # Build full valid mask
    valid_mask = np.zeros(len(xyz_virtual), dtype=bool)
    front_indices = np.where(front_mask)[0]
    valid_mask[front_indices[in_image]] = True

    return uv[in_image], depth[in_image], valid_mask


def compute_scale_factor(da3_depth, lidar_depth, da3_uv, min_points=50):
    """
    Compute scale factor using least squares: lidar_depth ≈ scale * da3_depth

    Args:
        da3_depth: DA3 predicted depth map (H, W)
        lidar_depth: LiDAR depth values at projected points (M,)
        da3_uv: pixel coordinates of LiDAR points (M, 2)
        min_points: minimum number of valid points required

    Returns:
        scale: scale factor
        num_points: number of points used
    """
    # Sample DA3 depth at LiDAR point locations
    u = da3_uv[:, 0].astype(int)
    v = da3_uv[:, 1].astype(int)

    H, W = da3_depth.shape
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[valid], v[valid]
    lidar_depth_valid = lidar_depth[valid]

    da3_depth_at_lidar = da3_depth[v, u]

    # Filter invalid depth values
    valid_depth = (da3_depth_at_lidar > 0.01) & (lidar_depth_valid > 0.1) & (lidar_depth_valid < 500)

    da3_d = da3_depth_at_lidar[valid_depth]
    lidar_d = lidar_depth_valid[valid_depth]

    if len(da3_d) < min_points:
        print(f"Warning: Only {len(da3_d)} valid points, need at least {min_points}")
        return 1.0, len(da3_d)

    # Least squares: scale = dot(lidar, da3) / dot(da3, da3)
    da3_tensor = torch.from_numpy(da3_d).float()
    lidar_tensor = torch.from_numpy(lidar_d).float()

    scale = least_squares_scale_scalar(lidar_tensor, da3_tensor)

    return scale.item(), len(da3_d)


def visualize_projection(img, uv, depth, output_path):
    """Visualize LiDAR points projected onto image."""
    vis = img.copy()
    if len(uv) == 0:
        cv2.imwrite(output_path, vis)
        return

    # Normalize depth for colormap
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colors = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    for (u, v_coord), c in zip(uv, colors):
        ui, vi = int(round(u)), int(round(v_coord))
        cv2.circle(vis, (ui, vi), 2, tuple(int(x) for x in c[0]), -1)

    cv2.imwrite(output_path, vis)
    print(f"Saved projection visualization: {output_path}")


def visualize_depth(depth, output_path, colormap=cv2.COLORMAP_MAGMA):
    """Visualize depth map."""
    # Normalize depth
    valid = depth > 0
    if valid.sum() == 0:
        cv2.imwrite(output_path, np.zeros_like(depth, dtype=np.uint8))
        return

    d_min, d_max = depth[valid].min(), depth[valid].max()
    d_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    d_norm = np.clip(d_norm, 0, 1)

    depth_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), colormap)
    depth_vis[~valid] = 0

    cv2.imwrite(output_path, depth_vis)
    print(f"Saved depth visualization: {output_path}")


def depth_to_pointcloud(depth, K, R_V2C, t_V2C, image=None, max_depth=200.0, downsample=1):
    """
    Convert depth map to point cloud in VirtualLidar (world) frame.

    Args:
        depth: (H, W) depth map in camera frame
        K: (3, 3) intrinsic matrix
        R_V2C: (3, 3) rotation from VirtualLidar to Camera
        t_V2C: (3,) translation from VirtualLidar to Camera
        image: (H, W, 3) RGB image for colors (optional)
        max_depth: maximum depth to include
        downsample: downsample factor (1 = full resolution)

    Returns:
        xyz_world: (N, 3) points in VirtualLidar frame
        colors: (N, 3) RGB colors (0-255) or None
    """
    H, W = depth.shape

    # Create pixel grid
    u = np.arange(0, W, downsample)
    v = np.arange(0, H, downsample)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    # Get depth values
    d = depth[v, u]

    # Filter valid depth
    valid = (d > 0.1) & (d < max_depth) & np.isfinite(d)
    u, v, d = u[valid], v[valid], d[valid]

    if len(d) == 0:
        return np.zeros((0, 3)), None

    # Backproject to camera frame
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X_cam = (u - cx) * d / fx
    Y_cam = (v - cy) * d / fy
    Z_cam = d

    xyz_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # (N, 3)

    # Transform to VirtualLidar (world) frame
    # Camera -> VirtualLidar: X_v = R_V2C^T @ (X_c - t_V2C)
    R_C2V = R_V2C.T
    t_C2V = -R_C2V @ t_V2C

    xyz_world = (R_C2V @ xyz_cam.T).T + t_C2V.reshape(1, 3)

    # Get colors
    colors = None
    if image is not None:
        colors = image[v, u]  # (N, 3) BGR
        colors = colors[:, ::-1]  # BGR -> RGB

    return xyz_world, colors


def save_pointcloud_ply(xyz, colors, output_path):
    """Save point cloud to PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        # Normalize colors to [0, 1]
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud: {output_path} ({len(xyz)} points)")


def main():
    parser = argparse.ArgumentParser(description="DA3 Depth Calibration with LiDAR")
    parser.add_argument("--data_dir", type=str, default="./SELF_data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-Large", help="DA3 model name")
    parser.add_argument("--process_res", type=int, default=504, help="Processing resolution")
    parser.add_argument("--cam_ids", type=str, default="0,3,6,9", help="Camera IDs to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse camera IDs
    cam_ids = [int(x) for x in args.cam_ids.split(",")]

    # Find data files
    data_dir = args.data_dir
    calib_path = os.path.join(data_dir, "calib.json")

    # Find timestamp from PCD file
    pcd_files = [f for f in os.listdir(data_dir) if f.endswith(".pcd") and not f.startswith("rad")]
    if not pcd_files:
        raise FileNotFoundError("No merged PCD file found")

    timestamp = pcd_files[0].replace(".pcd", "")
    print(f"Processing timestamp: {timestamp}")

    # Load calibration
    calib = load_calibration(calib_path)

    # Load point cloud (VirtualLidar frame)
    pcd_path = os.path.join(data_dir, f"{timestamp}.pcd")
    xyz_virtual = load_pcd(pcd_path)
    print(f"Loaded point cloud: {len(xyz_virtual)} points")

    # Collect camera images
    images = []
    valid_cam_ids = []
    for cam_id in cam_ids:
        img_path = os.path.join(data_dir, f"cam{cam_id}_{timestamp}.png")
        if os.path.exists(img_path):
            images.append(img_path)
            valid_cam_ids.append(cam_id)
            print(f"Found camera {cam_id}: {img_path}")
        else:
            print(f"Warning: Camera {cam_id} image not found")

    if not images:
        raise FileNotFoundError("No camera images found")

    # Load DA3 model
    print(f"\nLoading DA3 model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model)
    model.cuda().eval()

    # Run DA3 inference
    print("\nRunning DA3 inference...")
    prediction = model.inference(
        image=images,
        process_res=args.process_res,
    )

    print(f"DA3 output depth shape: {prediction.depth.shape}")

    # Process each camera
    all_da3_depths = []
    all_lidar_depths = []
    all_uvs = []

    for i, cam_id in enumerate(valid_cam_ids):
        print(f"\n--- Processing Camera {cam_id} ---")

        # Get camera parameters
        K, dist, R_V2C, t_V2C, is_fisheye = get_camera_params(calib, cam_id)

        # Get DA3 depth for this camera
        da3_depth = prediction.depth[i]  # (H, W)
        img_size = da3_depth.shape

        # Load original image for visualization
        img = cv2.imread(images[i])
        orig_size = img.shape[:2]  # (H, W)

        # Project LiDAR points to camera
        uv, lidar_depth, valid_mask = project_points_to_camera(
            xyz_virtual, R_V2C, t_V2C, K, dist, is_fisheye, orig_size
        )

        print(f"  Projected {len(uv)} LiDAR points to camera")

        # Scale UV coordinates to DA3 depth resolution
        scale_x = img_size[1] / orig_size[1]
        scale_y = img_size[0] / orig_size[0]
        uv_scaled = uv.copy()
        uv_scaled[:, 0] *= scale_x
        uv_scaled[:, 1] *= scale_y

        # Visualize projection on original image
        vis_path = os.path.join(args.output_dir, f"cam{cam_id}_lidar_projection.png")
        visualize_projection(img, uv, lidar_depth, vis_path)

        # Collect for scale computation
        all_da3_depths.append(da3_depth)
        all_lidar_depths.append(lidar_depth)
        all_uvs.append(uv_scaled)

    # Compute global scale factor using all cameras
    print("\n--- Computing Global Scale Factor ---")

    combined_da3 = []
    combined_lidar = []

    for i, (da3_depth, lidar_depth, uv) in enumerate(zip(all_da3_depths, all_lidar_depths, all_uvs)):
        u = uv[:, 0].astype(int)
        v = uv[:, 1].astype(int)

        H, W = da3_depth.shape
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        da3_at_lidar = da3_depth[v[valid], u[valid]]
        lidar_valid = lidar_depth[valid]

        # Filter valid depths
        mask = (da3_at_lidar > 0.01) & (lidar_valid > 0.1) & (lidar_valid < 500)

        combined_da3.extend(da3_at_lidar[mask])
        combined_lidar.extend(lidar_valid[mask])

    combined_da3 = np.array(combined_da3)
    combined_lidar = np.array(combined_lidar)

    print(f"Total valid point pairs: {len(combined_da3)}")

    if len(combined_da3) < 50:
        print("Warning: Too few valid points for reliable scale estimation")
        scale = 1.0
    else:
        # Least squares scale
        scale = least_squares_scale_scalar(
            torch.from_numpy(combined_lidar).float(),
            torch.from_numpy(combined_da3).float()
        ).item()

    print(f"Computed scale factor: {scale:.4f}")

    # Compute error statistics
    calibrated_depth = combined_da3 * scale
    abs_error = np.abs(calibrated_depth - combined_lidar)
    rel_error = abs_error / (combined_lidar + 1e-6)

    print(f"\n--- Calibration Results ---")
    print(f"Scale factor: {scale:.4f}")
    print(f"Mean absolute error: {abs_error.mean():.3f} m")
    print(f"Median absolute error: {np.median(abs_error):.3f} m")
    print(f"Mean relative error: {rel_error.mean()*100:.2f} %")
    print(f"Median relative error: {np.median(rel_error)*100:.2f} %")

    # Apply scale and save calibrated depth maps + point clouds
    print("\n--- Saving Calibrated Depth Maps and Point Clouds ---")

    all_xyz = []
    all_colors = []

    for i, cam_id in enumerate(valid_cam_ids):
        da3_depth = prediction.depth[i]
        calibrated = da3_depth * scale

        # Save as NPZ
        npz_path = os.path.join(args.output_dir, f"cam{cam_id}_depth_calibrated.npz")
        np.savez(npz_path, depth=calibrated, scale=scale)
        print(f"Saved: {npz_path}")

        # Visualize
        vis_path = os.path.join(args.output_dir, f"cam{cam_id}_depth_calibrated_vis.png")
        visualize_depth(calibrated, vis_path)

        # Generate point cloud for this camera
        K, dist, R_V2C, t_V2C, is_fisheye = get_camera_params(calib, cam_id)

        # Load and resize image to match depth resolution
        img = cv2.imread(images[i])
        img_resized = cv2.resize(img, (calibrated.shape[1], calibrated.shape[0]))

        # Adjust intrinsics for resized image
        orig_H, orig_W = img.shape[:2]
        new_H, new_W = calibrated.shape
        K_scaled = K.copy()
        K_scaled[0, 0] *= new_W / orig_W  # fx
        K_scaled[1, 1] *= new_H / orig_H  # fy
        K_scaled[0, 2] *= new_W / orig_W  # cx
        K_scaled[1, 2] *= new_H / orig_H  # cy

        # Convert depth to point cloud
        xyz, colors = depth_to_pointcloud(
            calibrated, K_scaled, R_V2C, t_V2C,
            image=img_resized, max_depth=200.0, downsample=2
        )

        if len(xyz) > 0:
            all_xyz.append(xyz)
            if colors is not None:
                all_colors.append(colors)

            # Save individual camera point cloud
            ply_path = os.path.join(args.output_dir, f"cam{cam_id}_pointcloud.ply")
            save_pointcloud_ply(xyz, colors, ply_path)

    # Merge all point clouds
    if all_xyz:
        print("\n--- Merging Point Clouds ---")
        merged_xyz = np.concatenate(all_xyz, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0) if all_colors else None

        # Save merged point cloud
        merged_ply_path = os.path.join(args.output_dir, "merged_pointcloud.ply")
        save_pointcloud_ply(merged_xyz, merged_colors, merged_ply_path)

        # Also save with LiDAR points for comparison
        print("\n--- Adding LiDAR points for comparison ---")
        lidar_pcd_path = os.path.join(data_dir, f"{timestamp}.pcd")
        lidar_xyz = load_pcd(lidar_pcd_path)

        # Create comparison point cloud (DA3 in color, LiDAR in red)
        lidar_colors = np.full((len(lidar_xyz), 3), [255, 0, 0], dtype=np.uint8)  # Red

        combined_xyz = np.concatenate([merged_xyz, lidar_xyz], axis=0)
        combined_colors = np.concatenate([merged_colors, lidar_colors], axis=0) if merged_colors is not None else None

        combined_ply_path = os.path.join(args.output_dir, "combined_da3_lidar.ply")
        save_pointcloud_ply(combined_xyz, combined_colors, combined_ply_path)

    # Save calibration results
    results = {
        "scale_factor": scale,
        "num_points": len(combined_da3),
        "mean_abs_error_m": float(abs_error.mean()),
        "median_abs_error_m": float(np.median(abs_error)),
        "mean_rel_error_percent": float(rel_error.mean() * 100),
        "median_rel_error_percent": float(np.median(rel_error) * 100),
        "cameras": valid_cam_ids,
        "timestamp": timestamp,
    }

    results_path = os.path.join(args.output_dir, "calibration_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results: {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
