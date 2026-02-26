#!/usr/bin/env python3
"""
Multi-view DA3 depth estimation and PLY point cloud fusion.
Uses calibrated camera poses for accurate point cloud fusion.
Supports LiDAR alignment to convert relative depth to metric depth.

Usage:
    # Relative depth with LiDAR alignment (recommended)
    python examples/da3_multiview_pointcloud.py \
        --data_root /mnt/car_road_data_TianJin \
        --scene 001_car0325_road0327_t1 \
        --timestamp 1742877031036 \
        --lidar_align \
        --output_dir ./output_da3_ply

    # Relative depth only (no alignment)
    python examples/da3_multiview_pointcloud.py \
        --scene 001_car0325_road0327_t1 \
        --timestamp 1742877031036 \
        --output_dir ./output_da3_ply
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.datasets.car_road_dataset import CarRoadDatasetLoader


def project_lidar_to_camera(
    lidar_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    img_width: int,
    img_height: int,
) -> tuple:
    """
    Project LiDAR points to camera image plane.

    Args:
        lidar_points: (N, 3) LiDAR points in world coordinates
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transformation
        img_width: image width
        img_height: image height

    Returns:
        uv: (M, 2) pixel coordinates of valid projections
        depths: (M,) depth values at those pixels
    """
    # Transform to camera coordinates
    N = lidar_points.shape[0]
    pts_homo = np.hstack([lidar_points, np.ones((N, 1))])  # (N, 4)
    pts_cam = (extrinsics @ pts_homo.T).T[:, :3]  # (N, 3)

    # Filter points behind the camera
    valid_depth = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid_depth]

    if len(pts_cam) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    # Project to image plane
    pts_2d = (intrinsics @ pts_cam.T).T  # (M, 3)
    uv = pts_2d[:, :2] / pts_2d[:, 2:3]  # (M, 2)
    depths = pts_cam[:, 2]  # (M,)

    # Filter points outside image
    valid_uv = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
    )
    uv = uv[valid_uv]
    depths = depths[valid_uv]

    return uv, depths


def align_depth_with_lidar(
    pred_depth: np.ndarray,
    lidar_uv: np.ndarray,
    lidar_depths: np.ndarray,
    method: str = "scale_shift",
) -> tuple:
    """
    Align predicted depth with LiDAR ground truth using least squares.

    Args:
        pred_depth: (H, W) predicted relative depth
        lidar_uv: (M, 2) pixel coordinates of LiDAR projections
        lidar_depths: (M,) LiDAR depth values
        method: "scale_shift" for affine alignment, "scale" for scale only

    Returns:
        aligned_depth: (H, W) aligned depth map
        scale: fitted scale factor
        shift: fitted shift (0 if method="scale")
    """
    if len(lidar_uv) < 10:
        print(f"  Warning: Only {len(lidar_uv)} LiDAR points, using default scale")
        return pred_depth, 1.0, 0.0

    # Sample predicted depth at LiDAR locations
    u = lidar_uv[:, 0].astype(int)
    v = lidar_uv[:, 1].astype(int)
    pred_at_lidar = pred_depth[v, u]

    # Filter out invalid predictions
    valid = pred_at_lidar > 0
    pred_at_lidar = pred_at_lidar[valid]
    lidar_d = lidar_depths[valid]

    if len(pred_at_lidar) < 10:
        print(f"  Warning: Only {len(pred_at_lidar)} valid matches, using default scale")
        return pred_depth, 1.0, 0.0

    if method == "scale_shift":
        # Solve: lidar_d = scale * pred_at_lidar + shift
        # Using least squares: [pred, 1] @ [scale, shift]^T = lidar_d
        A = np.vstack([pred_at_lidar, np.ones_like(pred_at_lidar)]).T
        result, _, _, _ = np.linalg.lstsq(A, lidar_d, rcond=None)
        scale, shift = result
    else:
        # Scale only: lidar_d = scale * pred_at_lidar
        scale = np.median(lidar_d / pred_at_lidar)
        shift = 0.0

    # Apply alignment
    aligned_depth = pred_depth * scale + shift
    aligned_depth = np.maximum(aligned_depth, 0)  # No negative depths

    # Calculate alignment error
    aligned_at_lidar = aligned_depth[v[valid], u[valid]]
    mae = np.mean(np.abs(aligned_at_lidar - lidar_d))
    print(f"  LiDAR alignment: scale={scale:.4f}, shift={shift:.4f}, MAE={mae:.4f}m")

    return aligned_depth, scale, shift


def depth_to_pointcloud(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    conf: np.ndarray = None,
    conf_threshold: float = 0.3,
    max_depth: float = 100.0,
    downsample: int = 1,
) -> tuple:
    """
    Convert depth map to colored point cloud in world coordinates.

    Args:
        depth: (H, W) depth map
        image: (H, W, 3) RGB image
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transformation
        conf: (H, W) confidence map (optional)
        conf_threshold: Minimum confidence to keep points
        max_depth: Maximum depth to keep
        downsample: Downsample factor for points

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB colors (0-255)
    """
    h, w = depth.shape

    # Resize image to match depth if needed
    if image.shape[:2] != depth.shape:
        image = cv2.resize(image, (w, h))

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Downsample
    if downsample > 1:
        u = u[::downsample, ::downsample]
        v = v[::downsample, ::downsample]
        depth = depth[::downsample, ::downsample]
        image = image[::downsample, ::downsample]
        if conf is not None:
            conf = conf[::downsample, ::downsample]

    # Flatten
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    colors = image.reshape(-1, 3)

    # Create mask for valid points
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    if conf is not None:
        conf_flat = conf.flatten()
        valid &= conf_flat > conf_threshold

    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = colors[valid]

    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (N, 3) camera coordinates
    points_cam = np.stack([x, y, z], axis=1)

    # Transform to world coordinates
    # extrinsics is world-to-camera: p_cam = R @ p_world + t
    # So p_world = R^T @ (p_cam - t)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    points_world = (R_inv @ points_cam.T).T + t_inv

    return points_world, colors


def save_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Save point cloud as PLY file."""
    n_points = len(points)

    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filepath, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Saved PLY with {n_points:,} points to {filepath}")


def visualize_depth(depth: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Visualize depth map."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_min, d_max = np.percentile(valid, [2, 98])
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    return depth_color


def main():
    parser = argparse.ArgumentParser(description="Multi-view DA3 point cloud fusion")
    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_TianJin")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_da3_ply")
    parser.add_argument("--model", type=str, default="da3-giant",
                        help="Model: da3-giant, da3-large, da3nested-giant-large")
    parser.add_argument("--lidar_align", action="store_true",
                        help="Align depth with LiDAR (recommended)")
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Max depth (meters)")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Manual depth scale")
    args = parser.parse_args()

    if args.lidar_align:
        print("LiDAR alignment enabled - will convert relative depth to metric depth")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.scene, args.timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    loader = CarRoadDatasetLoader(args.data_root)

    # Load model
    print(f"\nLoading DA3 model: {args.model}")
    model_repo_map = {
        "da3-giant": "depth-anything/DA3-GIANT",
        "da3-large": "depth-anything/DA3-LARGE",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-small": "depth-anything/DA3-SMALL",
        "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
        "da3nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "da3metric-large": "depth-anything/DA3METRIC-LARGE",
    }
    repo_id = model_repo_map.get(args.model.lower(), args.model)
    print(f"Loading from: {repo_id}")
    model = DepthAnything3.from_pretrained(repo_id)
    model.to("cuda")
    model.eval()

    is_metric = "nested" in args.model.lower() or "metric" in args.model.lower()
    print(f"Metric depth: {is_metric}")

    # Get pinhole camera IDs
    pinhole_camera_ids = ["0", "3", "6", "9"]

    # Load scene data (with LiDAR if alignment enabled)
    print(f"\nLoading scene: {args.scene} @ {args.timestamp}")
    scene_data = loader.load_scene_data(
        scene_name=args.scene,
        timestamp=args.timestamp,
        camera_ids=pinhole_camera_ids,
        load_merged_lidar=args.lidar_align,
        load_individual_lidars=False,
    )

    # Get merged LiDAR points if available
    lidar_points = None
    if args.lidar_align and scene_data.merged_points is not None:
        lidar_points = scene_data.merged_points[:, :3]  # (N, 3) xyz
        print(f"Loaded merged LiDAR: {len(lidar_points):,} points")

    available_cameras = sorted(scene_data.images.keys())
    print(f"Available cameras: {available_cameras}")

    if not available_cameras:
        print("ERROR: No images found!")
        return

    # Get calibration (using our accurate calibrated poses)
    intrinsics_all, extrinsics_all = loader.get_camera_arrays(available_cameras)

    # Collect all points and colors
    all_points = []
    all_colors = []

    # Process each camera
    for i, cam_id in enumerate(available_cameras):
        print(f"\n{'='*50}")
        print(f"Processing camera {cam_id} ({i+1}/{len(available_cameras)})")
        print(f"{'='*50}")

        # Get image path
        img_path = scene_data.image_paths[cam_id]
        print(f"Image: {img_path}")

        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")

        # Run DA3 inference
        print(f"Running DA3 inference (process_res={args.process_res})...")
        prediction = model.inference(
            image=[img_path],
            process_res=args.process_res,
        )

        depth = prediction.depth[0]  # (H, W)
        conf = prediction.conf[0] if prediction.conf is not None else None

        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")

        # Apply manual scale
        depth = depth * args.depth_scale

        # Resize depth to original image size
        depth_resized = cv2.resize(depth, (img.shape[1], img.shape[0]))
        if conf is not None:
            conf_resized = cv2.resize(conf, (img.shape[1], img.shape[0]))
        else:
            conf_resized = None

        # Get calibration for this camera (our accurate calibrated poses)
        K = intrinsics_all[i]  # (3, 3)
        E = extrinsics_all[i]  # (4, 4)

        # LiDAR alignment if enabled
        if lidar_points is not None:
            print("Aligning depth with LiDAR...")
            lidar_uv, lidar_depths = project_lidar_to_camera(
                lidar_points, K, E, img.shape[1], img.shape[0]
            )
            print(f"  Projected {len(lidar_uv)} LiDAR points to camera")
            depth_resized, scale, shift = align_depth_with_lidar(
                depth_resized, lidar_uv, lidar_depths, method="scale_shift"
            )
            print(f"  Aligned depth range: [{depth_resized.min():.4f}, {depth_resized.max():.4f}]")

        print(f"Using calibrated intrinsics:\n{K}")
        print(f"Using calibrated extrinsics (w2c):\n{E}")

        # Convert to point cloud
        print("Converting to point cloud...")
        points, colors = depth_to_pointcloud(
            depth=depth_resized,
            image=img_rgb,
            intrinsics=K,
            extrinsics=E,
            conf=conf_resized,
            conf_threshold=args.conf_threshold,
            max_depth=args.max_depth,
            downsample=args.downsample,
        )

        print(f"Generated {len(points):,} points")

        all_points.append(points)
        all_colors.append(colors)

        # Save individual camera results
        cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
        os.makedirs(cam_output_dir, exist_ok=True)

        # Save depth visualization
        depth_vis = visualize_depth(depth_resized)
        cv2.imwrite(os.path.join(cam_output_dir, "depth.jpg"), depth_vis)

        # Save individual PLY
        save_ply(
            os.path.join(cam_output_dir, "pointcloud.ply"),
            points, colors
        )

        # Save side-by-side visualization
        h, w = img.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = img
        vis[:, w:] = depth_vis
        cv2.imwrite(os.path.join(cam_output_dir, "comparison.jpg"), vis)

    # Fuse all point clouds
    print(f"\n{'='*50}")
    print("Fusing all point clouds...")
    print(f"{'='*50}")

    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)

    print(f"Total fused points: {len(fused_points):,}")

    # Save fused PLY
    fused_ply_path = os.path.join(output_dir, "fused_pointcloud.ply")
    save_ply(fused_ply_path, fused_points, fused_colors)

    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for i, cam_id in enumerate(available_cameras):
        print(f"  Camera {cam_id}: {len(all_points[i]):,} points")
    print(f"  Total: {len(fused_points):,} points")
    print(f"\nResults saved to: {output_dir}")
    print(f"Fused PLY: {fused_ply_path}")


if __name__ == "__main__":
    main()
