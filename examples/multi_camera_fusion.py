#!/usr/bin/env python3
"""
Multi-Camera Point Cloud Fusion.

Fuses depth from 4 pinhole cameras into a unified 3D point cloud.
Each camera: DA3 → relative depth → region-wise alignment → world points → merge
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.datasets.car_road_dataset import CarRoadDatasetLoader
from depth_anything_3.utils.region_depth_alignment import (
    RegionWiseDepthAligner,
    project_lidar_to_image,
    create_sparse_depth_map,
)


def colorize_depth(depth, vmin=None, vmax=None, cmap='turbo'):
    """Convert depth to colored image."""
    if vmin is None:
        vmin = np.percentile(depth[depth > 0], 1) if (depth > 0).any() else 0
    if vmax is None:
        vmax = np.percentile(depth[depth > 0], 99) if (depth > 0).any() else 1

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    colormap = plt.get_cmap(cmap)
    depth_color = colormap(depth_norm)[:, :, :3]
    depth_color = (depth_color * 255).astype(np.uint8)
    depth_color[depth <= 0] = 0

    return depth_color


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    confidence: np.ndarray = None,
    max_depth: float = 150.0,
    conf_threshold: float = 0.3,
    max_points: int = 300000,
):
    """
    Convert depth map to colored point cloud in world coordinates.

    Args:
        depth: (H, W) metric depth
        rgb: (H, W, 3) RGB image
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        confidence: (H, W) optional confidence map
        max_depth: maximum depth to keep
        conf_threshold: minimum confidence percentile
        max_points: maximum points to output

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB colors (0-255)
    """
    H, W = depth.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)
    z = depth.flatten()

    # Filter valid depth
    valid = (z > 0.1) & (z < max_depth) & np.isfinite(z)

    # Filter by confidence
    if confidence is not None:
        conf = confidence.flatten()
        conf_thresh = np.percentile(conf[conf > 0], conf_threshold * 100) if (conf > 0).any() else 0
        valid &= conf >= conf_thresh

    u, v, z = u[valid], v[valid], z[valid]

    # Subsample if too many
    if len(z) > max_points:
        indices = np.random.choice(len(z), max_points, replace=False)
        u, v, z = u[indices], v[indices], z[indices]

    # Unproject to camera space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)  # (N, 4)

    # Transform to world space
    c2w = np.linalg.inv(extrinsics)
    points_world = (c2w @ points_cam.T).T[:, :3]  # (N, 3)

    # Get colors
    u_int, v_int = u.astype(int), v.astype(int)
    colors = rgb[v_int, u_int]  # (N, 3)

    return points_world, colors


def save_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Save point cloud as PLY file."""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"    Saved: {filepath} ({len(points):,} points)")


def main():
    parser = argparse.ArgumentParser(description="Multi-camera point cloud fusion")

    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_TianJin")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--cameras", type=str, default="0,3,6,9",
                        help="Comma-separated camera IDs (default: 0,3,6,9 for 4 pinhole)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="da3-giant")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--max_depth", type=float, default=150.0)
    parser.add_argument("--max_points_per_cam", type=int, default=300000)

    args = parser.parse_args()

    camera_ids = [c.strip() for c in args.cameras.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Multi-Camera Point Cloud Fusion")
    print(f"{'='*60}")
    print(f"  Scene: {args.scene}")
    print(f"  Timestamp: {args.timestamp}")
    print(f"  Cameras: {camera_ids}")

    # Load data
    print("\n[1/5] Loading dataset...")
    loader = CarRoadDatasetLoader(data_root=args.data_root)

    # Load merged LiDAR
    lidar_points = loader.load_merged_points(args.scene, args.timestamp)
    print(f"  LiDAR: {len(lidar_points):,} points")

    # Load DA3 model
    print("\n[2/5] Loading DA3 model...")
    model = DepthAnything3.from_pretrained(
        {"da3-giant": "depth-anything/DA3-GIANT",
         "da3-large": "depth-anything/DA3-LARGE"}.get(args.model, args.model)
    )
    model.to(args.device)
    model.eval()

    # Initialize aligner
    print("\n[3/5] Initializing SAM aligner...")
    aligner = RegionWiseDepthAligner(
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        min_anchors_per_region=3,
        sam_points_per_side=32,
    )

    if not aligner.sam_available:
        print("  WARNING: SAM not available!")
        return

    # Process each camera
    print("\n[4/5] Processing cameras...")
    all_points = []
    all_colors = []

    for cam_id in camera_ids:
        print(f"\n  --- Camera {cam_id} ---")

        # Load image
        try:
            image, _ = loader.load_image(args.scene, cam_id, args.timestamp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"    Skip: {e}")
            continue

        cam = loader.cameras[cam_id]
        intrinsics = cam.intrinsics
        extrinsics = cam.extrinsics_w2c

        H, W = image.shape[:2]

        # DA3 inference
        prediction = model.inference(image=[image], process_res=518)
        relative_depth = prediction.depth[0]
        confidence = prediction.conf[0]

        proc_H, proc_W = relative_depth.shape
        print(f"    DA3: {proc_W}x{proc_H}")

        # Scale intrinsics
        scale_x = proc_W / W
        scale_y = proc_H / H
        K_scaled = intrinsics.copy()
        K_scaled[0, :] *= scale_x
        K_scaled[1, :] *= scale_y

        # Resize image
        image_resized = cv2.resize(image, (proc_W, proc_H))

        # Region-wise alignment
        result = aligner.align(
            image=image_resized,
            relative_depth=relative_depth,
            lidar_points=lidar_points,
            intrinsics=K_scaled,
            extrinsics=extrinsics,
        )

        print(f"    Regions: {result.num_regions}, Coverage: {result.coverage*100:.1f}%")

        # Convert to point cloud
        points, colors = depth_to_pointcloud(
            depth=result.aligned_depth,
            rgb=image_resized,
            intrinsics=K_scaled,
            extrinsics=extrinsics,
            confidence=confidence,
            max_depth=args.max_depth,
            conf_threshold=0.3,
            max_points=args.max_points_per_cam,
        )

        print(f"    Points: {len(points):,}")

        all_points.append(points)
        all_colors.append(colors)

        # Save per-camera results
        cam_dir = output_dir / f"cam{cam_id}"
        cam_dir.mkdir(exist_ok=True)

        Image.fromarray(image_resized).save(cam_dir / "rgb.png")
        Image.fromarray(colorize_depth(result.aligned_depth, vmin=0, vmax=args.max_depth)).save(
            cam_dir / "depth.png"
        )
        save_ply(str(cam_dir / "pointcloud.ply"), points, colors)

    # Merge all point clouds
    print(f"\n[5/5] Merging point clouds...")

    if len(all_points) == 0:
        print("  ERROR: No valid cameras!")
        return

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    print(f"  Total points: {len(merged_points):,}")

    # Save merged point cloud
    save_ply(str(output_dir / "merged_pointcloud.ply"), merged_points, merged_colors)

    # Also save LiDAR as reference
    lidar_colors = np.full((len(lidar_points), 3), [128, 128, 128], dtype=np.uint8)
    save_ply(str(output_dir / "lidar_reference.ply"), lidar_points[:, :3], lidar_colors)

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - merged_pointcloud.ply: Fused 4-camera point cloud")
    print(f"  - lidar_reference.ply: Original LiDAR (gray)")
    for cam_id in camera_ids:
        print(f"  - cam{cam_id}/: Per-camera RGB, depth, pointcloud")

    print(f"\nView in CloudCompare/MeshLab:")
    print(f"  cloudcompare {output_dir}/merged_pointcloud.ply {output_dir}/lidar_reference.ply")


if __name__ == "__main__":
    main()
