#!/usr/bin/env python3
"""
Multi-Camera Point Cloud Fusion with LiDAR Depth Completion.

Pipeline:
1. Project LiDAR points to each camera → sparse depth map
2. Use depth completion to fill in dense depth (guided by RGB)
3. Convert dense depth to point cloud
4. Merge all cameras

This approach is better than monocular depth estimation because:
- LiDAR provides accurate metric depth (no scale ambiguity)
- Depth completion preserves edges guided by RGB
- Much better coverage for far-field regions
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

from depth_anything_3.datasets.car_road_dataset import CarRoadDatasetLoader
from depth_anything_3.utils.depth_completion import (
    DepthCompleter,
    create_sparse_depth_from_lidar,
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
    max_depth: float = 300.0,
    min_depth: float = 0.5,
):
    """
    Convert depth map to colored point cloud in world coordinates.

    Args:
        depth: (H, W) metric depth
        rgb: (H, W, 3) RGB image
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        max_depth: maximum depth to keep
        min_depth: minimum depth to keep

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
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]

    if len(z) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

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
    parser = argparse.ArgumentParser(description="Multi-camera point cloud fusion with LiDAR depth completion")

    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_TianJin")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--cameras", type=str, default="0,3,6,9",
                        help="Comma-separated camera IDs (default: 0,3,6,9 for 4 pinhole)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--completion_method", type=str, default="sam",
                        choices=["sam", "simple", "completionformer"],
                        help="Depth completion method (default: sam)")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint (auto-download if not provided)")
    parser.add_argument("--sam_points_per_side", type=int, default=32,
                        help="SAM grid density for segmentation (higher = finer, default: 32)")
    parser.add_argument("--completionformer_weights", type=str, default=None,
                        help="Path to CompletionFormer weights (if using completionformer)")
    parser.add_argument("--max_depth", type=float, default=300.0,
                        help="Maximum depth (default: 300m)")
    parser.add_argument("--da3_encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl", "vitg"],
                        help="DA3 encoder size (default: vitl)")
    parser.add_argument("--process_res", type=int, default=None,
                        help="Processing resolution (default: native)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    camera_ids = [c.strip() for c in args.cameras.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Multi-Camera Fusion with LiDAR Depth Completion")
    print(f"{'='*60}")
    print(f"  Scene: {args.scene}")
    print(f"  Timestamp: {args.timestamp}")
    print(f"  Cameras: {camera_ids}")
    print(f"  Completion method: {args.completion_method}")
    print(f"  Max depth: {args.max_depth}m")

    # Load data
    print("\n[1/4] Loading dataset...")
    loader = CarRoadDatasetLoader(data_root=args.data_root)

    # Load merged LiDAR
    lidar_points = loader.load_merged_points(args.scene, args.timestamp)
    print(f"  LiDAR: {len(lidar_points):,} points")
    print(f"  LiDAR range: X=[{lidar_points[:,0].min():.1f}, {lidar_points[:,0].max():.1f}], "
          f"Y=[{lidar_points[:,1].min():.1f}, {lidar_points[:,1].max():.1f}], "
          f"Z=[{lidar_points[:,2].min():.1f}, {lidar_points[:,2].max():.1f}]")

    # Initialize DA3 model (for relative depth)
    print("\n[2/5] Initializing DA3 model...")
    import torch
    from depth_anything_3.api import DepthAnything3

    da3_model = DepthAnything3.from_pretrained(
        "depth-anything/DA3NESTED-GIANT-LARGE"
    ).to(args.device).eval()
    print(f"  DA3 model: DA3NESTED-GIANT-LARGE")

    # Initialize depth completer
    print("\n[3/5] Initializing depth completer...")
    model_path = args.sam_checkpoint if args.completion_method == "sam" else args.completionformer_weights
    completer = DepthCompleter(
        method=args.completion_method,
        model_path=model_path,
        device=args.device,
        max_depth=args.max_depth,
        sam_points_per_side=args.sam_points_per_side,
    )
    print(f"  Method: {args.completion_method}")

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
        intrinsics = cam.intrinsics.copy()
        extrinsics = cam.extrinsics_w2c

        H, W = image.shape[:2]
        print(f"    Image: {W}x{H}")

        # Optionally resize
        if args.process_res is not None and args.process_res != H:
            scale = args.process_res / H
            new_H = args.process_res
            new_W = int(W * scale)
            image = cv2.resize(image, (new_W, new_H))
            intrinsics[0, :] *= scale
            intrinsics[1, :] *= scale
            H, W = new_H, new_W
            print(f"    Resized to: {W}x{H}")

        # Step 1: Project LiDAR to camera → sparse depth
        sparse_depth, valid_mask = create_sparse_depth_from_lidar(
            lidar_points=lidar_points[:, :3],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
        )

        n_sparse = valid_mask.sum()
        sparse_ratio = n_sparse / (H * W) * 100
        print(f"    Sparse depth: {n_sparse:,} points ({sparse_ratio:.2f}% coverage)")

        if n_sparse < 100:
            print(f"    WARNING: Very few LiDAR points visible, skipping")
            continue

        sparse_depth_range = sparse_depth[valid_mask]
        print(f"    Sparse depth range: [{sparse_depth_range.min():.1f}, {sparse_depth_range.max():.1f}]m")

        # Step 2: DA3 relative depth inference
        print(f"    Running DA3...")
        prediction = da3_model.inference(image=[image], process_res=H)
        da3_depth_raw = prediction.depth[0]  # Relative depth

        # Resize DA3 output to match image size if needed
        da3_H, da3_W = da3_depth_raw.shape
        if da3_H != H or da3_W != W:
            da3_depth = cv2.resize(da3_depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            da3_depth = da3_depth_raw

        print(f"    DA3 depth range: [{da3_depth.min():.3f}, {da3_depth.max():.3f}] (relative)")

        # Step 3: Depth completion (SAM + DA3 + LiDAR)
        dense_depth, debug_info = completer.complete_with_debug(
            rgb=image,
            sparse_depth=sparse_depth,
            mask=valid_mask,
            da3_depth=da3_depth,
        )

        dense_valid = dense_depth > 0.1
        print(f"    Dense depth: {dense_valid.sum():,} pixels ({dense_valid.sum()/(H*W)*100:.1f}% coverage)")
        print(f"    Dense depth range: [{dense_depth[dense_valid].min():.1f}, {dense_depth[dense_valid].max():.1f}]m")

        # Step 4: Convert to point cloud
        points, colors = depth_to_pointcloud(
            depth=dense_depth,
            rgb=image,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            max_depth=args.max_depth,
        )

        print(f"    Points: {len(points):,}")
        if len(points) > 0:
            print(f"    Point cloud range: X=[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
                  f"Y=[{points[:,1].min():.1f}, {points[:,1].max():.1f}]")

        all_points.append(points)
        all_colors.append(colors)

        # Save per-camera results
        cam_dir = output_dir / f"cam{cam_id}"
        cam_dir.mkdir(exist_ok=True)

        # 1. RGB image
        Image.fromarray(image).save(cam_dir / "1_rgb.png")

        # 2. Sparse depth (LiDAR projection)
        sparse_vis = image.copy()
        ys, xs = np.where(valid_mask)
        sparse_color = colorize_depth(sparse_depth, vmin=0, vmax=args.max_depth)
        for y, x in zip(ys, xs):
            cv2.circle(sparse_vis, (x, y), 2, sparse_color[y, x].tolist(), -1)
        Image.fromarray(sparse_vis).save(cam_dir / "2_lidar_sparse.png")

        # 3. DA3 relative depth
        da3_vis = colorize_depth(da3_depth, vmin=0, vmax=1.0)  # Relative depth [0,1]
        Image.fromarray(da3_vis).save(cam_dir / "3_da3_relative.png")

        # 4. Dense depth (completed)
        Image.fromarray(colorize_depth(dense_depth, vmin=0, vmax=args.max_depth)).save(
            cam_dir / "4_dense_depth.png"
        )

        # 4b. SAM debug visualizations (if available)
        if debug_info:
            debug_dir = cam_dir / "debug_sam"
            debug_dir.mkdir(exist_ok=True)

            # SAM region visualization (colored regions)
            if 'region_vis' in debug_info:
                Image.fromarray(debug_info['region_vis']).save(debug_dir / "a_sam_regions.png")

            # Region labels as grayscale
            if 'region_labels' in debug_info:
                labels = debug_info['region_labels']
                labels_norm = (labels / max(labels.max(), 1) * 255).astype(np.uint8)
                Image.fromarray(labels_norm).save(debug_dir / "b_region_labels.png")

            # Sparse points overlaid on regions
            if 'region_vis' in debug_info:
                overlay = debug_info['region_vis'].copy()
                ys, xs = np.where(valid_mask)
                for y, x in zip(ys, xs):
                    cv2.circle(overlay, (x, y), 3, (255, 255, 255), -1)
                    cv2.circle(overlay, (x, y), 2, sparse_color[y, x].tolist(), -1)
                Image.fromarray(overlay).save(debug_dir / "c_regions_with_sparse.png")

            # Depth before smoothing
            if 'dense_before_smooth' in debug_info:
                Image.fromarray(colorize_depth(debug_info['dense_before_smooth'],
                                               vmin=0, vmax=args.max_depth)).save(
                    debug_dir / "d_depth_before_smooth.png"
                )

            # Completed mask (regions with enough points)
            if 'completed_mask_before_fill' in debug_info:
                completed = debug_info['completed_mask_before_fill'].astype(np.uint8) * 255
                Image.fromarray(completed).save(debug_dir / "e_completed_regions.png")

            # DA3 relative depth
            if 'da3_depth' in debug_info:
                da3_debug = colorize_depth(debug_info['da3_depth'], vmin=0, vmax=1.0)
                Image.fromarray(da3_debug).save(debug_dir / "f_da3_relative.png")

            # Region statistics
            if 'region_point_counts' in debug_info:
                counts = debug_info['region_point_counts']
                scales = debug_info.get('region_scales', {})
                with open(debug_dir / "region_stats.txt", 'w') as f:
                    f.write(f"Total regions: {debug_info.get('n_regions', 0)}\n")
                    f.write(f"Regions with >= 5 points (local scale): {len(scales)}\n")
                    f.write(f"Regions with < 5 points (global scale): {len([c for c in counts.values() if 0 < c < 5])}\n")
                    f.write(f"\nRegion point counts and scales:\n")
                    for rid, cnt in sorted(counts.items(), key=lambda x: -x[1])[:50]:
                        scale = scales.get(rid, "global")
                        if isinstance(scale, float):
                            f.write(f"  Region {rid}: {cnt} points, scale={scale:.3f}\n")
                        else:
                            f.write(f"  Region {rid}: {cnt} points, scale={scale}\n")

            print(f"    Saved SAM debug to: {debug_dir}")

        # 5. Depth error at LiDAR points
        if valid_mask.any():
            error_map = np.zeros_like(dense_depth)
            error_map[valid_mask] = np.abs(dense_depth[valid_mask] - sparse_depth[valid_mask])

            # Stats
            errors = error_map[valid_mask]
            print(f"    Completion error: mean={errors.mean():.2f}m, "
                  f"median={np.median(errors):.2f}m, max={errors.max():.2f}m")

            # Colorize (0-5m range)
            error_norm = np.clip(error_map / 5.0, 0, 1)
            error_color = (plt.get_cmap('hot')(error_norm)[:, :, :3] * 255).astype(np.uint8)
            error_color[~valid_mask] = 0
            Image.fromarray(error_color).save(cam_dir / "5_completion_error.png")

        # 6. Save point cloud
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

    # Also save LiDAR as reference (with colors based on height)
    lidar_z = lidar_points[:, 2]
    z_norm = (lidar_z - lidar_z.min()) / (lidar_z.max() - lidar_z.min() + 1e-6)
    lidar_colors = (plt.get_cmap('viridis')(z_norm)[:, :3] * 255).astype(np.uint8)
    save_ply(str(output_dir / "lidar_reference.ply"), lidar_points[:, :3], lidar_colors)

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - merged_pointcloud.ply: Fused camera point cloud (dense, colored)")
    print(f"  - lidar_reference.ply: Original LiDAR (height-colored)")
    for cam_id in camera_ids:
        print(f"  - cam{cam_id}/: RGB, sparse depth, dense depth, error, pointcloud")

    print(f"\nView in CloudCompare:")
    print(f"  cloudcompare {output_dir}/merged_pointcloud.ply {output_dir}/lidar_reference.ply")


if __name__ == "__main__":
    main()
