#!/usr/bin/env python3
"""
Roadside Reconstruction for Car-Road Cooperative Dataset.

This script reconstructs roadside scenes from the car-road cooperative
perception dataset using DA3 with LiDAR-guided depth alignment.

Dataset: /mnt/car_road_data_fix/
Calibration: /mnt/car_road_data_fix/support_info/calib.json

Usage:
    # Reconstruct a single timestamp from a scene
    python examples/car_road_reconstruction.py \
        --scene 001_car0325_road0327_t1 \
        --timestamp 1742877031036 \
        --output_dir ./output

    # Reconstruct all timestamps in a scene
    python examples/car_road_reconstruction.py \
        --scene 001_car0325_road0327_t1 \
        --all_timestamps \
        --output_dir ./output

    # List available scenes
    python examples/car_road_reconstruction.py --list_scenes

    # List timestamps in a scene
    python examples/car_road_reconstruction.py \
        --scene 001_car0325_road0327_t1 \
        --list_timestamps
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.datasets.car_road_dataset import (
    CarRoadDatasetLoader,
    get_lidar_for_camera,
)
from depth_anything_3.roadside_reconstruction import RoadsideReconstructor
from depth_anything_3.utils.lidar_alignment import (
    align_depth_with_lidar,
    create_sparse_depth_map,
)


def reconstruct_scene_timestamp(
    loader: CarRoadDatasetLoader,
    reconstructor: RoadsideReconstructor,
    scene_name: str,
    timestamp: str,
    output_dir: str,
    use_merged_lidar: bool = True,
    visualize: bool = True,
) -> dict:
    """
    Reconstruct a single timestamp from a scene.

    Args:
        loader: Dataset loader
        reconstructor: DA3 reconstructor
        scene_name: Scene folder name
        timestamp: Timestamp in milliseconds
        output_dir: Output directory
        use_merged_lidar: Use merged LiDAR instead of individual sensors
        visualize: Whether to save visualization images

    Returns:
        Dictionary with reconstruction results
    """
    print(f"\n{'='*60}")
    print(f"Reconstructing: {scene_name} @ {timestamp}")
    print(f"{'='*60}")

    # Get pinhole camera IDs
    pinhole_camera_ids = ["0", "3", "6", "9"]

    # Load scene data
    print("\n[1/4] Loading scene data...")
    scene_data = loader.load_scene_data(
        scene_name=scene_name,
        timestamp=timestamp,
        camera_ids=pinhole_camera_ids,
        load_merged_lidar=use_merged_lidar,
        load_individual_lidars=not use_merged_lidar,
    )

    # Check what data is available
    available_cameras = list(scene_data.images.keys())
    print(f"  Available cameras: {available_cameras}")
    print(f"  Merged LiDAR: {'Yes' if scene_data.merged_points is not None else 'No'}")
    if scene_data.merged_points is not None:
        print(f"  Merged points: {len(scene_data.merged_points):,}")

    if len(available_cameras) == 0:
        print("ERROR: No images found!")
        return {}

    # Prepare data for reconstruction
    print("\n[2/4] Preparing reconstruction data...")

    # Sort cameras for consistent ordering
    available_cameras = sorted(available_cameras)

    # Get image paths
    images = [scene_data.image_paths[cam_id] for cam_id in available_cameras]

    # Get calibration arrays
    intrinsics, extrinsics = loader.get_camera_arrays(available_cameras)
    print(f"  Cameras: {available_cameras}")
    print(f"  Intrinsics shape: {intrinsics.shape}")
    print(f"  Extrinsics shape: {extrinsics.shape}")

    # Prepare LiDAR data
    if use_merged_lidar and scene_data.merged_points is not None:
        # Use merged point cloud for all cameras
        lidar_points = [scene_data.merged_points] * len(available_cameras)
        lidar_mapping = list(range(len(available_cameras)))  # All use index 0
        lidar_mapping = [0] * len(available_cameras)
        lidar_points = [scene_data.merged_points]  # Single merged cloud
        print(f"  Using merged LiDAR: {len(scene_data.merged_points):,} points")
    else:
        # Use individual LiDARs paired with cameras
        lidar_points = []
        lidar_mapping = []
        for i, cam_id in enumerate(available_cameras):
            lid_id = get_lidar_for_camera(cam_id)
            if lid_id and lid_id in scene_data.lidar_points:
                lidar_points.append(scene_data.lidar_points[lid_id])
                lidar_mapping.append(len(lidar_points) - 1)
                print(f"  Camera {cam_id} -> LiDAR {lid_id}: {len(scene_data.lidar_points[lid_id]):,} points")
            else:
                lidar_mapping.append(None)
                print(f"  Camera {cam_id} -> No LiDAR")

    # Create output directory
    ts_output_dir = os.path.join(output_dir, scene_name, timestamp)
    os.makedirs(ts_output_dir, exist_ok=True)

    # Run reconstruction
    print("\n[3/4] Running DA3 reconstruction with LiDAR alignment...")

    result = reconstructor.reconstruct(
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        lidar_points=lidar_points if lidar_points else None,
        lidar_to_camera_mapping=lidar_mapping if lidar_points else None,
        export_dir=ts_output_dir,
        conf_threshold_percentile=30.0,
        use_ransac=True,
        ransac_threshold=0.15,
    )

    # Save visualizations
    if visualize:
        print("\n[4/4] Saving visualizations...")
        save_visualizations(
            loader, scene_data, result,
            available_cameras, ts_output_dir
        )

    # Print summary
    print(f"\n{'='*60}")
    print("Reconstruction Summary")
    print(f"{'='*60}")
    print(f"Total fused points: {len(result.fused_points):,}")
    for cam_id, n_points in result.num_points_per_camera.items():
        scale = result.scale_factors.get(cam_id, 1.0)
        print(f"  {cam_id}: {n_points:,} points, scale={scale:.4f}")

    if result.alignment_results:
        print(f"\nLiDAR Alignment:")
        for cam_id, align_result in result.alignment_results.items():
            print(f"  {cam_id}: scale={align_result.scale_factor:.4f}, "
                  f"inliers={align_result.inlier_ratio:.1%}, "
                  f"residual={align_result.residual_mean:.3f}m")

    print(f"\nResults saved to: {ts_output_dir}")

    return {
        "scene": scene_name,
        "timestamp": timestamp,
        "num_cameras": len(available_cameras),
        "total_points": len(result.fused_points),
        "scale_factors": result.scale_factors,
        "output_dir": ts_output_dir,
    }


def save_visualizations(
    loader: CarRoadDatasetLoader,
    scene_data,
    result,
    camera_ids: List[str],
    output_dir: str,
):
    """Save visualization images."""

    for i, cam_id in enumerate(camera_ids):
        result_cam_id = f"cam_{i}"

        if result_cam_id not in result.depth_maps:
            continue

        depth = result.depth_maps[result_cam_id]
        conf = result.confidence_maps[result_cam_id]

        # Load original image for overlay
        if cam_id in scene_data.images:
            img = scene_data.images[cam_id].copy()
        else:
            img = cv2.imread(scene_data.image_paths[cam_id])

        # Resize depth to match image if needed
        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
            conf = cv2.resize(conf, (img.shape[1], img.shape[0]))

        # Create depth visualization
        depth_valid = depth.copy()
        depth_valid[depth_valid <= 0] = np.nan

        depth_min = np.nanpercentile(depth_valid, 5)
        depth_max = np.nanpercentile(depth_valid, 95)

        depth_norm = (depth_valid - depth_min) / (depth_max - depth_min + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_norm = np.nan_to_num(depth_norm, nan=0)

        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # Create confidence visualization
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-6)
        conf_color = cv2.applyColorMap((conf_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Create side-by-side visualization
        h, w = img.shape[:2]
        vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
        vis[:, :w] = img
        vis[:, w:2*w] = depth_color
        vis[:, 2*w:] = conf_color

        # Add labels
        cv2.putText(vis, f"Camera {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, "Depth", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, "Confidence", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add scale factor if available
        if result_cam_id in result.scale_factors:
            scale = result.scale_factors[result_cam_id]
            cv2.putText(vis, f"Scale: {scale:.4f}", (w + 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        vis_path = os.path.join(output_dir, f"vis_cam{cam_id}.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  Saved: {vis_path}")

    # Also save LiDAR projection visualization if available
    if scene_data.merged_points is not None:
        for i, cam_id in enumerate(camera_ids):
            if cam_id not in scene_data.images:
                continue

            img = scene_data.images[cam_id].copy()
            uv, depths, valid = loader.project_points_to_camera(
                scene_data.merged_points, cam_id
            )

            if len(uv) > 0:
                # Color by depth
                d_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
                colors = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

                for (u, v), c in zip(uv, colors):
                    cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in c[0]), -1)

                vis_path = os.path.join(output_dir, f"lidar_proj_cam{cam_id}.jpg")
                cv2.imwrite(vis_path, img)
                print(f"  Saved: {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Roadside Reconstruction for Car-Road Cooperative Dataset"
    )

    # Data paths
    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_fix",
                        help="Root directory of the dataset")
    parser.add_argument("--calib_path", type=str, default=None,
                        help="Path to calib.json (default: data_root/support_info/calib.json)")

    # Scene selection
    parser.add_argument("--scene", type=str, help="Scene folder name")
    parser.add_argument("--timestamp", type=str, help="Timestamp to reconstruct")
    parser.add_argument("--all_timestamps", action="store_true",
                        help="Reconstruct all timestamps in the scene")
    parser.add_argument("--max_timestamps", type=int, default=None,
                        help="Maximum number of timestamps to process")

    # Listing options
    parser.add_argument("--list_scenes", action="store_true",
                        help="List available scenes and exit")
    parser.add_argument("--list_timestamps", action="store_true",
                        help="List timestamps in a scene and exit")

    # Model options
    parser.add_argument("--model", type=str, default="da3-giant",
                        help="DA3 model name (default: da3-giant)")
    parser.add_argument("--process_res", type=int, default=518,
                        help="Processing resolution (default: 518)")

    # LiDAR options
    parser.add_argument("--use_individual_lidars", action="store_true",
                        help="Use individual LiDARs instead of merged point cloud")

    # Output options
    parser.add_argument("--output_dir", type=str, default="./output_car_road",
                        help="Output directory")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable visualization output")

    args = parser.parse_args()

    # Initialize dataset loader
    print("Initializing dataset loader...")
    loader = CarRoadDatasetLoader(
        data_root=args.data_root,
        calib_path=args.calib_path,
        use_fisheye=False,  # Start with pinhole only
    )

    # Handle listing options
    if args.list_scenes:
        scenes = loader.list_scenes()
        print(f"\nAvailable scenes ({len(scenes)}):")
        for scene in scenes:
            print(f"  {scene}")
        return

    if args.list_timestamps:
        if not args.scene:
            parser.error("--scene is required with --list_timestamps")

        timestamps = loader.get_timestamps(args.scene)
        print(f"\nTimestamps in {args.scene} ({len(timestamps)}):")
        for ts in timestamps[:20]:  # Show first 20
            print(f"  {ts}")
        if len(timestamps) > 20:
            print(f"  ... and {len(timestamps) - 20} more")
        return

    # Validate scene selection
    if not args.scene:
        parser.error("--scene is required for reconstruction")

    if not args.timestamp and not args.all_timestamps:
        parser.error("Either --timestamp or --all_timestamps is required")

    # Initialize reconstructor
    print(f"\nInitializing DA3 reconstructor (model: {args.model})...")
    reconstructor = RoadsideReconstructor(
        model_name=args.model,
        process_res=args.process_res,
    )

    # Get timestamps to process
    if args.all_timestamps:
        timestamps = loader.get_timestamps(args.scene)
        if args.max_timestamps:
            timestamps = timestamps[:args.max_timestamps]
        print(f"\nProcessing {len(timestamps)} timestamps...")
    else:
        timestamps = [args.timestamp]

    # Process each timestamp
    results = []
    for i, ts in enumerate(timestamps):
        print(f"\n[{i+1}/{len(timestamps)}] Processing timestamp {ts}")

        try:
            result = reconstruct_scene_timestamp(
                loader=loader,
                reconstructor=reconstructor,
                scene_name=args.scene,
                timestamp=ts,
                output_dir=args.output_dir,
                use_merged_lidar=not args.use_individual_lidars,
                visualize=not args.no_visualize,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {ts}: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete: {len(results)}/{len(timestamps)} successful")
        print(f"{'='*60}")

        total_points = sum(r.get("total_points", 0) for r in results)
        print(f"Total reconstructed points: {total_points:,}")


if __name__ == "__main__":
    main()
