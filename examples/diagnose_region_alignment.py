#!/usr/bin/env python3
"""
Diagnose Region-wise Depth Alignment.

New approach:
    1. DA3 → relative depth (good structure)
    2. SAM "Segment Everything" → all regions (roads, vehicles, poles, etc.)
    3. LiDAR → sparse anchor points
    4. Per-region: compute ax+b using anchors
    5. SAM boundaries = hard constraints

This is better than global scaling because:
    - Different regions may have different scale/shift
    - Preserves DA3's structural accuracy
    - Uses LiDAR anchors for local metric alignment
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


def main():
    parser = argparse.ArgumentParser(description="Diagnose region-wise depth alignment")

    # Car-road dataset mode
    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_TianJin")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--camera_id", type=str, default="0")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="da3-giant")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint (auto-detect if not specified)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Region-wise Depth Alignment Diagnosis")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    loader = CarRoadDatasetLoader(data_root=args.data_root)

    cam = loader.cameras[args.camera_id]
    intrinsics = cam.intrinsics
    extrinsics = cam.extrinsics_w2c

    image, image_path = loader.load_image(args.scene, args.camera_id, args.timestamp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  Image: {image_path}")

    lidar_points = loader.load_merged_points(args.scene, args.timestamp)
    print(f"  LiDAR: {len(lidar_points):,} points")

    H, W = image.shape[:2]
    Image.fromarray(image).save(output_dir / "rgb.png")

    # DA3 inference
    print("\n[2/5] DA3 inference...")
    model = DepthAnything3.from_pretrained(
        {"da3-giant": "depth-anything/DA3-GIANT",
         "da3-large": "depth-anything/DA3-LARGE"}.get(args.model, args.model)
    )
    model.to(args.device)
    model.eval()

    prediction = model.inference(image=[image], process_res=518)
    relative_depth = prediction.depth[0]
    confidence = prediction.conf[0]

    proc_H, proc_W = relative_depth.shape
    print(f"  Output: {proc_W}x{proc_H}")
    print(f"  Depth range (relative): {relative_depth.min():.2f} - {relative_depth.max():.2f}")

    # Scale intrinsics
    scale_x = proc_W / W
    scale_y = proc_H / H
    K_scaled = intrinsics.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y

    # Save relative depth
    rel_depth_color = colorize_depth(relative_depth)
    Image.fromarray(rel_depth_color).save(output_dir / "depth_relative.png")

    # Project LiDAR to get sparse depth
    print("\n[3/5] LiDAR projection...")
    uv, depths, _ = project_lidar_to_image(
        lidar_points, K_scaled, extrinsics, (proc_H, proc_W)
    )
    lidar_depth, lidar_mask = create_sparse_depth_map(uv, depths, (proc_H, proc_W))
    print(f"  Anchors: {lidar_mask.sum():,} pixels")

    # Save LiDAR depth visualization
    lidar_vis = colorize_depth(lidar_depth, vmin=0, vmax=150)
    Image.fromarray(lidar_vis).save(output_dir / "lidar_sparse.png")

    # Region-wise alignment
    print("\n[4/5] Region-wise alignment...")
    image_resized = cv2.resize(image, (proc_W, proc_H))

    aligner = RegionWiseDepthAligner(
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        min_anchors_per_region=3,
        sam_points_per_side=32,
    )

    if not aligner.sam_available:
        print("  WARNING: SAM not available!")
        print("  Install with: pip install segment-anything")
        print("  Download checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return

    result = aligner.align(
        image=image_resized,
        relative_depth=relative_depth,
        lidar_points=lidar_points,
        intrinsics=K_scaled,
        extrinsics=extrinsics,
    )

    # Save results
    print("\n[5/5] Saving results...")

    # Aligned depth
    aligned_color = colorize_depth(result.aligned_depth, vmin=0, vmax=150)
    Image.fromarray(aligned_color).save(output_dir / "depth_aligned.png")

    # Region visualization
    region_vis = (plt.cm.tab20(result.region_masks % 20)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(region_vis).save(output_dir / "sam_regions.png")

    # Per-region scale map
    scale_map = np.ones_like(relative_depth)
    for region_id, (scale, shift) in result.region_params.items():
        mask = result.region_masks == region_id
        scale_map[mask] = scale

    scale_norm = (scale_map - 0.5) / 1.5  # Normalize around 1.0
    scale_color = (plt.cm.coolwarm(scale_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(scale_color).save(output_dir / "scale_map.png")

    # Summary figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(rel_depth_color)
    axes[0, 1].set_title("DA3 Relative Depth")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(lidar_vis)
    axes[0, 2].set_title(f"LiDAR Anchors ({lidar_mask.sum():,})")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(aligned_color)
    axes[0, 3].set_title("Aligned Metric Depth")
    axes[0, 3].axis('off')

    axes[1, 0].imshow(region_vis)
    axes[1, 0].set_title(f"SAM Regions ({result.num_regions})")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(scale_color)
    axes[1, 1].set_title("Per-Region Scale (blue<1, red>1)")
    axes[1, 1].axis('off')

    # Overlay: regions on image
    overlay = image_resized.copy().astype(float)
    overlay = overlay * 0.6 + region_vis.astype(float) * 0.4
    axes[1, 2].imshow(overlay.astype(np.uint8))
    axes[1, 2].set_title("Regions Overlay")
    axes[1, 2].axis('off')

    # Confidence
    conf_color = colorize_depth(confidence, cmap='viridis')
    axes[1, 3].imshow(conf_color)
    axes[1, 3].set_title("DA3 Confidence")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_summary.png", dpi=150)
    plt.close()

    # Print region statistics
    print(f"\n{'='*60}")
    print("Region Statistics:")
    print(f"{'='*60}")

    sorted_regions = sorted(
        result.region_stats.items(),
        key=lambda x: x[1].get('n_anchors', 0),
        reverse=True
    )

    for region_id, stats in sorted_regions[:20]:  # Top 20
        scale, shift = result.region_params[region_id]
        n_anchors = stats.get('n_anchors', 0)
        rmse = stats.get('rmse', 'N/A')
        if isinstance(rmse, float):
            print(f"  Region {region_id:3d}: scale={scale:.3f}, shift={shift:+.2f}, anchors={n_anchors:4d}, RMSE={rmse:.3f}m")
        else:
            print(f"  Region {region_id:3d}: scale={scale:.3f}, shift={shift:+.2f}, anchors={n_anchors:4d} (fallback)")

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nKey files:")
    print(f"  - diagnosis_summary.png: 8-panel overview")
    print(f"  - depth_relative.png: DA3 raw relative depth")
    print(f"  - depth_aligned.png: Final metric depth (region-wise)")
    print(f"  - sam_regions.png: SAM segmentation")
    print(f"  - scale_map.png: Per-region scale factors")
    print(f"  - lidar_sparse.png: LiDAR anchor points")


if __name__ == "__main__":
    main()
