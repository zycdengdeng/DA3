"""
Batch monocular depth estimation for SparseGS scenes using DA3.

Usage:
    python batch_depth_inference.py \
        --data_root /mnt/zyc_wzh/SparseGS/data/car_road \
        --model_name da3-large \
        --process_res 504
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def get_scene_dirs(data_root):
    """Get all scene directories that contain an images/ subfolder."""
    scene_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d) and os.path.isdir(os.path.join(d, "images"))]
    return scene_dirs


def get_image_paths(images_dir):
    """Get all image paths in a directory."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff", "*.tif")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Batch DA3 depth estimation for SparseGS")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing scene folders")
    parser.add_argument("--model_name", type=str, default="da3-large",
                        help="DA3 model name (default: da3-large for relative depth)")
    parser.add_argument("--process_res", type=int, default=504,
                        help="Processing resolution (default: 504)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Specific scene names to process (default: all)")
    args = parser.parse_args()

    # Load model once
    print(f"Loading DA3 model: {args.model_name} ...")
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name.upper()}")
    model = model.to(args.device)
    model.eval()
    print("Model loaded.")

    # Get scene directories
    scene_dirs = get_scene_dirs(args.data_root)
    if args.scenes:
        scene_dirs = [d for d in scene_dirs if os.path.basename(d) in args.scenes]

    print(f"Found {len(scene_dirs)} scenes to process.")

    total_images = 0
    total_time = 0.0

    for i, scene_dir in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_dir)
        images_dir = os.path.join(scene_dir, "images")
        depths_dir = os.path.join(scene_dir, "depths")

        image_paths = get_image_paths(images_dir)
        if not image_paths:
            print(f"[{i+1}/{len(scene_dirs)}] {scene_name}: No images found, skipping.")
            continue

        os.makedirs(depths_dir, exist_ok=True)

        print(f"[{i+1}/{len(scene_dirs)}] {scene_name}: Processing {len(image_paths)} images ...")
        t0 = time.time()

        # Run inference on all images in this scene at once
        prediction = model.inference(
            image=image_paths,
            process_res=args.process_res,
        )

        # prediction.depth shape: (N, H, W) at processing resolution
        # We need to resize depth back to original image resolution
        for j, img_path in enumerate(image_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            depth = prediction.depth[j]  # (H_proc, W_proc)

            # Read original image to get its resolution
            orig_img = Image.open(img_path)
            orig_w, orig_h = orig_img.size

            # Resize depth to original resolution if needed
            if depth.shape[0] != orig_h or depth.shape[1] != orig_w:
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Save as .npy file
            depth_path = os.path.join(depths_dir, f"{img_name}.npy")
            np.save(depth_path, depth.astype(np.float32))

        elapsed = time.time() - t0
        total_images += len(image_paths)
        total_time += elapsed
        print(f"    Done in {elapsed:.1f}s, saved {len(image_paths)} depth files to {depths_dir}")

    print(f"\nAll done! Processed {total_images} images across {len(scene_dirs)} scenes in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
