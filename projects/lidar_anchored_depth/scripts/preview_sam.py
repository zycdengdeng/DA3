"""Visualize SAM masks (bbox-prompted, dynamic-only) overlaid on an image.

For one ``(scene, ts, cam)`` frame:

* loads ``preview/sam/<scene>_ts<ts>_cam<cam>_sam.npz`` (output of
  ``run_sam_inference.py``),
* colourises each mask with a deterministic random colour keyed by
  the V2X bbox id,
* alpha-blends the union onto the source RGB image,
* saves a PNG to ``--output``.

Usage
-----
    PYTHONPATH=src python scripts/preview_sam.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --ts 1742879650704 --cam 3 \\
        --sam-mask-dir preview/sam/ \\
        --output preview/sam_viz_cam3.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lidar_anchored_depth.data import RoadsideV2XLoader


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _color_for_id(mid: int) -> tuple[int, int, int]:
    """Deterministic colour per id; uses a fixed-seed RNG indexed by id."""
    rng = np.random.default_rng(int(mid) * 2654435761 % (2**32))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    ids: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """``masks`` is ``(K, H, W)`` bool, ``ids`` ``(K,)`` int.

    Returns an ``(H, W, 3)`` uint8 image with masks blended in.
    """
    H, W = image.shape[:2]
    out = image.astype(np.float32).copy()
    for k in range(masks.shape[0]):
        col = np.array(_color_for_id(int(ids[k])), dtype=np.float32)
        m = masks[k]
        out[m] = (1.0 - alpha) * out[m] + alpha * col[None]
    return out.clip(0, 255).astype(np.uint8)


def _draw_id_labels(
    pil_img: Image.Image,
    masks: np.ndarray,
    ids: np.ndarray,
) -> None:
    """Annotate the centroid of each mask with its V2X bbox id."""
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for k in range(masks.shape[0]):
        ys, xs = np.where(masks[k])
        if ys.size == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        col = _color_for_id(int(ids[k]))
        # Black halo + coloured text for readability.
        text = str(int(ids[k]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((cx + dx, cy + dy), text, fill="black", font=font)
        draw.text((cx, cy), text, fill=col, font=font)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ts", required=True, type=int)
    parser.add_argument("--cam", required=True)
    parser.add_argument("--sam-mask-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--no-labels", action="store_true",
        help="skip the per-mask id label overlay",
    )
    parser.add_argument("--loader-min-points", type=int, default=0)
    args = parser.parse_args()

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)

    idx = loader.find_frame_idx(scene_id, int(args.ts), args.cam)
    frame = loader.get_frame(idx)
    image = frame.image
    H, W = image.shape[:2]

    sam_npz = (
        Path(args.sam_mask_dir)
        / f"{scene_id}_ts{int(args.ts)}_cam{args.cam}_sam.npz"
    )
    if not sam_npz.is_file():
        raise SystemExit(f"SAM file not found: {sam_npz}")

    data = np.load(sam_npz, allow_pickle=True)
    masks = data["masks"]
    ids = data["mask_ids"]
    print(f"[load] {sam_npz.name}  {masks.shape[0]} masks  image {H}x{W}")

    if masks.shape[0] == 0:
        print("[warn] zero masks — output will just be the source image")
        out_img = image
    else:
        if masks.shape[1:] != (H, W):
            raise SystemExit(
                f"mask shape {masks.shape[1:]} != image {H, W}; "
                "did the SAM run match this image resolution?"
            )
        out_img = _overlay_masks(image, masks, ids, alpha=args.alpha)

    pil_img = Image.fromarray(out_img)
    if not args.no_labels and masks.shape[0] > 0:
        _draw_id_labels(pil_img, masks, ids)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, quality=92)
    print(f"  ↳ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
