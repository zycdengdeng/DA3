"""Run a SegFormer model fine-tuned on Cityscapes and save per-pixel
semantic class ids — needed for the semantic-aware target rule
(Q2 scheme 3) in the residual point completion training.

Output (per ``(scene, ts, cam)``):

    <output>/<scene>_ts<ts>_cam<cam>_seg.npz
        class_id   : (H, W) uint8     — Cityscapes 0..18
        confidence : (H, W) float16   — per-pixel softmax max
        scene_id, ts_ms, cam_id, image_hw, model

    <output>/<scene>_ts<ts>_cam<cam>_seg.png   (optional --save-viz)

Cityscapes 19-class label set (id : name)
    0 road        1 sidewalk    2 building   3 wall       4 fence
    5 pole        6 t-light     7 t-sign     8 vegetation 9 terrain
   10 sky        11 person     12 rider     13 car       14 truck
   15 bus        16 train      17 motorcyc. 18 bicycle

Usage
-----
    HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src \\
    python scripts/run_segformer_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --all-timestamps --cam all \\
        --output preview/segformer/ \\
        --model /mnt/zyc_wzh/hf_models/segformer-b5-cityscapes \\
        --save-viz \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.data import RoadsideV2XLoader


CITYSCAPES_NAMES = (
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
)
CITYSCAPES_PALETTE = np.array([
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70, 70, 70),     # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),   # traffic light
    (220, 220, 0),    # traffic sign
    (107, 142, 35),   # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),   # sky
    (220, 20, 60),    # person
    (255, 0, 0),      # rider
    (0, 0, 142),      # car
    (0, 0, 70),       # truck
    (0, 60, 100),     # bus
    (0, 80, 100),     # train
    (0, 0, 230),      # motorcycle
    (119, 11, 32),    # bicycle
], dtype=np.uint8)


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _save_seg_viz(image: np.ndarray, class_id: np.ndarray, out_path: Path,
                  alpha: float = 0.55) -> None:
    from PIL import Image

    H, W = image.shape[:2]
    palette = CITYSCAPES_PALETTE
    colour = palette[class_id.clip(0, len(palette) - 1)]
    blend = ((1 - alpha) * image.astype(np.float32) + alpha * colour.astype(np.float32))
    Image.fromarray(blend.clip(0, 255).astype(np.uint8)).save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ts", default=None, type=int,
                        help="single ts; default = first in scene")
    parser.add_argument("--all-timestamps", action="store_true")
    parser.add_argument("--ts-shard", default=None,
                        help="k/N for parallel sharding across GPUs")
    parser.add_argument("--cam", default="3",
                        help="single cam id or 'all'")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        help="HF model id OR a local path (preferred when the network "
        "is flaky). Download once via huggingface-cli, then pass the "
        "local-dir here.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-viz", action="store_true",
        help="also save a coloured Cityscapes-palette overlay PNG",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
    )
    parser.add_argument("--loader-min-points", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    t0 = time.time()
    import torch
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model = model.to(device=args.device).eval()
    print(f"  ↳ loaded in {time.time() - t0:.1f}s on {args.device}")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)

    if args.all_timestamps:
        ts_list = list(scene.timestamps_ms)
    elif args.ts is not None:
        ts_list = [int(args.ts)]
    else:
        ts_list = [scene.timestamps_ms[0]]

    if args.ts_shard:
        try:
            shard_k_str, shard_n_str = args.ts_shard.split("/")
            shard_k, shard_n = int(shard_k_str), int(shard_n_str)
        except ValueError as e:
            raise SystemExit(f"--ts-shard must be 'k/N': {args.ts_shard!r}") from e
        ts_list = [ts for i, ts in enumerate(ts_list) if i % shard_n == shard_k]
        print(f"[shard] {shard_k}/{shard_n} -> {len(ts_list)} ts on this worker")

    cams = ["0", "3", "6", "9"] if args.cam == "all" else [args.cam]
    print(f"[scene] {scene_id}  ts_count={len(ts_list)}  cams={cams}")
    print()

    n_done = 0
    n_skipped = 0
    t_total = time.time()
    for ts_ms in ts_list:
      for cam_id in cams:
        out_npz = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_seg.npz"
        if args.skip_existing and out_npz.is_file():
            n_skipped += 1
            continue
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            print(f"  ts={ts_ms} cam{cam_id}: no frame")
            continue
        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]

        t_inf = time.time()
        inputs = processor(images=frame.image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(args.device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits  # (1, C, h, w)
            # Bilinear upsample to original resolution; argmax.
            logits_up = torch.nn.functional.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False,
            )
            probs = logits_up.softmax(dim=1)
            conf, class_id = probs.max(dim=1)
        class_id = class_id.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence = conf.squeeze(0).cpu().numpy().astype(np.float16)

        np.savez_compressed(
            out_npz,
            class_id=class_id,
            confidence=confidence,
            scene_id=scene_id,
            ts_ms=int(ts_ms),
            cam_id=cam_id,
            model=args.model,
            image_hw=np.array([H, W], dtype=np.int32),
        )
        if args.save_viz:
            _save_seg_viz(frame.image, class_id, out_npz.with_suffix(".png"))

        # Quick stats: most-frequent classes in the image
        ids, counts = np.unique(class_id, return_counts=True)
        top = ids[np.argsort(-counts)[:3]]
        top_str = ", ".join(
            f"{CITYSCAPES_NAMES[i]}({counts[ids == i][0] / class_id.size:.0%})"
            for i in top
        )
        print(
            f"  ts={ts_ms} cam{cam_id}  "
            f"{(time.time() - t_inf)*1000:.0f}ms  ->  {out_npz.name}    "
            f"top: {top_str}"
        )
        n_done += 1

    print()
    print(
        f"done.  wrote {n_done}  skipped {n_skipped}  "
        f"total {time.time() - t_total:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
