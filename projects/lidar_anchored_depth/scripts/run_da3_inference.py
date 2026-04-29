"""Run DA3 inference on one or more THICV-R2A frames and save ``d̃.npz``.

This decouples the heavy DA3-inference step (needs torch + GPU + a few-GB
checkpoint) from the lightweight diagnostic / fitting step
(:mod:`scripts.depth_model_diagnostic`). After this script writes a
``*_d.npz`` per (scene, ts, cam), the diagnostic can be re-run as many
times as needed without re-running the model.

Output (``preview/da3/<scene_id>_ts<ts_ms>_cam<id>_d.npz``):
    depth        : (H, W) float32  — DA3 relative depth, resized to the
                   ORIGINAL image size so it aligns with our calibration.
    conf         : (H, W) float32 or absent if the model did not
                   produce confidence.
    proc_h, proc_w : ints         — DA3 processing resolution
    image_hw     : (2,) int       — original image size
    model        : str            — HF model id used
    scene_id     : str
    ts_ms        : int
    cam_id       : str

Usage
-----
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/run_da3_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --cam all \\
        --output preview/da3/

    # specific timestamp + single cam
    python scripts/run_da3_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --timestamp 1742879642908 --cam 3 \\
        --output preview/da3/

    # different model variant
    python scripts/run_da3_inference.py ... --model depth-anything/DA3-Large
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lidar_anchored_depth.data import (
    PINHOLE_CAMERA_IDS,
    RoadsideV2XLoader,
)


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    """Accept either a full scene name or a numeric prefix like ``008``."""
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(
            f"no scene matched {scene_arg!r} in {loader.data_root}"
        )
    return cands[0].scene_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--scene", required=True,
        help="full scene name or numeric prefix (e.g. '008')",
    )
    parser.add_argument(
        "--timestamp", default=None, type=str,
        help="ms timestamp; default = first available in the scene",
    )
    parser.add_argument(
        "--cam", default="3",
        help="pinhole cam id (0/3/6/9) or 'all' for all 4",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model", default="depth-anything/DA3Mono-Large",
        help=(
            "HuggingFace model id. Defaults to DA3Mono-Large (relative "
            "monocular depth). Try DA3-Giant or DA3NESTED-GIANT-LARGE "
            "for the strongest model; DA3Metric-Large for a metric head."
        ),
    )
    parser.add_argument(
        "--process-res", type=int, default=504,
        help="DA3 internal processing resolution (default: 504)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device. Default: cuda",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports here so --help is fast.
    import torch
    from depth_anything_3.api import DepthAnything3

    print(f"[load] {args.model}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=args.device).eval()
    print(f"  ↳ loaded in {time.time() - t0:.1f}s on {args.device}")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    ts_ms = int(args.timestamp) if args.timestamp else scene.timestamps_ms[0]

    cams: list[str]
    if args.cam == "all":
        cams = list(PINHOLE_CAMERA_IDS)
    else:
        if args.cam not in PINHOLE_CAMERA_IDS:
            raise SystemExit(
                f"--cam={args.cam!r} not in pinhole set {PINHOLE_CAMERA_IDS}"
            )
        cams = [args.cam]

    print(f"[scene] {scene_id}  ts={ts_ms}  cams={cams}")
    print()

    for cam_id in cams:
        idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        frame = loader.get_frame(idx)
        H_orig, W_orig = frame.image.shape[:2]

        t0 = time.time()
        with torch.inference_mode():
            prediction = model.inference(
                image=[frame.image],
                process_res=args.process_res,
            )
        t_inf = time.time() - t0

        depth_proc = prediction.depth[0]  # (h_p, w_p), float32
        h_p, w_p = depth_proc.shape
        # Resize DA3 depth back to the original image dims so it aligns
        # with our calibration (K, T_wc are at ORIGINAL resolution).
        depth_full = cv2.resize(
            depth_proc.astype(np.float32),
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR,
        )

        conf_full = None
        if prediction.conf is not None:
            conf_full = cv2.resize(
                prediction.conf[0].astype(np.float32),
                (W_orig, H_orig),
                interpolation=cv2.INTER_LINEAR,
            )

        out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_d.npz"
        save_kwargs = dict(
            depth=depth_full,
            proc_h=h_p, proc_w=w_p,
            image_hw=np.array([H_orig, W_orig], dtype=np.int32),
            model=args.model,
            scene_id=scene_id,
            ts_ms=int(ts_ms),
            cam_id=cam_id,
        )
        if conf_full is not None:
            save_kwargs["conf"] = conf_full
        np.savez_compressed(out_path, **save_kwargs)

        d_min = float(np.nanmin(depth_full))
        d_max = float(np.nanmax(depth_full))
        d_med = float(np.nanmedian(depth_full))
        print(
            f"  cam{cam_id}  ->  {out_path.name}    "
            f"d̃ range [{d_min:.3f}, {d_max:.3f}] median {d_med:.3f}    "
            f"DA3 inference {t_inf*1000:.0f} ms"
        )

    print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
