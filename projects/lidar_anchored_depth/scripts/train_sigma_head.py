"""Train the per-pixel sigma-uncertainty MLP — Stage 4A.

The static branch of the scene reconstruction uses a per-camera affine
``z = a * d_tilde + b`` calibrated against static LiDAR. The residual
``r = z_lidar - (a * d_tilde + b)`` is heteroscedastic — typical RMSE
on scene 008 is ~3 m per camera, but most of that is concentrated in a
few image regions (depth discontinuities, the periphery, reflections).
This script trains a tiny MLP to predict ``log sigma`` per pixel under
heteroscedastic Gaussian NLL, so that downstream multi-camera voxel
fusion can inverse-variance-weight the camera votes instead of mean-
fusing them blindly.

Training pairs come from the SAME (camera, ts) frames used for the
static-branch (a, b) calibration:

  * For each ts, project the STATIC portion of the LiDAR cloud onto
    the camera plane.
  * Drop pixels covered by the dynamic SAM mask (so we don't learn on
    moving-object pixels).
  * For every surviving pixel ``(u, v)``, record:
      X = features_for_pixels(uv, d_tilde[v, u], K, a_cam, b_cam, ...)
      y_residual = z_lidar - (a_cam * d_tilde + b_cam)
  * Pool over all ts and all cameras into one big (N, 8) / (N,) dataset.

Then train SigmaHead with Adam + heteroscedastic NLL. The checkpoint is
saved as a torch ``.pt`` file containing state_dict + (a, b) per camera
+ feature metadata; ``run_static_scene_recon.py --sigma-checkpoint``
loads it and uses ``predict_sigma`` at fusion time.

Usage
-----
    PYTHONPATH=src python scripts/train_sigma_head.py \\
        --data-root /mnt/car_road_data_TianJin --scene 008 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --calib-json preview/scene_recon/008_static_calib.json \\
        --output preview/scene_recon/008_sigma_head.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import points_in_oriented_bbox
from lidar_anchored_depth.alignment.projection import world_to_image
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.models import (
    SIGMA_HEAD_FEATURE_DIM,
    SigmaHead,
    features_for_pixels,
    gaussian_nll_loss,
)
from lidar_anchored_depth.segmentation.sam_io import load_sam_dynamic_mask


def _filter_finite_in_image(uv, z, hw):
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z = z[finite]
    if uv.size == 0:
        return uv, z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    uv_int = np.round(uv).astype(np.int64)
    H, W = hw
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    return uv[inside], z[inside], uv_int[inside, 0], uv_int[inside, 1]


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _collect_training_pairs(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_id: str,
    a: float,
    b: float,
    d_paths_index: dict,
    sam_dir: Path | None,
    *,
    z_min: float,
    z_max: float,
    max_per_frame: int,
    rng: np.random.Generator,
    sam_dilate_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel ``X`` (N, 8) and residual ``y`` (N,)."""
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for ts_ms in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            continue
        d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
        if d_path is None or not d_path.is_file():
            continue
        d_data = np.load(d_path, allow_pickle=True)
        d_image = d_data["depth"]

        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]
        if d_image.shape != (H, W):
            continue

        is_dynamic = np.zeros(frame.lidar_world.shape[0], dtype=bool)
        for obj in frame.dynamic_objects or []:
            is_dynamic |= points_in_oriented_bbox(
                frame.lidar_world, obj, expand=0.10,
            )
        static_lidar = frame.lidar_world[~is_dynamic]
        if static_lidar.shape[0] < 100:
            continue

        dist = frame.meta.get("distortion")
        uv, z_cam, _ = world_to_image(static_lidar, frame.K, frame.T_wc, dist)
        uv, z_cam, u, v = _filter_finite_in_image(uv, z_cam, (H, W))
        if u.size < 50:
            continue

        dyn_mask = load_sam_dynamic_mask(
            sam_dir, scene_id, ts_ms, cam_id, (H, W),
            dilate_px=sam_dilate_px,
        )
        keep = ~dyn_mask[v, u]
        u = u[keep]
        v = v[keep]
        z_cam = z_cam[keep]
        uv = uv[keep]
        if u.size < 50:
            continue

        d_at = d_image[v, u].astype(np.float64)
        valid = (
            np.isfinite(d_at) & (d_at > 0)
            & (z_cam >= z_min) & (z_cam <= z_max)
        )
        if int(valid.sum()) < 50:
            continue
        u_v = uv[valid]
        d_v = d_at[valid]
        z_v = z_cam[valid].astype(np.float64)

        # Per-frame subsample to keep the dataset balanced across ts
        n_keep = min(int(u_v.shape[0]), max_per_frame)
        if u_v.shape[0] > n_keep:
            sel = rng.choice(u_v.shape[0], size=n_keep, replace=False)
            u_v = u_v[sel]
            d_v = d_v[sel]
            z_v = z_v[sel]

        feats = features_for_pixels(
            u_v, d_v, frame.K, a=a, b=b, image_hw=(H, W),
        )
        residual = z_v - (a * d_v + b)
        X_chunks.append(feats)
        y_chunks.append(residual.astype(np.float32))

    if not X_chunks:
        return np.zeros((0, SIGMA_HEAD_FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(X_chunks), np.concatenate(y_chunks)


def _train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    device: str,
    log_every: int = 50,
):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    Xtr, ytr = Xt[tr_idx], yt[tr_idx]
    Xva, yva = Xt[val_idx], yt[val_idx]

    # Standardise inputs (saved into the checkpoint and re-applied at
    # inference time).
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp(min=1e-6)
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std

    head = SigmaHead(in_dim=SIGMA_HEAD_FEATURE_DIM).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    n_tr = Xtr_n.shape[0]
    fmt = "  epoch {:>3}/{:<3}  step {:>5}/{:<5}  train_nll={:>7.4f}  val_nll={:>7.4f}"
    history = []
    step = 0
    for ep in range(epochs):
        perm_e = torch.randperm(n_tr, generator=g).to(device)
        for s in range(0, n_tr, batch_size):
            sel = perm_e[s:s + batch_size]
            xb = Xtr_n[sel]
            rb = ytr[sel]
            log_sigma = head(xb)
            loss = gaussian_nll_loss(log_sigma, rb, reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                head.eval()
                with torch.no_grad():
                    log_sigma_v = head(Xva_n)
                    val_loss = gaussian_nll_loss(log_sigma_v, yva, reduction="mean").item()
                head.train()
                print(fmt.format(
                    ep + 1, epochs, step, (n_tr + batch_size - 1) // batch_size * epochs,
                    float(loss.item()), val_loss,
                ))
                history.append({
                    "step": int(step),
                    "epoch": int(ep + 1),
                    "train_nll": float(loss.item()),
                    "val_nll": float(val_loss),
                })

    head.eval()
    with torch.no_grad():
        log_sigma_v = head(Xva_n).clamp(min=-5.0, max=5.0)
        sigma_v = log_sigma_v.exp().cpu().numpy()
        r_v = yva.cpu().numpy()
    final_val_nll = float(0.5 * np.mean((r_v / sigma_v) ** 2 + 2.0 * np.log(sigma_v)))
    print()
    print(f"[final]  val NLL = {final_val_nll:.4f}")
    print(f"         val |r| mean = {float(np.mean(np.abs(r_v))):.3f} m")
    print(f"         val sigma mean = {float(np.mean(sigma_v)):.3f} m   "
          f"median = {float(np.median(sigma_v)):.3f} m")
    print(f"         val |r|/sigma p50 = {float(np.median(np.abs(r_v) / sigma_v)):.2f}  "
          f"p90 = {float(np.quantile(np.abs(r_v) / sigma_v, 0.9)):.2f}  "
          f"(well-calibrated -> p50 ~ 0.67, p90 ~ 1.64 for Gaussian)")

    return head, mean, std, history, final_val_nll


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--d-paths-glob", required=True)
    parser.add_argument("--sam-mask-dir", default=None)
    parser.add_argument(
        "--calib-json", required=True,
        help="JSON written by run_static_scene_recon.py with per-camera "
        "static (a, b). Format: {cam_id: {a, b, ...}}.",
    )
    parser.add_argument("--output", required=True, help="output .pt path")
    parser.add_argument("--cams", nargs="+", default=["0", "3", "6", "9"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-per-frame", type=int, default=4000)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loader-min-points", type=int, default=0)
    parser.add_argument(
        "--sam-dilate-px", type=int, default=0,
        help="match the dilation used at recon time (so the head sees "
        "training pixels with the same static/dynamic split it will "
        "see at inference). Default 0.",
    )
    args = parser.parse_args()

    import torch  # imported here so the script fails fast if torch is missing

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"[warn] cuda not available; falling back to cpu")
        args.device = "cpu"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calib = json.loads(Path(args.calib_json).read_text())
    print(f"[load] static calib for cams: {sorted(calib.keys())}")

    d_paths_index: dict[tuple[str, int, str], Path] = {}
    for p in sorted(glob.glob(args.d_paths_glob)):
        try:
            data = np.load(p, allow_pickle=True)
        except Exception:
            continue
        try:
            key = (str(data["scene_id"]), int(data["ts_ms"]), str(data["cam_id"]))
        except KeyError:
            continue
        d_paths_index[key] = Path(p)
    print(f"[load] {len(d_paths_index)} d_tilde frames indexed")

    sam_dir = Path(args.sam_mask_dir) if args.sam_mask_dir else None

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    print(f"[scene] {scene_id}")
    print()

    rng = np.random.default_rng(args.seed)
    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    print(f"[1/2] collecting training pairs (max {args.max_per_frame} per frame)")
    print("  cam   a       b       n_pairs   resid_rmse")
    for cam_id in args.cams:
        if cam_id not in calib:
            print(f"  cam{cam_id}: missing from calib JSON, skipping")
            continue
        a = float(calib[cam_id]["a"])
        b = float(calib[cam_id]["b"])
        t0 = time.time()
        Xc, yc = _collect_training_pairs(
            loader, scene_id, cam_id, a, b, d_paths_index, sam_dir,
            z_min=args.z_min, z_max=args.z_max,
            max_per_frame=args.max_per_frame, rng=rng,
            sam_dilate_px=args.sam_dilate_px,
        )
        if Xc.shape[0] == 0:
            print(f"  cam{cam_id}: no pairs collected")
            continue
        rmse = float(np.sqrt(np.mean(yc ** 2)))
        print(f"  cam{cam_id}  {a:>6.3f}  {b:>6.3f}  {Xc.shape[0]:>7}  {rmse:>7.3f}m   ({time.time() - t0:.1f}s)")
        X_all.append(Xc)
        y_all.append(yc)

    if not X_all:
        raise SystemExit("no training pairs collected")
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    print(f"\n  total pairs: {X.shape[0]}  baseline RMSE: {float(np.sqrt(np.mean(y ** 2))):.3f} m\n")

    print(f"[2/2] training SigmaHead on {args.device}")
    head, mean, std, history, final_val_nll = _train(
        X, y,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_frac=args.val_frac, seed=args.seed, device=args.device,
    )

    ckpt = {
        "state_dict": head.state_dict(),
        "in_dim": SIGMA_HEAD_FEATURE_DIM,
        "feature_mean": mean.cpu().numpy(),
        "feature_std": std.cpu().numpy(),
        "calib": calib,
        "scene_id": scene_id,
        "history": history,
        "final_val_nll": final_val_nll,
        "args": {k: v for k, v in vars(args).items() if k != "device"},
    }
    torch.save(ckpt, out_path)
    print()
    print(f"  ↳ checkpoint: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
