"""Train the per-point flow-matching velocity field — Stage 7C.

Reads ``.npz`` samples produced by ``prepare_completion_3d_data.py``,
trains :class:`PointCloudVelocityNet` under conditional rectified flow
matching to predict per-point displacement ``Δxyz``.

The 3-D loss mirrors the 2-D Stage 6 setup but on point clouds:

    x_1 = Δxyz_target / DISP_SCALE   (per point, valid where LiDAR exists)
    x_0 ~ N(0, I)
    x_t = (1 - t) x_0 + t x_1
    L = || v_θ(x_t, t | cond) - (x_1 - x_0) ||²

Loss weighted as in Stage 6: ``valid_weight=1`` at LiDAR-supervised
points, ``invalid_weight=0`` (default) elsewhere — meaning we do NOT
push the network to output zero residuals at unsupervised points,
only that it can do whatever it wants there. Inference-time gating
(distance-to-LiDAR) is applied separately by the caller.

Usage
-----
    PYTHONPATH=src python scripts/train_point_completion.py \\
        --train-dir preview/completion_3d_data/ \\
        --val-dir   preview/completion_3d_data/ \\
        --output    preview/point_completion_ckpt/ \\
        --epochs 200 --batch-size 4 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.data.completion_3d_dataset import (
    Completion3DDataset,
    DISP_SCALE,
    PRIOR_FEAT_DIM,
)
from lidar_anchored_depth.models import (
    PointCloudVelocityNet,
    RectifiedFlowMatcher,
)


def _flow_loss(
    v_pred,
    u_target,
    valid,
    *,
    valid_weight: float,
    invalid_weight: float,
):
    import torch

    diff = (v_pred - u_target).pow(2).sum(dim=-1)  # (B, N)
    if invalid_weight == 0.0:
        # Only supervise valid points.
        denom = valid.sum().clamp_min(1.0)
        return (diff * valid).sum() / denom
    w = torch.where(
        valid > 0,
        torch.full_like(diff, float(valid_weight)),
        torch.full_like(diff, float(invalid_weight)),
    )
    return (diff * w).sum() / w.sum().clamp_min(1.0)


def _per_point_rmse_m(
    v_sampled,
    target_disp_norm,
    valid,
):
    import torch

    err = (v_sampled - target_disp_norm) * DISP_SCALE
    err2 = err.pow(2).sum(dim=-1)  # (B, N)
    valid_mask = valid > 0
    if not valid_mask.any():
        return float("nan")
    return float(err2[valid_mask].mean().sqrt().item())


def _train_one_epoch(net, matcher, loader, optimizer, device, *, valid_weight, invalid_weight):
    import torch

    net.train()
    total = 0.0
    n_steps = 0
    for batch in loader:
        prior_xyz = batch["prior_xyz"].to(device, non_blocking=True)
        prior_rgb = batch["prior_rgb"].to(device, non_blocking=True)
        prior_feat = batch["prior_feat"].to(device, non_blocking=True)
        target_disp = batch["target_disp"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        fm = matcher.sample_train_pair(target_disp)
        v_pred = net(prior_xyz, prior_rgb, prior_feat, fm.x_t, fm.t)
        loss = _flow_loss(
            v_pred, fm.u_target, valid,
            valid_weight=valid_weight, invalid_weight=invalid_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
        n_steps += 1
    return {"train_loss": total / max(n_steps, 1), "steps": n_steps}


def _eval_one_epoch(net, matcher, loader, device, *, n_sample_steps):
    import torch

    net.eval()
    total = 0.0
    rmse_total = 0.0
    n_valid_total = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            prior_xyz = batch["prior_xyz"].to(device, non_blocking=True)
            prior_rgb = batch["prior_rgb"].to(device, non_blocking=True)
            prior_feat = batch["prior_feat"].to(device, non_blocking=True)
            target_disp = batch["target_disp"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            fm = matcher.sample_train_pair(target_disp)
            v_pred = net(prior_xyz, prior_rgb, prior_feat, fm.x_t, fm.t)
            loss = _flow_loss(
                v_pred, fm.u_target, valid,
                valid_weight=1.0, invalid_weight=0.0,
            )
            total += float(loss.item())
            n += 1

            # Sample Δxyz with N-step Euler integration.
            x_init = torch.randn_like(target_disp)
            x = x_init
            dt = 1.0 / n_sample_steps
            for k in range(n_sample_steps):
                t_now = torch.full(
                    (x.shape[0],), float(k * dt),
                    device=x.device, dtype=x.dtype,
                )
                v = net(prior_xyz, prior_rgb, prior_feat, x, t_now)
                x = x + dt * v
            err2 = ((x - target_disp) * DISP_SCALE).pow(2).sum(dim=-1)
            valid_mask = valid > 0
            if valid_mask.any():
                rmse_total += float(err2[valid_mask].sum().item())
                n_valid_total += int(valid_mask.sum().item())

    val_loss = total / max(n, 1)
    val_rmse = (rmse_total / max(n_valid_total, 1)) ** 0.5 if n_valid_total else float("nan")
    return {"val_loss": val_loss, "val_rmse_m": val_rmse}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n-points", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument(
        "--knn-chunk", type=int, default=2048,
        help="row-chunk size for KNN distance matrix to cap peak "
        "memory at (B * chunk * N * 4) bytes. Default 2048; set 0 "
        "to disable chunking (faster but more memory).",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="wrap the model in nn.DataParallel; splits batch across "
        "all visible CUDA devices. Effective batch = batch_size; "
        "per-GPU batch = batch_size / n_gpus. With 8 cards you'll want "
        "--batch-size 16 or 32 for good utilisation.",
    )
    parser.add_argument(
        "--valid-weight", type=float, default=1.0,
    )
    parser.add_argument(
        "--invalid-weight", type=float, default=0.0,
        help="loss weight for points without LiDAR supervision (default 0 - "
        "do NOT supervise those; the displacement at those points is "
        "free to be anything the network considers reasonable, gated "
        "at inference by distance-to-LiDAR).",
    )
    parser.add_argument(
        "--sample-steps", type=int, default=4,
        help="Euler steps for validation Δxyz sampling (default 4).",
    )
    parser.add_argument("--valid-oversample", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[warn] cuda not available; falling back to cpu")
        args.device = "cpu"
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Completion3DDataset(
        args.train_dir, n_points=args.n_points,
        valid_oversample=args.valid_oversample,
        random_flip_x=True, random_yaw=True,
    )
    val_ds = (
        Completion3DDataset(
            args.val_dir, n_points=args.n_points,
            valid_oversample=args.valid_oversample,
            random_flip_x=False, random_yaw=False,
        ) if args.val_dir else None
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, args.num_workers // 2), pin_memory=True,
        ) if val_ds else None
    )

    n_train_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    print(
        f"[loader] train batches/epoch = {n_train_batches}  "
        f"(samples={len(train_ds)}, batch_size={args.batch_size})"
    )
    if n_train_batches == 0:
        raise SystemExit(
            "train DataLoader has 0 batches; --batch-size > len(dataset)?"
        )

    knn_chunk = None if args.knn_chunk <= 0 else int(args.knn_chunk)
    net = PointCloudVelocityNet(
        prior_feat_dim=PRIOR_FEAT_DIM,
        base=args.base_channels,
        k=args.knn_k,
        knn_chunk=knn_chunk,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[net] PointCloudVelocityNet  params={n_params / 1e6:.2f}M  "
          f"base={args.base_channels}  k={args.knn_k}  N={args.n_points}  "
          f"knn_chunk={knn_chunk}")
    if args.multi_gpu and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print(f"[multi-gpu] DataParallel across {torch.cuda.device_count()} GPUs")

    matcher = RectifiedFlowMatcher(sigma=0.0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = _train_one_epoch(
            net, matcher, train_loader, optimizer, device,
            valid_weight=args.valid_weight,
            invalid_weight=args.invalid_weight,
        )
        elapsed = time.time() - t0
        ev = (
            _eval_one_epoch(
                net, matcher, val_loader, device,
                n_sample_steps=args.sample_steps,
            ) if val_loader else {}
        )
        rec = {"epoch": epoch + 1, "train_loss": tr["train_loss"],
               "epoch_time_s": elapsed, **ev}
        history.append(rec)
        print(
            f"  epoch {epoch + 1:>3}/{args.epochs}  "
            f"train={tr['train_loss']:.4f}  "
            + (f"val={ev['val_loss']:.4f}  rmse={ev['val_rmse_m']:.2f}m  "
               if val_loader else "")
            + f"({elapsed:.1f}s)"
        )

        save_module = net.module if isinstance(net, torch.nn.DataParallel) else net
        torch.save(
            {"state_dict": save_module.state_dict(),
             "args": vars(args), "epoch": epoch + 1},
            out_dir / "last.pt",
        )
        if val_loader and ev.get("val_loss", float("inf")) < best_val:
            best_val = ev["val_loss"]
            torch.save(
                {"state_dict": save_module.state_dict(),
                 "args": vars(args), "epoch": epoch + 1,
                 "val_loss": best_val,
                 "val_rmse_m": ev["val_rmse_m"]},
                out_dir / "best.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print()
    print(f"  ↳ ckpt   : {out_dir}/last.pt  best.pt")
    print(f"  ↳ history: {out_dir}/history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
