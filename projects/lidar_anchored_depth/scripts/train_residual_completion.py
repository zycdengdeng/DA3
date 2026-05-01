"""Train the residual depth-completion U-Net via Conditional Flow Matching.

Usage
-----
    PYTHONPATH=src python scripts/train_residual_completion.py \\
        --train-dir preview/completion_data/ \\
        --val-dir   preview/completion_data_val/ \\
        --output    preview/completion_ckpt/ \\
        --epochs 200 --batch-size 16 --lr 1e-4

After ``prepare_completion_data.py`` writes per-frame samples, this
trains :class:`ResidualVelocityUNet` on the rectified-flow loss. Loss
is weighted: 1.0 at LiDAR pixels (true supervision), 0.05 elsewhere
(soft prior toward Δz = 0, i.e. trust AA-HAD).

Outputs:
    <output>/best.pt         lowest-val checkpoint
    <output>/last.pt         last-epoch checkpoint
    <output>/history.json    per-epoch train/val NLL + step count
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.data.completion_dataset import (
    CompletionDataset,
    DZ_NORM,
)
from lidar_anchored_depth.models import (
    COND_CHANNELS,
    RectifiedFlowMatcher,
    ResidualVelocityUNet,
    masked_cfm_loss,
)


def _train_one_epoch(
    net,
    matcher,
    loader,
    optimizer,
    device,
    *,
    valid_weight: float,
    invalid_weight: float,
) -> dict:
    import torch

    net.train()
    total = 0.0
    n_steps = 0
    for batch in loader:
        cond = batch["cond"].to(device, non_blocking=True)
        target_dz = batch["target_dz"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        fm_batch = matcher.sample_train_pair(target_dz)
        v_pred = net(fm_batch.x_t, fm_batch.t, cond)
        loss = masked_cfm_loss(
            v_pred, fm_batch.u_target, valid,
            valid_weight=valid_weight, invalid_weight=invalid_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
        n_steps += 1
    return {"train_loss": total / max(n_steps, 1), "steps": n_steps}


def _eval_one_epoch(
    net,
    matcher,
    loader,
    device,
    *,
    valid_weight: float,
    invalid_weight: float,
    n_sample_steps: int,
) -> dict:
    import torch

    net.eval()
    total_cfm = 0.0
    rmse_sum = 0.0
    rmse_count = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            cond = batch["cond"].to(device, non_blocking=True)
            target_dz = batch["target_dz"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            fm_batch = matcher.sample_train_pair(target_dz)
            v_pred = net(fm_batch.x_t, fm_batch.t, cond)
            loss = masked_cfm_loss(
                v_pred, fm_batch.u_target, valid,
                valid_weight=valid_weight, invalid_weight=invalid_weight,
            )
            total_cfm += float(loss.item())
            n += 1

            # Also: sample Δz with N-step rectified Euler and report
            # RMSE in metres at LiDAR pixels (de-normalising the Δz
            # by ``DZ_NORM`` and comparing to the un-normalised
            # target).
            x_pred = matcher.sample(
                net, cond,
                n_steps=n_sample_steps,
                target_shape=target_dz.shape,
            )
            err = (x_pred - target_dz) * DZ_NORM
            mask = valid.bool()
            if mask.any():
                err_flat = err[mask]
                rmse_sum += float((err_flat ** 2).sum().item())
                rmse_count += int(err_flat.numel())
    val_loss = total_cfm / max(n, 1)
    val_rmse = (rmse_sum / max(rmse_count, 1)) ** 0.5
    return {"val_loss": val_loss, "val_rmse_m": val_rmse}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--valid-weight", type=float, default=1.0,
        help="loss weight at LiDAR pixels (default 1.0)",
    )
    parser.add_argument(
        "--invalid-weight", type=float, default=0.05,
        help="loss weight outside LiDAR pixels (soft AA-HAD prior; default 0.05)",
    )
    parser.add_argument(
        "--sample-steps", type=int, default=4,
        help="Euler steps for validation Δz sampling (default 4)",
    )
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

    train_ds = CompletionDataset(
        args.train_dir, crop_size=args.crop_size, random_flip=True,
    )
    val_ds = (
        CompletionDataset(args.val_dir, crop_size=args.crop_size, random_flip=False)
        if args.val_dir else None
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, args.num_workers // 2), pin_memory=True,
        ) if val_ds else None
    )

    net = ResidualVelocityUNet(
        cond_channels=COND_CHANNELS, base=args.base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[net] ResidualVelocityUNet  params={n_params / 1e6:.2f}M")

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
                valid_weight=args.valid_weight,
                invalid_weight=args.invalid_weight,
                n_sample_steps=args.sample_steps,
            ) if val_loader else {}
        )
        rec = {
            "epoch": epoch + 1,
            "train_loss": tr["train_loss"],
            "epoch_time_s": elapsed,
            **ev,
        }
        history.append(rec)
        print(
            f"  epoch {epoch + 1:>3}/{args.epochs}  "
            f"train={tr['train_loss']:.4f}  "
            + (f"val={ev['val_loss']:.4f}  rmse={ev['val_rmse_m']:.2f}m  "
               if val_loader else "")
            + f"({elapsed:.1f}s)"
        )

        torch.save(
            {"state_dict": net.state_dict(),
             "args": vars(args), "epoch": epoch + 1},
            out_dir / "last.pt",
        )
        if val_loader and ev["val_loss"] < best_val:
            best_val = ev["val_loss"]
            torch.save(
                {"state_dict": net.state_dict(),
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
