"""PyTorch Dataset for residual depth-completion training — Stage 6.

Each ``.npz`` sample (produced by ``scripts/prepare_completion_data.py``)
holds one ``(scene, ts, cam)`` frame's conditioning + supervision:

    rgb               (H, W, 3)  uint8     — raw camera image
    d_tilde           (H, W)     float32   — DA3 raw depth
    aahad_init        (H, W)     float32   — a · d̃ + b (closed-form prior)
    sparse_lidar_z    (H, W)     float32   — projected static-LiDAR z (0 = miss)
    sparse_lidar_mask (H, W)     bool      — True where LiDAR projects
    target_dz         (H, W)     float32   — z_lidar - aahad_init at LiDAR (0 else)
    valid_mask        (H, W)     bool      — same as sparse_lidar_mask (loss mask)

The Dataset:
1. Random-crops to ``crop_size`` (default 384).
2. Optional horizontal flip.
3. Returns a dict of tensors ready for the U-Net.

Normalisations:
    RGB / 255.0
    d̃ / d_norm                (default 50)
    aahad_init / z_norm        (default 100)
    sparse_lidar_z / z_norm    (default 100)
    target_dz / dz_norm        (default 5)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_OK = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    _TORCH_OK = False

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401


D_TILDE_NORM = 50.0
Z_NORM = 100.0
DZ_NORM = 5.0


class CompletionDataset(Dataset):
    """Random-crop dataset over a directory of completion ``.npz`` files."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        crop_size: int = 384,
        random_flip: bool = True,
        require_lidar_pixels: int = 50,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("CompletionDataset requires torch")
        self.npz_dir = Path(npz_dir)
        self.files: list[Path] = sorted(self.npz_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found in {self.npz_dir}")
        self.crop_size = int(crop_size)
        self.random_flip = bool(random_flip)
        self.require_lidar_pixels = int(require_lidar_pixels)

    def __len__(self) -> int:
        return len(self.files)

    def _random_crop(
        self, arrays: dict[str, np.ndarray], H: int, W: int,
    ) -> dict[str, np.ndarray]:
        cs = self.crop_size
        if H < cs or W < cs:
            # Pad-then-crop fallback: pad sparse_lidar_mask=False so loss
            # naturally ignores padded pixels.
            pad_h = max(0, cs - H)
            pad_w = max(0, cs - W)
            padded = {}
            for k, v in arrays.items():
                if v.ndim == 3:
                    padded[k] = np.pad(
                        v, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant",
                    )
                else:
                    padded[k] = np.pad(
                        v, ((0, pad_h), (0, pad_w)), mode="constant",
                    )
            arrays = padded
            H, W = arrays["d_tilde"].shape

        # Try a few crops to find one with enough LiDAR pixels.
        for _ in range(8):
            top = random.randint(0, H - cs)
            left = random.randint(0, W - cs)
            mask_crop = arrays["sparse_lidar_mask"][top:top + cs, left:left + cs]
            if int(mask_crop.sum()) >= self.require_lidar_pixels:
                break
        out = {}
        for k, v in arrays.items():
            if v.ndim == 3:
                out[k] = v[top:top + cs, left:left + cs, :]
            else:
                out[k] = v[top:top + cs, left:left + cs]
        return out

    def __getitem__(self, idx: int) -> dict[str, "torch.Tensor"]:
        path = self.files[idx]
        with np.load(path) as data:
            arrays = {
                "rgb": data["rgb"],
                "d_tilde": data["d_tilde"],
                "aahad_init": data["aahad_init"],
                "sparse_lidar_z": data["sparse_lidar_z"],
                "sparse_lidar_mask": data["sparse_lidar_mask"],
                "target_dz": data["target_dz"],
                "valid_mask": data["valid_mask"],
            }
        H, W = arrays["d_tilde"].shape
        arrays = self._random_crop(arrays, H, W)
        if self.random_flip and random.random() < 0.5:
            for k, v in arrays.items():
                arrays[k] = np.ascontiguousarray(np.flip(v, axis=1))

        rgb = arrays["rgb"].astype(np.float32) / 255.0
        d_tilde = arrays["d_tilde"].astype(np.float32) / D_TILDE_NORM
        aahad_init = arrays["aahad_init"].astype(np.float32) / Z_NORM
        sparse_z = arrays["sparse_lidar_z"].astype(np.float32) / Z_NORM
        sparse_mask = arrays["sparse_lidar_mask"].astype(np.float32)
        target_dz = arrays["target_dz"].astype(np.float32) / DZ_NORM
        valid = arrays["valid_mask"].astype(np.float32)

        # (H, W, 3) -> (3, H, W); (H, W) -> (1, H, W)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        d_tilde_t = torch.from_numpy(d_tilde).unsqueeze(0)
        aahad_t = torch.from_numpy(aahad_init).unsqueeze(0)
        sparse_z_t = torch.from_numpy(sparse_z).unsqueeze(0)
        sparse_mask_t = torch.from_numpy(sparse_mask).unsqueeze(0)
        target_dz_t = torch.from_numpy(target_dz).unsqueeze(0)
        valid_t = torch.from_numpy(valid).unsqueeze(0)

        cond = torch.cat([
            rgb_t, d_tilde_t, aahad_t, sparse_z_t, sparse_mask_t,
        ], dim=0)

        return {
            "cond": cond,
            "target_dz": target_dz_t,
            "valid": valid_t,
            "path": str(path),
        }


__all__ = [
    "CompletionDataset",
    "D_TILDE_NORM",
    "Z_NORM",
    "DZ_NORM",
]
