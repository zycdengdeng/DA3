"""PyTorch Dataset for 3-D residual depth-completion training — Stage 7B.

Each ``.npz`` (produced by ``scripts/prepare_completion_3d_data.py``)
holds one ``(scene, ts, cam)`` frame's prior cloud, image-sampled
RGB, per-point conditioning features, the sparse-LiDAR target cloud,
and a per-prior-point displacement target ``Δxyz`` (``target_disp``)
together with a ``valid_mask``.

The Dataset:

* Subsamples each prior cloud to ``--n-points`` (default 8192) using
  random indices, optionally keeping a higher fraction of valid
  (LiDAR-supervised) points.
* Centres XYZ around the cloud mean and scales by ``xyz_scale``
  (default 30 m) so the network sees a consistent metric range.
* Returns a dict of tensors ready for a per-point flow head.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_OK = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    _TORCH_OK = False


XYZ_SCALE = 30.0
"""Metres of half-extent used to normalise XYZ into roughly [-1, 1].

Scene 008's intersection extends ~30 m around the rig centre. Scaling
by 30 maps it to [-1, 1] which matches the network's preferred range.
"""

DISP_SCALE = 5.0
"""Metres used to normalise per-point displacement target to [-1, 1]."""

PRIOR_FEAT_DIM = 5
"""Number of conditioning feature channels emitted by
``prepare_completion_3d_data.py``:
``[source_label, d_tilde, aahad_init_z, dist_to_lidar_clipped, sparse_present]``."""


class Completion3DDataset(Dataset):
    """Random-subsample dataset for the per-point flow-matching head."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        n_points: int = 8192,
        valid_oversample: float = 1.5,
        random_flip_x: bool = True,
        random_yaw: bool = True,
        require_valid_pts: int = 50,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("Completion3DDataset requires torch")
        self.npz_dir = Path(npz_dir)
        self.files: list[Path] = sorted(self.npz_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found in {self.npz_dir}")
        self.n_points = int(n_points)
        self.valid_oversample = float(valid_oversample)
        self.random_flip_x = bool(random_flip_x)
        self.random_yaw = bool(random_yaw)
        self.require_valid_pts = int(require_valid_pts)

    def __len__(self) -> int:
        return len(self.files)

    def _subsample(
        self, n_total: int, valid: np.ndarray,
    ) -> np.ndarray:
        """Pick ``self.n_points`` indices, oversampling valid points
        by a factor of ``valid_oversample``."""
        if n_total <= self.n_points:
            # Pad with random repeats to fixed size.
            base = np.arange(n_total)
            extra = np.random.choice(n_total, size=self.n_points - n_total, replace=True)
            return np.concatenate([base, extra])

        valid_idx = np.flatnonzero(valid)
        invalid_idx = np.flatnonzero(~valid)
        if valid_idx.size == 0:
            return np.random.choice(n_total, size=self.n_points, replace=False)

        # Aim for roughly ``valid_oversample × natural_ratio`` valid
        # points in the subsample.
        natural_ratio = valid_idx.size / n_total
        target_valid_ratio = min(0.95, natural_ratio * self.valid_oversample)
        n_valid_target = int(round(self.n_points * target_valid_ratio))
        n_valid_target = min(n_valid_target, valid_idx.size)
        n_valid_target = max(n_valid_target, min(self.require_valid_pts, valid_idx.size))
        n_invalid_target = self.n_points - n_valid_target

        v_pick = np.random.choice(valid_idx, size=n_valid_target, replace=False)
        if n_invalid_target > 0:
            if invalid_idx.size >= n_invalid_target:
                i_pick = np.random.choice(invalid_idx, size=n_invalid_target, replace=False)
            else:
                # Not enough invalid points; pad with valid repeats.
                i_pick = np.random.choice(invalid_idx, size=n_invalid_target, replace=True) \
                    if invalid_idx.size else np.random.choice(valid_idx, size=n_invalid_target, replace=True)
        else:
            i_pick = np.zeros(0, dtype=np.int64)
        idx = np.concatenate([v_pick, i_pick])
        np.random.shuffle(idx)
        return idx

    def __getitem__(self, item_idx: int) -> dict[str, "torch.Tensor"]:
        path = self.files[item_idx]
        with np.load(path) as data:
            prior_xyz = data["prior_xyz"].astype(np.float64)
            prior_rgb = data["prior_rgb"].astype(np.float32) / 255.0
            prior_feat = data["prior_feat"].astype(np.float32)
            target_disp = data["target_disp"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
            # segment_id from SAM Auto (-1 means "no SAM Auto data" or
            # "key absent in older npz"); the segment-aware EdgeConv
            # treats the all-equal case (e.g. all -1) as no masking.
            if "segment_id" in data.files:
                segment_id = data["segment_id"].astype(np.int64)
            else:
                segment_id = np.full(prior_xyz.shape[0], -1, dtype=np.int64)

        idx = self._subsample(prior_xyz.shape[0], valid)
        prior_xyz = prior_xyz[idx]
        prior_rgb = prior_rgb[idx]
        prior_feat = prior_feat[idx]
        target_disp = target_disp[idx]
        valid = valid[idx]
        segment_id = segment_id[idx]

        # Centre + scale XYZ.
        centre = prior_xyz.mean(axis=0, keepdims=True)
        prior_xyz_c = (prior_xyz - centre) / XYZ_SCALE

        if self.random_flip_x and random.random() < 0.5:
            prior_xyz_c[:, 0] *= -1.0
            target_disp[:, 0] *= -1.0
        if self.random_yaw:
            theta = random.uniform(-math.pi / 6, math.pi / 6)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            prior_xyz_c = prior_xyz_c @ R.T
            target_disp = (target_disp.astype(np.float64) @ R.T).astype(np.float32)

        target_disp_n = (target_disp / DISP_SCALE).astype(np.float32)

        return {
            "prior_xyz": torch.from_numpy(prior_xyz_c.astype(np.float32)),
            "prior_rgb": torch.from_numpy(prior_rgb.astype(np.float32)),
            "prior_feat": torch.from_numpy(prior_feat),
            "segment_id": torch.from_numpy(segment_id.astype(np.int64)),
            "target_disp": torch.from_numpy(target_disp_n),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "centre": torch.from_numpy(centre.squeeze(0).astype(np.float32)),
            "path": str(path),
        }


__all__ = [
    "Completion3DDataset",
    "DISP_SCALE",
    "PRIOR_FEAT_DIM",
    "XYZ_SCALE",
]
