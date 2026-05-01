"""Inference wrapper for the residual completion U-Net — Stage 6E.

Loads a trained :class:`ResidualVelocityUNet` checkpoint and exposes
``refine_depth_image(...)``, which takes the per-frame inputs that the
completion script already builds (RGB, ``d̃``, AA-HAD initial depth,
plus the sparse static LiDAR projected to the image) and returns the
refined per-pixel metric depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import points_in_oriented_bbox
from lidar_anchored_depth.alignment.projection import world_to_image
from lidar_anchored_depth.data.completion_dataset import (
    DZ_NORM,
    D_TILDE_NORM,
    Z_NORM,
)
from lidar_anchored_depth.models.flow_matching import RectifiedFlowMatcher
from lidar_anchored_depth.models.residual_unet import (
    COND_CHANNELS,
    ResidualVelocityUNet,
)

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401


def _build_sparse_lidar_map(
    lidar_world: np.ndarray,
    dynamic_objects,
    K: np.ndarray,
    T_wc: np.ndarray,
    distortion: np.ndarray | None,
    image_hw: tuple[int, int],
    *,
    bbox_expand: float = 0.10,
    z_min: float = 0.5,
    z_max: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``LiDAR \\ V2X-bbox`` onto the camera plane.

    Returns ``(sparse_z (H, W) float32, sparse_mask (H, W) bool)``.
    Same logic as the training-time data prep in
    ``scripts/prepare_completion_data.py``, kept consistent so the
    network sees the same input distribution at train / inference time.
    """
    H, W = image_hw
    sparse_z = np.zeros((H, W), dtype=np.float32)
    sparse_mask = np.zeros((H, W), dtype=bool)
    if lidar_world.size == 0:
        return sparse_z, sparse_mask

    is_dyn = np.zeros(lidar_world.shape[0], dtype=bool)
    for obj in dynamic_objects or []:
        is_dyn |= points_in_oriented_bbox(lidar_world, obj, expand=bbox_expand)
    static = lidar_world[~is_dyn]
    if static.shape[0] == 0:
        return sparse_z, sparse_mask

    uv, z_cam, _ = world_to_image(static, K, T_wc, distortion)
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]; z_cam = z_cam[finite]
    if uv.size == 0:
        return sparse_z, sparse_mask
    uv_int = np.round(uv).astype(np.int64)
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    uv_int = uv_int[inside]; z_cam = z_cam[inside]
    keep = (z_cam >= z_min) & (z_cam <= z_max)
    uv_int = uv_int[keep]; z_cam = z_cam[keep]
    if uv_int.shape[0] == 0:
        return sparse_z, sparse_mask
    order = np.argsort(-z_cam)  # smallest z overwrites
    sparse_z[uv_int[order, 1], uv_int[order, 0]] = z_cam[order].astype(np.float32)
    sparse_mask[uv_int[order, 1], uv_int[order, 0]] = True
    return sparse_z, sparse_mask


class ResidualPredictor:
    """Loads a trained CFM residual U-Net and refines per-frame depth.

    Usage
    -----
    >>> p = ResidualPredictor(ckpt_path, device="cuda")
    >>> z_refined = p.refine_depth_image(
    ...     rgb=frame.image,             # (H, W, 3) uint8
    ...     d_image=d_image,             # (H, W) float32
    ...     z_aahad=z_aahad_image,       # (H, W) float32 = a*d̃+b (or grid)
    ...     sparse_z=sparse_z,           # (H, W) float32
    ...     sparse_mask=sparse_mask,     # (H, W) bool
    ... )
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        n_steps: int = 4,
        base_channels: int | None = None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ImportError("ResidualPredictor requires torch>=2.1") from e

        if not torch.cuda.is_available() and device.startswith("cuda"):
            device = "cpu"
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {})
        base = base_channels or int(train_args.get("base_channels", 32))
        net = ResidualVelocityUNet(
            cond_channels=COND_CHANNELS, base=base,
        ).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.net = net
        self.device = device
        self.n_steps = int(n_steps)
        self.matcher = RectifiedFlowMatcher(sigma=0.0)

    def refine_depth_image(
        self,
        rgb: np.ndarray,
        d_image: np.ndarray,
        z_aahad: np.ndarray,
        sparse_z: np.ndarray,
        sparse_mask: np.ndarray,
    ) -> np.ndarray:
        """Return ``(H, W) float32`` refined metric depth.

        All inputs are at native camera resolution. The network expects
        H, W divisible by 8 (4 levels of 2× pooling); the wrapper
        handles uneven sizes by padding to a multiple of 8 and
        cropping back.
        """
        import torch

        H, W = d_image.shape[:2]
        if rgb.shape[:2] != (H, W) or z_aahad.shape != (H, W):
            raise ValueError("rgb / d_image / z_aahad must share (H, W)")

        # Build conditioning tensor (1, 7, H, W).
        rgb_n = rgb.astype(np.float32) / 255.0
        d_n = d_image.astype(np.float32) / D_TILDE_NORM
        ahad_n = z_aahad.astype(np.float32) / Z_NORM
        sz_n = sparse_z.astype(np.float32) / Z_NORM
        sm_n = sparse_mask.astype(np.float32)
        cond = np.concatenate([
            rgb_n.transpose(2, 0, 1),
            d_n[None],
            ahad_n[None],
            sz_n[None],
            sm_n[None],
        ], axis=0)
        cond_t = torch.from_numpy(cond).unsqueeze(0).to(self.device)

        # Pad H, W to a multiple of 8.
        pad_h = (-H) % 8
        pad_w = (-W) % 8
        if pad_h or pad_w:
            cond_t = torch.nn.functional.pad(
                cond_t, (0, pad_w, 0, pad_h), mode="reflect",
            )

        with torch.no_grad():
            x_pred = self.matcher.sample(
                self.net, cond_t,
                n_steps=self.n_steps,
                target_shape=(1, 1, cond_t.shape[2], cond_t.shape[3]),
            )
        if pad_h or pad_w:
            x_pred = x_pred[..., :H, :W]

        delta_z = x_pred.squeeze(0).squeeze(0).cpu().numpy() * DZ_NORM
        return (z_aahad.astype(np.float32) + delta_z.astype(np.float32)).astype(np.float32)


__all__ = ["ResidualPredictor", "_build_sparse_lidar_map"]
