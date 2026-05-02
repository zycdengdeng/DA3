"""Inference wrapper for the per-point flow-matching residual head — Stage 7D.

Loads a trained :class:`PointCloudVelocityNet` checkpoint and exposes
``refine_cloud(...)``, which takes a prior cloud (Stage 1+3 output)
plus conditioning features and returns a refined cloud:

    refined_xyz = prior_xyz + Δxyz_predicted

Δxyz is sampled by integrating the rectified flow from
``x_0 ~ N(0, I)`` for ``n_steps`` Euler steps (default 4), then
de-normalised by ``DISP_SCALE`` (5 m).

The wrapper also exposes a ``--gate-distance-m`` knob to clamp the
predicted displacement to zero outside a distance-to-LiDAR threshold,
matching the inference-side gating discussed for the 2-D residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_anchored_depth.data.completion_3d_dataset import (
    DISP_SCALE,
    PRIOR_FEAT_DIM,
    XYZ_SCALE,
)
from lidar_anchored_depth.models.flow_matching import RectifiedFlowMatcher
from lidar_anchored_depth.models.point_cloud_unet import PointCloudVelocityNet


def build_prior_features(
    prior_xyz_world: np.ndarray,
    prior_uv: np.ndarray,
    d_image: np.ndarray,
    a: float,
    b: float,
    target_lidar_xyz: np.ndarray | None,
    *,
    dist_clip_m: float = 5.0,
    sparse_present_thresh_m: float = 0.10,
) -> np.ndarray:
    """Recompute the per-point conditioning features the data-prep
    script writes: ``[source_label, d_tilde, aahad_init_z,
    dist_to_lidar_clipped, sparse_present]``.

    Parameters
    ----------
    prior_xyz_world : (N, 3) world-frame XYZ.
    prior_uv : (N, 2) integer source pixel coords.
    d_image : (H, W) DA3 raw depth.
    a, b : per-camera AA-HAD scalars.
    target_lidar_xyz : optional (M, 3) static-LiDAR cloud for the KDTree
        distance lookup. If ``None``, ``dist_to_lidar`` is set to the
        clip ceiling and ``sparse_present`` to 0.
    """
    n = prior_xyz_world.shape[0]
    rows = prior_uv[:, 1]
    cols = prior_uv[:, 0]
    d_at = d_image[rows, cols].astype(np.float32)
    aahad_init = (a * d_at + b).astype(np.float32)

    if target_lidar_xyz is not None and target_lidar_xyz.shape[0] > 0:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(target_lidar_xyz)
            d, _ = tree.query(prior_xyz_world, k=1)
            dist_clipped = np.minimum(d, dist_clip_m).astype(np.float32)
        except ModuleNotFoundError:
            dist_clipped = np.full(n, dist_clip_m, dtype=np.float32)
    else:
        dist_clipped = np.full(n, dist_clip_m, dtype=np.float32)

    sparse_present = (dist_clipped <= sparse_present_thresh_m).astype(np.float32)
    source_label = np.zeros(n, dtype=np.float32)
    feat = np.stack([
        source_label, d_at, aahad_init, dist_clipped, sparse_present,
    ], axis=1).astype(np.float32)
    return feat


class PointResidualPredictor:
    """Inference-time wrapper around a trained ``PointCloudVelocityNet``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        n_steps: int = 4,
        gate_distance_m: float | None = None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ImportError("PointResidualPredictor requires torch>=2.1") from e

        if not torch.cuda.is_available() and device.startswith("cuda"):
            device = "cpu"
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {})
        base = int(train_args.get("base_channels", 64))
        k = int(train_args.get("knn_k", 16))
        knn_chunk = int(train_args.get("knn_chunk", 2048)) or None

        net = PointCloudVelocityNet(
            prior_feat_dim=PRIOR_FEAT_DIM,
            base=base, k=k, knn_chunk=knn_chunk,
        ).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.net = net
        self.device = device
        self.n_steps = int(n_steps)
        self.gate_distance_m = (
            float(gate_distance_m) if gate_distance_m is not None else None
        )
        self.matcher = RectifiedFlowMatcher(sigma=0.0)

    @staticmethod
    def _normalize_inputs(
        prior_xyz: np.ndarray,
        prior_rgb: np.ndarray,
        prior_feat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centre = prior_xyz.mean(axis=0, keepdims=True)
        xyz_n = ((prior_xyz - centre) / XYZ_SCALE).astype(np.float32)
        rgb_n = (prior_rgb.astype(np.float32) / 255.0).astype(np.float32)
        feat_n = prior_feat.astype(np.float32)
        return xyz_n, rgb_n, feat_n, centre

    def refine_cloud(
        self,
        prior_xyz: np.ndarray,
        prior_rgb: np.ndarray,
        prior_feat: np.ndarray,
        dist_to_lidar: np.ndarray | None = None,
        segment_id: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict per-point displacement Δxyz and return the refined
        world-frame cloud ``prior_xyz + Δxyz``.

        Parameters
        ----------
        prior_xyz : (N, 3) world XYZ
        prior_rgb : (N, 3) uint8 RGB (will be normalised to [0, 1])
        prior_feat : (N, F) per-point features (F = PRIOR_FEAT_DIM = 5)
        dist_to_lidar : optional (N,) per-point distance to nearest
            LiDAR. When ``self.gate_distance_m`` is set, points with
            ``dist > gate_distance_m`` keep ``Δxyz = 0`` (network not
            applied there) — matches the inference-side gating
            discussed for the 2-D head.
        """
        import torch

        if prior_xyz.shape[0] == 0:
            return prior_xyz.astype(np.float32)
        if prior_feat.shape[1] != PRIOR_FEAT_DIM:
            raise ValueError(
                f"prior_feat must have {PRIOR_FEAT_DIM} channels, "
                f"got {prior_feat.shape[1]}"
            )

        xyz_n, rgb_n, feat_n, centre = self._normalize_inputs(
            prior_xyz, prior_rgb, prior_feat,
        )
        N = xyz_n.shape[0]

        # Single-batch inference (B=1). For very large N (> 32k) we'd
        # need to subsample or chunk; keep simple for now and let the
        # caller decide.
        xyz_t = torch.from_numpy(xyz_n).unsqueeze(0).to(self.device)
        rgb_t = torch.from_numpy(rgb_n).unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(feat_n).unsqueeze(0).to(self.device)
        seg_t = None
        if segment_id is not None:
            seg_t = torch.from_numpy(
                segment_id.astype(np.int64),
            ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x_0 = torch.randn(1, N, 3, device=self.device)
            x = x_0
            dt = 1.0 / self.n_steps
            for k in range(self.n_steps):
                t_now = torch.full(
                    (1,), float(k * dt), device=self.device, dtype=x.dtype,
                )
                v = self.net(xyz_t, rgb_t, feat_t, x, t_now, segment_id=seg_t)
                x = x + dt * v
        delta_xyz = (x.squeeze(0).cpu().numpy() * DISP_SCALE).astype(np.float32)

        if self.gate_distance_m is not None and dist_to_lidar is not None:
            far = dist_to_lidar > float(self.gate_distance_m)
            delta_xyz[far] = 0.0

        return (prior_xyz.astype(np.float32) + delta_xyz).astype(np.float32)


__all__ = [
    "PointResidualPredictor",
    "build_prior_features",
]
