"""Per-pixel sigma-uncertainty head for multi-camera static-scene fusion.

The closed-form AA-HAD prediction at a pixel ``(u, v)`` is

    z_pred(u, v) = a * d_tilde(u, v) + b

where ``a, b`` are the per-camera affine parameters fit on the static
LiDAR. The residual ``z_lidar - z_pred`` is heteroscedastic — it is
small in the well-fit body of the road and large at depth
discontinuities, the image periphery, or in regions where DA3 fails
(reflections, shadowed lanes). Plain mean-fusion across the four
pinhole cameras therefore inherits the worst camera's failure modes
and produces visible "shells".

This module trains a tiny MLP that predicts ``log sigma`` per pixel
from a small set of geometric features, optimised under heteroscedastic
Gaussian NLL

    L = log sigma + 0.5 * r^2 * exp(-2 * log sigma)

so well-fit pixels collapse to small sigma and outliers expand to
large sigma. At fusion time, voxels combine per-camera votes with
inverse-variance weights ``w = 1 / sigma^2`` instead of the unweighted
mean, which deflates the contribution of unreliable cameras voxel by
voxel.

The pure-numpy helpers ``features_for_pixels`` and
``sigma_weighted_voxel`` are usable without torch installed; only the
``SigmaHead`` class and ``gaussian_nll_loss`` require torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_OK = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_OK = False

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401


SIGMA_HEAD_FEATURE_DIM = 8
"""Number of per-pixel features the sigma head consumes.

Order: ``[d_tilde, u_norm, v_norm, radial, z_pred, log1p(z_pred), conf, a_per_cam]``
"""

_LOG_SIGMA_MIN = -5.0
_LOG_SIGMA_MAX = 5.0


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError(
            "torch is required for SigmaHead / gaussian_nll_loss but is not "
            "installed. Install with `pip install torch>=2.1`."
        )


if _TORCH_OK:

    class SigmaHead(nn.Module):
        """8-D -> log sigma MLP.

        Two hidden layers of 64 with GELU, single scalar output that is
        the predicted ``log sigma``. ``predict_sigma`` clamps the output
        to ``[exp(-5), exp(5)]`` for numerical stability before
        exponentiation.
        """

        def __init__(self, in_dim: int = SIGMA_HEAD_FEATURE_DIM, hidden: int = 64) -> None:
            super().__init__()
            self.in_dim = in_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            log_sigma = self.net(x).squeeze(-1)
            return log_sigma

        @torch.no_grad()
        def predict_sigma(self, x: "torch.Tensor") -> "torch.Tensor":
            log_sigma = self.forward(x).clamp(min=_LOG_SIGMA_MIN, max=_LOG_SIGMA_MAX)
            return log_sigma.exp()

    def gaussian_nll_loss(
        log_sigma: "torch.Tensor",
        residual: "torch.Tensor",
        reduction: str = "mean",
    ) -> "torch.Tensor":
        """Heteroscedastic Gaussian NLL.

        ``L = log sigma + 0.5 * r^2 * exp(-2 * log sigma)`` (constants
        dropped). For numerical safety, ``log_sigma`` is clamped before
        the inner ``exp(-2 log sigma)``; gradients still flow through
        the unclamped term outside the exponent.
        """
        log_sigma_c = log_sigma.clamp(min=_LOG_SIGMA_MIN, max=_LOG_SIGMA_MAX)
        inv_var = torch.exp(-2.0 * log_sigma_c)
        loss = log_sigma + 0.5 * residual.pow(2) * inv_var
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"unknown reduction {reduction!r}")

else:  # pragma: no cover - exercised on torch-less envs only

    class SigmaHead:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    def gaussian_nll_loss(*args, **kwargs):  # type: ignore[no-redef]
        _require_torch()


def features_for_pixels(
    uv: np.ndarray,
    d_tilde: np.ndarray,
    K: np.ndarray,
    a: float,
    b: float,
    image_hw: tuple[int, int] | None = None,
    conf: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 8-D feature vector at each pixel.

    Parameters
    ----------
    uv : (N, 2) array of integer or float pixel coordinates ``(u, v)``.
    d_tilde : (N,) array of raw DA3 depths at those pixels.
    K : (3, 3) intrinsics matrix.
    a, b : per-camera AA-HAD affine ``(z = a * d_tilde + b)``.
    image_hw : optional ``(H, W)``; used to build ``u_norm``, ``v_norm``
        in [-1, 1]. If ``None``, the principal point and focal length
        are used as a proxy via ``(u - cx) / fx`` for normalisation,
        which gives the camera-space x/z (a unit-depth ray
        coordinate).
    conf : optional (N,) confidence in [0, 1] (e.g. inverse of DA3
        per-pixel variance). If ``None``, filled with ones.

    Returns
    -------
    (N, 8) float32 feature array.
    """
    uv = np.asarray(uv, dtype=np.float64)
    d_tilde = np.asarray(d_tilde, dtype=np.float64).reshape(-1)
    if uv.shape[0] != d_tilde.shape[0]:
        raise ValueError(
            f"uv has {uv.shape[0]} rows but d_tilde has {d_tilde.shape[0]}"
        )
    K = np.asarray(K, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = uv[:, 0]
    v = uv[:, 1]
    if image_hw is not None:
        H, W = image_hw
        u_norm = (u - 0.5 * W) / (0.5 * W)
        v_norm = (v - 0.5 * H) / (0.5 * H)
    else:
        u_norm = (u - cx) / max(fx, 1e-6)
        v_norm = (v - cy) / max(fy, 1e-6)
    radial = np.hypot(u_norm, v_norm)

    z_pred = a * d_tilde + b
    log_z = np.log1p(np.clip(z_pred, a_min=0.0, a_max=None))

    if conf is None:
        conf_arr = np.ones_like(d_tilde)
    else:
        conf_arr = np.asarray(conf, dtype=np.float64).reshape(-1)
        if conf_arr.shape[0] != d_tilde.shape[0]:
            raise ValueError(
                f"conf has {conf_arr.shape[0]} rows but d_tilde has {d_tilde.shape[0]}"
            )

    a_col = np.full_like(d_tilde, float(a))

    feats = np.stack(
        [d_tilde, u_norm, v_norm, radial, z_pred, log_z, conf_arr, a_col],
        axis=1,
    ).astype(np.float32)
    return feats


def sigma_weighted_voxel(
    points: np.ndarray,
    sigmas: np.ndarray,
    voxel_size: float,
    colors: np.ndarray | None = None,
    sigma_floor: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Inverse-variance-weighted voxel grid downsample.

    For every voxel the centroid is

        c = sum(w_i * p_i) / sum(w_i),  w_i = 1 / max(sigma_i, sigma_floor)^2

    and the per-voxel sigma is reported as ``1 / sqrt(sum w_i)`` (the
    standard error of the weighted mean assuming the per-vote sigmas
    are well calibrated). Colors are weighted by the same ``w_i``.

    Parameters
    ----------
    points : (N, 3) world-frame XYZ.
    sigmas : (N,) predicted per-point sigma in metres.
    voxel_size : voxel edge length in metres.
    colors : optional (N, 3) uint8 RGB.
    sigma_floor : minimum sigma in metres (defaults to 1 cm).

    Returns
    -------
    centroids : (M, 3) float32 weighted-mean XYZ.
    voxel_colors : (M, 3) uint8, or ``None`` if ``colors`` was ``None``.
    voxel_sigma : (M,) float32 standard error per voxel.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if sigmas.shape[0] != points.shape[0]:
        raise ValueError(
            f"sigmas has {sigmas.shape[0]} rows but points has {points.shape[0]}"
        )
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    pts = np.asarray(points, dtype=np.float64)
    sig = np.maximum(np.asarray(sigmas, dtype=np.float64).reshape(-1), float(sigma_floor))
    w = 1.0 / (sig * sig)

    keys = np.floor(pts / float(voxel_size)).astype(np.int64)
    # Linearise the 3-D voxel index for fast unique. Use a 1-D view to
    # avoid building string keys.
    flat = (
        keys[:, 0].astype(np.int64) * 73856093
        ^ keys[:, 1].astype(np.int64) * 19349663
        ^ keys[:, 2].astype(np.int64) * 83492791
    )
    # Hash collisions are unlikely but possible; resolve by also
    # grouping on the raw triplet.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    pts_s = pts[order]
    w_s = w[order]
    keys_s = keys[order]
    if colors is not None:
        col_s = np.asarray(colors, dtype=np.float64)[order]

    # Find run boundaries on the sorted (kx, ky, kz) tuples.
    diff = np.any(keys_s[1:] != keys_s[:-1], axis=1)
    starts = np.concatenate([[0], np.where(diff)[0] + 1])
    ends = np.concatenate([starts[1:], [pts_s.shape[0]]])

    centroids = np.empty((starts.size, 3), dtype=np.float64)
    voxel_sigma = np.empty(starts.size, dtype=np.float64)
    voxel_colors = (
        np.empty((starts.size, 3), dtype=np.float64) if colors is not None else None
    )
    for i, (s, e) in enumerate(zip(starts, ends)):
        ws = w_s[s:e]
        ps = pts_s[s:e]
        wsum = ws.sum()
        centroids[i] = (ps * ws[:, None]).sum(axis=0) / wsum
        voxel_sigma[i] = 1.0 / np.sqrt(wsum)
        if voxel_colors is not None:
            voxel_colors[i] = (col_s[s:e] * ws[:, None]).sum(axis=0) / wsum

    centroids_f = centroids.astype(np.float32)
    voxel_sigma_f = voxel_sigma.astype(np.float32)
    if voxel_colors is not None:
        voxel_colors_u8 = np.clip(voxel_colors, 0.0, 255.0).astype(np.uint8)
    else:
        voxel_colors_u8 = None  # type: ignore[assignment]
    _ = flat  # silence unused; kept for future hash-based bucketing
    return centroids_f, voxel_colors_u8, voxel_sigma_f


class SigmaPredictor:
    """Inference-time wrapper around a trained ``SigmaHead`` checkpoint.

    Loads the saved ``state_dict``, feature mean / std, and per-camera
    ``(a, b)`` calibration. Calling the instance with the per-pixel
    geometric inputs returns per-pixel ``sigma`` in metres.
    """

    def __init__(self, checkpoint_path, device: str = "cpu") -> None:
        _require_torch()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        in_dim = int(ckpt.get("in_dim", SIGMA_HEAD_FEATURE_DIM))
        head = SigmaHead(in_dim=in_dim).to(device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        self.head = head
        self.device = device
        self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1)
        self.feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1)
        self.calib = ckpt.get("calib", {})
        self.scene_id = ckpt.get("scene_id")

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        x = (features.astype(np.float32) - self.feature_mean) / self.feature_std
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            sigma = self.head.predict_sigma(xt)
        return sigma.cpu().numpy().astype(np.float32)

    def predict_for_pixels(
        self,
        uv: np.ndarray,
        d_tilde: np.ndarray,
        K: np.ndarray,
        a: float,
        b: float,
        image_hw: tuple[int, int] | None = None,
        conf: np.ndarray | None = None,
    ) -> np.ndarray:
        feats = features_for_pixels(uv, d_tilde, K, a, b, image_hw, conf)
        return self.predict(feats)
