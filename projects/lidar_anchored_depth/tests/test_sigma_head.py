"""Tests for ``models.sigma_head`` — Stage 4A.

Covers the pure-numpy helpers (``features_for_pixels``,
``sigma_weighted_voxel``) unconditionally and the torch-dependent
parts (``SigmaHead``, ``gaussian_nll_loss``, ``SigmaPredictor``)
conditionally on torch availability.
"""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.models import (
    SIGMA_HEAD_FEATURE_DIM,
    features_for_pixels,
    sigma_weighted_voxel,
)
from lidar_anchored_depth.models.sigma_head import _TORCH_OK


# --- features_for_pixels ----------------------------------------------------

def _K(fx: float = 1000.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])


def test_features_shape_and_dim():
    K = _K()
    uv = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float64)
    d = np.array([5.0, 10.0, 20.0])
    feats = features_for_pixels(uv, d, K, a=0.9, b=2.0, image_hw=(480, 640))
    assert feats.shape == (3, SIGMA_HEAD_FEATURE_DIM)
    assert feats.dtype == np.float32


def test_features_z_pred_matches_linear_model():
    K = _K()
    uv = np.array([[320, 240]], dtype=np.float64)
    d = np.array([10.0])
    feats = features_for_pixels(uv, d, K, a=0.9, b=2.0, image_hw=(480, 640))
    # z_pred is the 5th feature (index 4)
    assert np.isclose(feats[0, 4], 0.9 * 10.0 + 2.0)
    # log1p z_pred is the 6th feature
    assert np.isclose(feats[0, 5], np.log1p(0.9 * 10.0 + 2.0))
    # a_per_cam is the last feature
    assert np.isclose(feats[0, 7], 0.9)


def test_features_normalised_uv_in_image_hw_mode():
    K = _K()
    H, W = 480, 640
    # Centre pixel maps to (0, 0) in [-1, 1] normalisation
    uv = np.array([[W / 2, H / 2], [0, 0], [W - 1, H - 1]], dtype=np.float64)
    d = np.array([5.0, 5.0, 5.0])
    feats = features_for_pixels(uv, d, K, a=1.0, b=0.0, image_hw=(H, W))
    assert np.allclose(feats[0, 1:3], [0.0, 0.0])
    assert np.allclose(feats[1, 1:3], [-1.0, -1.0])
    # Last pixel close to (1, 1) (with 1-pixel offset due to half-pixel centre)
    assert feats[2, 1] > 0.99
    assert feats[2, 2] > 0.99


def test_features_radial_is_hypot_of_uv_norm():
    K = _K()
    uv = np.array([[100, 100], [320, 240]], dtype=np.float64)
    d = np.array([5.0, 5.0])
    feats = features_for_pixels(uv, d, K, a=1.0, b=0.0, image_hw=(480, 640))
    expected = np.hypot(feats[:, 1], feats[:, 2])
    assert np.allclose(feats[:, 3], expected)


def test_features_conf_default_ones_and_overrideable():
    K = _K()
    uv = np.array([[100, 100], [200, 200]], dtype=np.float64)
    d = np.array([5.0, 10.0])
    feats_default = features_for_pixels(uv, d, K, a=1.0, b=0.0, image_hw=(480, 640))
    assert np.allclose(feats_default[:, 6], 1.0)

    conf = np.array([0.3, 0.7])
    feats_set = features_for_pixels(
        uv, d, K, a=1.0, b=0.0, image_hw=(480, 640), conf=conf,
    )
    assert np.allclose(feats_set[:, 6], conf)


def test_features_raises_on_mismatched_lengths():
    K = _K()
    with pytest.raises(ValueError):
        features_for_pixels(
            np.zeros((3, 2)), np.zeros(2), K, a=1.0, b=0.0, image_hw=(480, 640),
        )


# --- sigma_weighted_voxel ---------------------------------------------------

def test_sigma_voxel_groups_by_voxel_key():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.05, 0.05],     # same voxel as above (0.1m voxel)
        [10.0, 10.0, 10.0],     # different voxel
    ])
    sigmas = np.array([0.1, 0.1, 0.1])
    c, _, _ = sigma_weighted_voxel(pts, sigmas, voxel_size=0.1)
    assert c.shape == (2, 3)


def test_sigma_voxel_low_sigma_dominates_centroid():
    # Two points in the same voxel; one has 100x lower sigma -> centroid
    # should be much closer to it than a plain mean would be.
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    sigmas = np.array([1.0, 0.01])  # second point has 10000x more weight
    c, _, vs = sigma_weighted_voxel(pts, sigmas, voxel_size=0.2)
    assert c.shape == (1, 3)
    # Plain mean would be at x=0.025; weighted should be near x=0.05
    assert c[0, 0] > 0.049
    # The voxel sigma should be near 0.01 (dominated by the precise point)
    assert vs[0] < 0.02


def test_sigma_voxel_color_weighting():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sigmas = np.array([1.0, 0.01])
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    c, col, _ = sigma_weighted_voxel(pts, sigmas, voxel_size=0.1, colors=colors)
    assert c.shape == (1, 3)
    assert col is not None
    # Green (high-confidence) should dominate
    assert col[0, 1] > col[0, 0]
    assert col[0, 1] > 200


def test_sigma_voxel_per_voxel_sigma_decreases_with_more_votes():
    # Many low-sigma votes in one voxel should give a tighter standard
    # error than a single vote.
    rng = np.random.default_rng(0)
    n = 100
    pts_one = np.array([[0.0, 0.0, 0.0]])
    pts_many = rng.uniform(-0.04, 0.04, size=(n, 3))
    s_const = np.full(n, 0.1)

    _, _, vs_one = sigma_weighted_voxel(pts_one, np.array([0.1]), voxel_size=0.1)
    _, _, vs_many = sigma_weighted_voxel(pts_many, s_const, voxel_size=0.1)
    # SE = 1/sqrt(sum w) so 100x votes -> 10x smaller SE
    assert vs_many[0] < 0.5 * vs_one[0]


def test_sigma_voxel_floor_clips_zero_sigma():
    # A "perfect" sigma of 0 must not overflow to infinite weight.
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]])
    sigmas = np.array([0.0, 0.5])
    c, _, vs = sigma_weighted_voxel(pts, sigmas, voxel_size=0.2, sigma_floor=0.01)
    assert np.isfinite(c).all()
    assert np.isfinite(vs).all()


def test_sigma_voxel_raises_on_bad_inputs():
    pts = np.zeros((5, 3))
    with pytest.raises(ValueError):
        sigma_weighted_voxel(pts, np.zeros(4), voxel_size=0.1)
    with pytest.raises(ValueError):
        sigma_weighted_voxel(pts, np.zeros(5), voxel_size=0.0)
    with pytest.raises(ValueError):
        sigma_weighted_voxel(np.zeros((5, 2)), np.zeros(5), voxel_size=0.1)


# --- torch-only paths -------------------------------------------------------

torch_required = pytest.mark.skipif(not _TORCH_OK, reason="requires torch")


@torch_required
def test_sigma_head_forward_shape():
    import torch
    from lidar_anchored_depth.models import SigmaHead

    head = SigmaHead()
    x = torch.randn(7, SIGMA_HEAD_FEATURE_DIM)
    log_sigma = head(x)
    assert log_sigma.shape == (7,)
    sigma = head.predict_sigma(x)
    assert sigma.shape == (7,)
    assert (sigma > 0).all()


@torch_required
def test_gaussian_nll_minimum_near_log_abs_residual():
    """For a single sample with residual r, the loss
    ``log sigma + 0.5 r^2 / sigma^2`` is minimised at sigma = |r|. We
    optimise log_sigma directly with autograd and check it converges to
    log |r|."""
    import torch
    from lidar_anchored_depth.models import gaussian_nll_loss

    r = torch.tensor([2.0])
    log_sigma = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([log_sigma], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss = gaussian_nll_loss(log_sigma, r, reduction="mean")
        loss.backward()
        opt.step()
    assert torch.allclose(log_sigma.detach(), torch.log(torch.abs(r)), atol=1e-2)


@torch_required
def test_gaussian_nll_reductions():
    import torch
    from lidar_anchored_depth.models import gaussian_nll_loss

    log_sigma = torch.zeros(4)
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    none_loss = gaussian_nll_loss(log_sigma, r, reduction="none")
    assert none_loss.shape == (4,)
    assert torch.allclose(
        gaussian_nll_loss(log_sigma, r, reduction="mean"),
        none_loss.mean(),
    )
    assert torch.allclose(
        gaussian_nll_loss(log_sigma, r, reduction="sum"),
        none_loss.sum(),
    )
    with pytest.raises(ValueError):
        gaussian_nll_loss(log_sigma, r, reduction="bogus")


@torch_required
def test_sigma_predictor_roundtrip(tmp_path):
    """Train a tiny head for a few steps, save it, reload via
    ``SigmaPredictor``, and verify the inference output matches the
    original model on the same inputs."""
    import torch

    from lidar_anchored_depth.models import SigmaHead, SigmaPredictor, gaussian_nll_loss

    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, SIGMA_HEAD_FEATURE_DIM)).astype(np.float32)
    # Synthetic residual scale that depends on the first feature -> the
    # head has something to learn (just verifies plumbing).
    y = (rng.standard_normal(512) * (0.5 + np.abs(X[:, 0]))).astype(np.float32)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True).clip(min=1e-6)
    Xn = (X - mean) / std

    head = SigmaHead()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y)
    for _ in range(20):
        opt.zero_grad()
        gaussian_nll_loss(head(Xt), yt, reduction="mean").backward()
        opt.step()

    ckpt_path = tmp_path / "head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": SIGMA_HEAD_FEATURE_DIM,
            "feature_mean": mean,
            "feature_std": std,
            "calib": {"0": {"a": 1.0, "b": 0.0}},
            "scene_id": "synthetic",
        },
        ckpt_path,
    )

    pred = SigmaPredictor(ckpt_path, device="cpu")
    sigma_via_predictor = pred.predict(X[:16])

    head.eval()
    with torch.no_grad():
        sigma_direct = head.predict_sigma(torch.from_numpy(Xn[:16])).numpy()
    assert np.allclose(sigma_via_predictor, sigma_direct, atol=1e-5)


# --- error path on torch-less envs ------------------------------------------

@pytest.mark.skipif(_TORCH_OK, reason="only meaningful when torch is missing")
def test_sigma_head_raises_without_torch():
    from lidar_anchored_depth.models import SigmaHead

    with pytest.raises(ImportError):
        SigmaHead()
