"""Tests for the rectified flow matcher + masked CFM loss."""

from __future__ import annotations

import pytest

try:
    import torch
    _TORCH_OK = True
except ModuleNotFoundError:
    _TORCH_OK = False

torch_required = pytest.mark.skipif(not _TORCH_OK, reason="requires torch")


@torch_required
def test_sample_train_pair_linear_path_at_endpoints():
    from lidar_anchored_depth.models import RectifiedFlowMatcher

    fm = RectifiedFlowMatcher(sigma=0.0)
    x_1 = torch.zeros(4, 1, 8, 8)
    x_0 = torch.ones(4, 1, 8, 8)
    pair = fm.sample_train_pair(x_1, x_0=x_0)
    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1, so x_t[i] should
    # equal x_0 at t=0 and x_1 at t=1. With sigma=0 there is no noise.
    t = pair.t
    expected = (1.0 - t.view(4, 1, 1, 1)) * x_0 + t.view(4, 1, 1, 1) * x_1
    assert torch.allclose(pair.x_t, expected, atol=1e-6)
    assert torch.allclose(pair.u_target, x_1 - x_0)


@torch_required
def test_sample_train_pair_zero_target_velocity_at_zero_distance():
    from lidar_anchored_depth.models import RectifiedFlowMatcher

    fm = RectifiedFlowMatcher(sigma=0.0)
    x = torch.randn(2, 1, 4, 4)
    pair = fm.sample_train_pair(x, x_0=x)  # x_0 == x_1
    assert torch.allclose(pair.u_target, torch.zeros_like(x))


@torch_required
def test_sample_integrates_to_target_under_perfect_velocity():
    """If the velocity field is constant ``x_1 - x_0``, integrating
    from ``x_0`` to ``t=1`` lands exactly on ``x_1``."""
    from lidar_anchored_depth.models import RectifiedFlowMatcher

    fm = RectifiedFlowMatcher(sigma=0.0)
    x_0 = torch.zeros(1, 1, 4, 4)
    x_1 = torch.full((1, 1, 4, 4), 3.0)

    def perfect_velocity(x, t, c):
        return x_1 - x_0  # constant for the linear path

    out = fm.sample(
        perfect_velocity, cond=torch.zeros(1, 0, 4, 4),
        n_steps=4, x_init=x_0,
    )
    assert torch.allclose(out, x_1, atol=1e-5)


@torch_required
def test_masked_cfm_loss_weight_at_valid_only():
    from lidar_anchored_depth.models import masked_cfm_loss

    v = torch.zeros(1, 1, 2, 2)
    u = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)
    loss = masked_cfm_loss(v, u, valid, valid_weight=1.0, invalid_weight=0.0)
    # Only the (0, 0) entry contributes: diff² = 4. Mean over only that
    # weighted entry = 4.
    assert torch.allclose(loss, torch.tensor(4.0))


@torch_required
def test_masked_cfm_loss_weighted_combination():
    from lidar_anchored_depth.models import masked_cfm_loss

    v = torch.zeros(1, 1, 2, 2)
    u = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    valid = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)
    loss = masked_cfm_loss(v, u, valid, valid_weight=1.0, invalid_weight=0.5)
    # weights = [1, 0.5, 0.5, 0.5]; sum = 2.5
    # weighted diff² = [1, 0.5, 0.5, 0.5].sum() = 2.5
    # loss = 2.5 / 2.5 = 1.0
    assert torch.allclose(loss, torch.tensor(1.0))


@torch_required
def test_residual_unet_forward_shape():
    from lidar_anchored_depth.models import COND_CHANNELS, ResidualVelocityUNet

    net = ResidualVelocityUNet(cond_channels=COND_CHANNELS, base=8)
    x_t = torch.randn(2, 1, 32, 32)
    t = torch.rand(2)
    cond = torch.randn(2, COND_CHANNELS, 32, 32)
    out = net(x_t, t, cond)
    assert out.shape == (2, 1, 32, 32)


@torch_required
def test_residual_unet_handles_non_square_inputs():
    from lidar_anchored_depth.models import COND_CHANNELS, ResidualVelocityUNet

    net = ResidualVelocityUNet(cond_channels=COND_CHANNELS, base=8)
    x_t = torch.randn(1, 1, 32, 64)
    t = torch.zeros(1)
    cond = torch.randn(1, COND_CHANNELS, 32, 64)
    out = net(x_t, t, cond)
    assert out.shape == (1, 1, 32, 64)


@torch_required
def test_residual_unet_param_count_in_expected_range():
    from lidar_anchored_depth.models import COND_CHANNELS, ResidualVelocityUNet

    net = ResidualVelocityUNet(cond_channels=COND_CHANNELS, base=32)
    n_params = sum(p.numel() for p in net.parameters())
    # Sanity: should be roughly 3-7M for base=32
    assert 1_000_000 < n_params < 20_000_000
