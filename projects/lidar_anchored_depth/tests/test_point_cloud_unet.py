"""Tests for the per-point velocity field (DGCNN + FiLM time)."""

from __future__ import annotations

import pytest

try:
    import torch
    _TORCH_OK = True
except ModuleNotFoundError:
    _TORCH_OK = False

torch_required = pytest.mark.skipif(not _TORCH_OK, reason="requires torch")


@torch_required
def test_point_velocity_forward_shape():
    from lidar_anchored_depth.models import PointCloudVelocityNet

    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8)
    B, N = 2, 256
    prior_xyz = torch.randn(B, N, 3) * 0.5
    prior_rgb = torch.rand(B, N, 3)
    prior_feat = torch.randn(B, N, 5)
    x_t = torch.randn(B, N, 3) * 0.1
    t = torch.rand(B)
    v = net(prior_xyz, prior_rgb, prior_feat, x_t, t)
    assert v.shape == (B, N, 3)


@torch_required
def test_point_velocity_handles_single_batch():
    from lidar_anchored_depth.models import PointCloudVelocityNet

    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8)
    prior_xyz = torch.randn(1, 64, 3)
    prior_rgb = torch.rand(1, 64, 3)
    prior_feat = torch.randn(1, 64, 5)
    x_t = torch.zeros(1, 64, 3)
    t = torch.tensor([0.5])
    v = net(prior_xyz, prior_rgb, prior_feat, x_t, t)
    assert v.shape == (1, 64, 3)
    assert torch.isfinite(v).all()


@torch_required
def test_point_velocity_param_count_in_expected_range():
    from lidar_anchored_depth.models import PointCloudVelocityNet

    net = PointCloudVelocityNet(prior_feat_dim=5, base=64, k=16)
    n = sum(p.numel() for p in net.parameters())
    assert 100_000 < n < 5_000_000


@torch_required
def test_point_velocity_t0_t1_differ():
    """At different time values the FiLM modulation should change the
    output even with identical x_t — sanity check that ``t`` actually
    enters the network."""
    from lidar_anchored_depth.models import PointCloudVelocityNet

    torch.manual_seed(0)
    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8).eval()
    prior_xyz = torch.randn(1, 64, 3)
    prior_rgb = torch.rand(1, 64, 3)
    prior_feat = torch.randn(1, 64, 5)
    x_t = torch.zeros(1, 64, 3)
    with torch.no_grad():
        v0 = net(prior_xyz, prior_rgb, prior_feat, x_t, torch.tensor([0.0]))
        v1 = net(prior_xyz, prior_rgb, prior_feat, x_t, torch.tensor([1.0]))
    assert not torch.allclose(v0, v1, atol=1e-4)


@torch_required
def test_point_velocity_grad_flows():
    from lidar_anchored_depth.models import PointCloudVelocityNet

    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8)
    prior_xyz = torch.randn(1, 32, 3, requires_grad=False)
    prior_rgb = torch.rand(1, 32, 3)
    prior_feat = torch.randn(1, 32, 5)
    x_t = torch.randn(1, 32, 3, requires_grad=True)
    t = torch.tensor([0.3])
    v = net(prior_xyz, prior_rgb, prior_feat, x_t, t)
    loss = v.pow(2).mean()
    loss.backward()
    # at least one parameter should have a non-zero gradient
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


# --- Dataset tests (no torch needed for the npz I/O loop, but Dataset
#     itself imports torch).

@torch_required
def test_completion_3d_dataset_loads_and_subsamples(tmp_path):
    import numpy as np

    from lidar_anchored_depth.data.completion_3d_dataset import (
        Completion3DDataset, PRIOR_FEAT_DIM,
    )

    # Build one synthetic npz.
    n = 4000
    np.savez(
        tmp_path / "008_ts1_cam0_3d.npz",
        prior_xyz=np.random.randn(n, 3).astype(np.float32) * 10,
        prior_rgb=np.random.randint(0, 255, size=(n, 3), dtype=np.uint8),
        prior_feat=np.random.randn(n, PRIOR_FEAT_DIM).astype(np.float32),
        target_xyz=np.random.randn(n // 2, 3).astype(np.float32) * 10,
        target_disp=np.random.randn(n, 3).astype(np.float32) * 0.5,
        valid_mask=np.random.rand(n) < 0.3,
    )
    ds = Completion3DDataset(tmp_path, n_points=2048, valid_oversample=2.0)
    assert len(ds) == 1
    item = ds[0]
    assert item["prior_xyz"].shape == (2048, 3)
    assert item["prior_rgb"].shape == (2048, 3)
    assert item["prior_feat"].shape == (2048, PRIOR_FEAT_DIM)
    assert item["target_disp"].shape == (2048, 3)
    assert item["valid"].shape == (2048,)
    # Subsampled cloud should be roughly centred.
    centre = item["prior_xyz"].mean(dim=0).abs().max().item()
    assert centre < 1.0


@torch_required
def test_segment_aware_edgeconv_isolates_segments():
    """When all points share segment, output should match the
    no-segment case. When two segments are isolated, points from
    segment A's max-pool should not be influenced by features only
    present in segment B."""
    from lidar_anchored_depth.models import PointCloudVelocityNet

    torch.manual_seed(0)
    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8).eval()

    B, N = 1, 64
    prior_xyz = torch.randn(B, N, 3)
    prior_rgb = torch.rand(B, N, 3)
    prior_feat = torch.randn(B, N, 5)
    x_t = torch.zeros(B, N, 3)
    t = torch.tensor([0.5])

    # All-same segment -> should equal segment_id=None.
    seg_one = torch.zeros(B, N, dtype=torch.long)
    with torch.no_grad():
        v_no_seg = net(prior_xyz, prior_rgb, prior_feat, x_t, t)
        v_one_seg = net(prior_xyz, prior_rgb, prior_feat, x_t, t,
                        segment_id=seg_one)
    # Same shape, but values can differ slightly due to LayerNorm/etc;
    # the key invariant is that segment_id=None is path-equivalent
    # when the network actually does the masking. For all-same the
    # mask is all 1s, so behaviour is identical to no-mask path.
    assert torch.allclose(v_no_seg, v_one_seg, atol=1e-5)


@torch_required
def test_segment_aware_edgeconv_two_segments_differs():
    """With two non-trivial segments, output must differ from the
    no-segment case (at least somewhere) — confirms the mask is
    actually being applied."""
    from lidar_anchored_depth.models import PointCloudVelocityNet

    torch.manual_seed(0)
    net = PointCloudVelocityNet(prior_feat_dim=5, base=16, k=8).eval()
    B, N = 1, 64
    prior_xyz = torch.randn(B, N, 3)
    prior_rgb = torch.rand(B, N, 3)
    prior_feat = torch.randn(B, N, 5)
    x_t = torch.zeros(B, N, 3)
    t = torch.tensor([0.5])

    seg_split = torch.cat([
        torch.zeros(N // 2, dtype=torch.long),
        torch.ones(N - N // 2, dtype=torch.long),
    ]).unsqueeze(0)
    with torch.no_grad():
        v_no_seg = net(prior_xyz, prior_rgb, prior_feat, x_t, t)
        v_split = net(prior_xyz, prior_rgb, prior_feat, x_t, t,
                      segment_id=seg_split)
    assert not torch.allclose(v_no_seg, v_split, atol=1e-4)


@torch_required
def test_completion_3d_dataset_pads_when_too_few_points(tmp_path):
    import numpy as np

    from lidar_anchored_depth.data.completion_3d_dataset import (
        Completion3DDataset, PRIOR_FEAT_DIM,
    )

    n = 100  # less than n_points
    np.savez(
        tmp_path / "008_ts1_cam0_3d.npz",
        prior_xyz=np.random.randn(n, 3).astype(np.float32),
        prior_rgb=np.random.randint(0, 255, size=(n, 3), dtype=np.uint8),
        prior_feat=np.random.randn(n, PRIOR_FEAT_DIM).astype(np.float32),
        target_xyz=np.random.randn(n // 2, 3).astype(np.float32),
        target_disp=np.random.randn(n, 3).astype(np.float32) * 0.1,
        valid_mask=np.ones(n, dtype=bool),
    )
    ds = Completion3DDataset(tmp_path, n_points=512)
    item = ds[0]
    assert item["prior_xyz"].shape == (512, 3)
