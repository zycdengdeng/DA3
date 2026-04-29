"""Tests for ``alignment.region_affine`` (B4)."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.region_affine import (
    apply_region_fits,
    region_affine_align,
)


def _make_three_masks(H: int = 64, W: int = 64) -> np.ndarray:
    """Three non-overlapping masks of distinct shapes."""
    m = np.zeros((3, H, W), dtype=bool)
    m[0, 5:30, 5:30] = True   # square
    m[1, 35:55, 35:55] = True
    m[2, 5:25, 35:55] = True
    return m


def test_recovers_per_region_affines():
    """Three masks each with their own (a, b)."""
    H, W = 64, 64
    masks = _make_three_masks(H, W)
    rng = np.random.default_rng(0)

    a_b_per_mask = [(2.0, 0.5), (3.5, -0.3), (1.2, 1.0)]
    pixel_uv_list = []
    d_list = []
    z_list = []
    for k, (a, b) in enumerate(a_b_per_mask):
        rows, cols = np.where(masks[k])
        idx = rng.choice(len(rows), 50, replace=False)
        u = cols[idx]
        v = rows[idx]
        d = rng.uniform(0.5, 5.0, 50)
        z = a * d + b + rng.normal(0, 0.01, 50)
        pixel_uv_list.append(np.stack([u, v], axis=1))
        d_list.append(d)
        z_list.append(z)

    pixel_uv = np.concatenate(pixel_uv_list, axis=0)
    d_pred = np.concatenate(d_list)
    z_lidar = np.concatenate(z_list)

    res = region_affine_align(
        d_pred, z_lidar, pixel_uv, masks,
        min_lidar_per_region=5, use_ransac=False,
    )
    assert res.n_regions_attempted == 3
    assert res.n_regions_solved == 3
    for k, (a, b) in enumerate(a_b_per_mask):
        assert k in res.fits
        assert res.fits[k].a == pytest.approx(a, abs=0.05)
        assert res.fits[k].b == pytest.approx(b, abs=0.1)


def test_falls_back_for_low_count_regions():
    """Mask with too few LiDAR samples → fallback path."""
    H, W = 64, 64
    masks = _make_three_masks(H, W)
    rng = np.random.default_rng(1)

    pixel_uv_list = []
    d_list = []
    z_list = []
    a_b = (2.0, 0.5)
    # Only mask 0 gets data
    rows, cols = np.where(masks[0])
    idx = rng.choice(len(rows), 50, replace=False)
    u = cols[idx]
    v = rows[idx]
    d = rng.uniform(0.5, 5.0, 50)
    z = a_b[0] * d + a_b[1] + rng.normal(0, 0.01, 50)
    pixel_uv_list.append(np.stack([u, v], axis=1))
    d_list.append(d)
    z_list.append(z)
    pixel_uv = np.concatenate(pixel_uv_list, axis=0)
    d_pred = np.concatenate(d_list)
    z_lidar = np.concatenate(z_list)

    res = region_affine_align(
        d_pred, z_lidar, pixel_uv, masks,
        min_lidar_per_region=5, use_ransac=False,
    )
    assert 0 in res.fits
    assert 1 not in res.fits
    assert 2 not in res.fits
    assert res.n_regions_fallback == 2
    assert res.fallback_global is not None


def test_apply_region_fits_to_image():
    H, W = 64, 64
    masks = _make_three_masks(H, W)
    rng = np.random.default_rng(2)
    a_b_per_mask = [(2.0, 0.5), (3.5, -0.3), (1.2, 1.0)]

    pixel_uv_list = []
    d_list = []
    z_list = []
    for k, (a, b) in enumerate(a_b_per_mask):
        rows, cols = np.where(masks[k])
        idx = rng.choice(len(rows), 100, replace=False)
        u = cols[idx]
        v = rows[idx]
        d = rng.uniform(0.5, 5.0, 100)
        z = a * d + b
        pixel_uv_list.append(np.stack([u, v], axis=1))
        d_list.append(d)
        z_list.append(z)
    pixel_uv = np.concatenate(pixel_uv_list, axis=0)
    d_pred = np.concatenate(d_list)
    z_lidar = np.concatenate(z_list)

    res = region_affine_align(
        d_pred, z_lidar, pixel_uv, masks,
        min_lidar_per_region=5, use_ransac=False,
    )

    # Apply to a constant d̃ image
    d_image = np.full((H, W), 2.0, dtype=np.float32)
    out = apply_region_fits(d_image, masks, res, fill_value=-1.0)
    # Mask 0: z = 2.0 * 2.0 + 0.5 = 4.5
    expected_0 = 2.0 * 2.0 + 0.5
    assert np.allclose(out[masks[0]], expected_0, atol=1e-3)
    expected_1 = 3.5 * 2.0 + (-0.3)
    assert np.allclose(out[masks[1]], expected_1, atol=1e-3)
    # Outside any mask: covered by fallback (because everything got fit)
    not_covered = ~(masks[0] | masks[1] | masks[2])
    assert np.all(out[not_covered] != -1.0)
