"""Tests for ``alignment.global_scale`` (B0–B3)."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.global_scale import (
    b0_identity,
    b1_median_scale,
    b2_lsq_affine,
    b3_ransac_affine,
)


# --------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------- #
def test_b0_identity_returns_unit_affine():
    fit = b0_identity(
        np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.8, 2.7]),
    )
    assert fit.a == 1.0
    assert fit.b == 0.0
    assert fit.n_inliers == 3


# --------------------------------------------------------------------- #
# B1
# --------------------------------------------------------------------- #
def test_b1_median_recovers_pure_scale():
    """If z = 5 · d̃ exactly, b1 should report a ≈ 5."""
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 5.0, 200)
    z = 5.0 * d
    fit = b1_median_scale(d, z)
    assert fit.a == pytest.approx(5.0, abs=1e-9)
    assert fit.b == 0.0


def test_b1_median_robust_to_outliers():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 5.0, 200)
    z = 3.0 * d
    # Inject 30% outliers
    bad = rng.choice(200, 60, replace=False)
    z[bad] = rng.uniform(0, 100, 60)
    fit = b1_median_scale(d, z)
    assert fit.a == pytest.approx(3.0, rel=0.05)


def test_b1_zero_d_raises():
    with pytest.raises(ValueError, match="no valid"):
        b1_median_scale(np.array([0.0, 0.0]), np.array([1.0, 2.0]))


# --------------------------------------------------------------------- #
# B2
# --------------------------------------------------------------------- #
def test_b2_recovers_known_affine():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 5.0, 200)
    a_true, b_true = 2.5, 0.7
    z = a_true * d + b_true
    fit = b2_lsq_affine(d, z)
    assert fit.a == pytest.approx(a_true, abs=1e-9)
    assert fit.b == pytest.approx(b_true, abs=1e-9)
    assert fit.residual_rmse < 1e-9


def test_b2_recovers_with_gaussian_noise():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 5.0, 1000)
    a_true, b_true = 3.0, 1.0
    z = a_true * d + b_true + rng.normal(0, 0.05, d.shape)
    fit = b2_lsq_affine(d, z)
    assert fit.a == pytest.approx(a_true, abs=0.02)
    assert fit.b == pytest.approx(b_true, abs=0.02)


def test_b2_too_few_samples():
    with pytest.raises(ValueError, match="need >="):
        b2_lsq_affine(np.array([1.0]), np.array([1.0]))


# --------------------------------------------------------------------- #
# B3
# --------------------------------------------------------------------- #
def test_b3_ransac_recovers_with_outliers():
    """30 % of samples are wild outliers; B3 should still nail (a, b)."""
    rng = np.random.default_rng(0)
    n = 500
    d = rng.uniform(0.5, 5.0, n)
    a_true, b_true = 2.0, 0.5
    z = a_true * d + b_true + rng.normal(0, 0.02, n)
    bad = rng.choice(n, n * 30 // 100, replace=False)
    z[bad] = rng.uniform(0, 50, len(bad))

    fit = b3_ransac_affine(
        d, z, threshold_rel=0.05, threshold_abs=0.3, n_iterations=300, seed=0,
    )
    assert fit.a == pytest.approx(a_true, abs=0.05)
    assert fit.b == pytest.approx(b_true, abs=0.1)
    # Inliers should be most of the clean samples
    assert fit.n_inliers >= int(0.6 * n)


def test_b3_apply_produces_metric():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 5.0, 100)
    z = 4.0 * d - 0.3
    fit = b3_ransac_affine(d, z, n_iterations=100, seed=0)
    z_pred = fit.apply(d)
    np.testing.assert_allclose(z_pred, z, atol=0.05)
