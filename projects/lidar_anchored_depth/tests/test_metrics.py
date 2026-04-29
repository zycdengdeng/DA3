"""Smoke tests for depth metrics — these are real numerics (not stubs)."""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.eval.metrics import (
    abs_rel,
    delta_threshold,
    rmse,
    rmse_log,
)


def test_perfect_prediction_is_zero_error():
    gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    pred = gt.copy()
    assert abs_rel(pred, gt) == 0.0
    assert rmse(pred, gt) == 0.0
    assert rmse_log(pred, gt) == 0.0
    assert delta_threshold(pred, gt) == 1.0


def test_constant_factor_two_known_metrics():
    gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pred = gt * 2.0  # 100% over-estimate
    assert abs_rel(pred, gt) == 1.0
    # delta < 1.25 fails for ratio 2.0
    assert delta_threshold(pred, gt, t=1.25) == 0.0
    # delta < 1.25^3 = 1.953... still fails (ratio is 2.0)
    assert delta_threshold(pred, gt, t=1.25**3) == 0.0


def test_invalid_pixels_excluded():
    gt = np.array([10.0, 0.0, np.nan, 20.0], dtype=np.float32)
    pred = np.array([10.0, 5.0, 5.0, 22.0], dtype=np.float32)
    # only indices 0 and 3 are valid
    expected = (abs(10 - 10) / 10 + abs(22 - 20) / 20) / 2
    np.testing.assert_allclose(abs_rel(pred, gt), expected)


def test_all_invalid_returns_nan():
    gt = np.zeros(5, dtype=np.float32)
    pred = np.ones(5, dtype=np.float32)
    assert np.isnan(abs_rel(pred, gt))
    assert np.isnan(rmse(pred, gt))
