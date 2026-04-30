"""Tests for the per-image-grid affine calibration wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.pipeline.region_static_calib import (
    build_grid_masks,
    fit_grid_static_calib,
)


def test_build_grid_masks_partition_covers_image_exactly_once():
    masks = build_grid_masks((30, 40), n_rows=3, n_cols=4)
    assert masks.shape == (12, 30, 40)
    # Every pixel covered by exactly one cell
    counts = masks.sum(axis=0)
    assert (counts == 1).all()


def test_build_grid_masks_row_major_indexing():
    masks = build_grid_masks((20, 20), n_rows=2, n_cols=2)
    # Cell 0 = top-left quadrant
    assert masks[0, 0, 0]
    assert not masks[0, 19, 19]
    # Cell 3 = bottom-right
    assert masks[3, 19, 19]
    assert not masks[3, 0, 0]


def test_build_grid_masks_rejects_zero_dims():
    with pytest.raises(ValueError):
        build_grid_masks((20, 20), n_rows=0, n_cols=2)
    with pytest.raises(ValueError):
        build_grid_masks((20, 20), n_rows=2, n_cols=0)


def test_fit_grid_static_calib_recovers_per_cell_affine():
    """Each grid cell has its own (a, b); sample points uniformly per
    cell and verify the fit recovers the per-cell coefficients."""
    H, W = 30, 40
    n_rows, n_cols = 3, 4

    rng = np.random.default_rng(42)
    d_pieces: list[np.ndarray] = []
    z_pieces: list[np.ndarray] = []
    uv_pieces: list[np.ndarray] = []
    true_ab: dict[int, tuple[float, float]] = {}
    masks = build_grid_masks((H, W), n_rows, n_cols)
    for k in range(n_rows * n_cols):
        a_k = 1.0 + 0.1 * k
        b_k = 0.5 * k
        true_ab[k] = (a_k, b_k)

        rows, cols = np.where(masks[k])
        sel = rng.choice(rows.size, size=200, replace=True)
        u = cols[sel].astype(np.float64)
        v = rows[sel].astype(np.float64)
        d = rng.uniform(1.0, 5.0, size=200)
        z = a_k * d + b_k + rng.normal(0.0, 0.01, size=200)
        d_pieces.append(d)
        z_pieces.append(z)
        uv_pieces.append(np.stack([u, v], axis=1))

    d_full = np.concatenate(d_pieces)
    z_full = np.concatenate(z_pieces)
    uv_full = np.concatenate(uv_pieces)

    calib = fit_grid_static_calib(
        d_full, z_full, uv_full, (H, W),
        n_rows=n_rows, n_cols=n_cols,
        min_lidar_per_cell=20,
    )
    assert calib.n_cells_solved == n_rows * n_cols
    for k, (a_true, b_true) in true_ab.items():
        fit = calib.result.fits[k]
        assert abs(fit.a - a_true) < 0.05
        assert abs(fit.b - b_true) < 0.1


def test_fit_grid_static_calib_apply_per_pixel():
    """``apply`` should map each pixel through its own cell's (a, b)."""
    H, W = 20, 20
    n_rows, n_cols = 2, 2
    masks = build_grid_masks((H, W), n_rows, n_cols)

    rng = np.random.default_rng(0)
    d_pieces: list[np.ndarray] = []
    z_pieces: list[np.ndarray] = []
    uv_pieces: list[np.ndarray] = []
    true_ab = [(1.0, 0.0), (2.0, 0.0), (1.0, 5.0), (2.0, 5.0)]
    for k, (a_k, b_k) in enumerate(true_ab):
        rows, cols = np.where(masks[k])
        sel = rng.choice(rows.size, size=100, replace=True)
        u = cols[sel].astype(np.float64)
        v = rows[sel].astype(np.float64)
        d = rng.uniform(1.0, 5.0, size=100)
        z = a_k * d + b_k
        d_pieces.append(d)
        z_pieces.append(z)
        uv_pieces.append(np.stack([u, v], axis=1))

    calib = fit_grid_static_calib(
        np.concatenate(d_pieces),
        np.concatenate(z_pieces),
        np.concatenate(uv_pieces),
        (H, W), n_rows=n_rows, n_cols=n_cols,
        min_lidar_per_cell=10,
    )

    d_image = np.full((H, W), 3.0, dtype=np.float64)
    z_image = calib.apply(d_image)

    # Top-left quadrant should be 1.0 * 3.0 + 0.0 = 3.0
    assert abs(float(z_image[0, 0]) - 3.0) < 0.1
    # Top-right quadrant should be 2.0 * 3.0 + 0.0 = 6.0
    assert abs(float(z_image[0, 19]) - 6.0) < 0.1
    # Bottom-right should be 2.0 * 3.0 + 5.0 = 11.0
    assert abs(float(z_image[19, 19]) - 11.0) < 0.2


def test_fit_grid_static_calib_falls_back_for_sparse_cells():
    """Cells with too few samples should fall back to the global fit
    rather than producing wild per-cell estimates."""
    H, W = 20, 20
    n_rows, n_cols = 2, 2
    masks = build_grid_masks((H, W), n_rows, n_cols)
    rng = np.random.default_rng(0)

    d_pieces, z_pieces, uv_pieces = [], [], []
    # Cells 0..2 get plenty of points; cell 3 gets 2.
    for k, n in enumerate([100, 100, 100, 2]):
        rows, cols = np.where(masks[k])
        sel = rng.choice(rows.size, size=n, replace=True)
        u = cols[sel].astype(np.float64)
        v = rows[sel].astype(np.float64)
        d = rng.uniform(1.0, 5.0, size=n)
        z = 2.0 * d + 1.0
        d_pieces.append(d)
        z_pieces.append(z)
        uv_pieces.append(np.stack([u, v], axis=1))

    calib = fit_grid_static_calib(
        np.concatenate(d_pieces),
        np.concatenate(z_pieces),
        np.concatenate(uv_pieces),
        (H, W), n_rows=n_rows, n_cols=n_cols,
        min_lidar_per_cell=10,
    )
    assert calib.n_cells_solved == 3
    # Cell 3 should not have its own fit; pixels in it use the fallback.
    assert 3 not in calib.result.fits
    assert calib.fallback() is not None


def test_to_serialisable_round_trip_keys():
    H, W = 10, 10
    rng = np.random.default_rng(0)
    n = 200
    rows = rng.integers(0, H, size=n)
    cols = rng.integers(0, W, size=n)
    d = rng.uniform(1.0, 5.0, size=n)
    z = 1.5 * d + 0.5
    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)

    calib = fit_grid_static_calib(
        d, z, uv, (H, W), n_rows=2, n_cols=2, min_lidar_per_cell=10,
    )
    s = calib.to_serialisable()
    assert s["n_rows"] == 2 and s["n_cols"] == 2
    assert s["image_hw"] == [H, W]
    for cell in s["cells"]:
        assert {"k", "row", "col", "a", "b", "rmse_m"} <= set(cell.keys())
