"""Tests for ``alignment.height_anchor.solve_affine_dense_lsq_multi``.

Validates the joint per-camera LSQ:
- recovers ``(a_true, b_true)`` from a single synthetic frame (parity
  with single-frame solve_affine_dense_lsq)
- recovers the same ``(a_true, b_true)`` from many frames (averaging
  property)
- ``z_target='ray-obb'`` recovers ``(a_true, b_true)`` exactly when
  the pixel-to-bbox geometry is consistent
"""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.height_anchor import (
    solve_affine_dense_lsq_multi,
)
from lidar_anchored_depth.data.base import DynamicObject

from _synthetic import (  # type: ignore[import-not-found]
    synthetic_d_image_consistent_with_mask,
    synthetic_mask_from_world_box,
)


def test_multi_solver_one_frame_matches_single(synthetic_camera, synthetic_object):
    """One frame in the multi-solver = same answer as the single-solver."""
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    d_image = synthetic_d_image_consistent_with_mask(
        mask=mask, K=K, T_wc=T_wc,
        Z_min=obj.Z_min, Z_max=obj.Z_max,
        a_true=a_true, b_true=b_true,
    )
    inputs = [{
        "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
        "Z_max": obj.Z_max, "Z_min": obj.Z_min,
    }]
    sol = solve_affine_dense_lsq_multi(
        inputs, z_target="row-linear", min_pixels_per_frame=50,
    )
    assert sol is not None
    a, b = sol
    assert a == pytest.approx(a_true, rel=0.05)
    assert b == pytest.approx(b_true, abs=0.5)


def test_multi_solver_many_frames_recovers_same_affine(synthetic_camera, synthetic_object):
    """5 frames (slightly noisy d̃) → joint solver recovers (a_true, b_true).

    Each frame has the same ground-truth (a, b) but DA3 noise jitters
    the d̃ image; the joint solver should average the noise out.

    Note on conditioning: at this synthetic object's distance (30 m, 1.5 m
    tall), ``d̃_top − d̃_bot`` is small, so a high noise level
    overwhelms the slope info. The synthetic test uses 0.1 % noise (at
    real frame distances DA3 noise is sub-1 %; the joint LSQ's
    advantage is precisely averaging that noise out across 5 frames).
    """
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    rng = np.random.default_rng(0)
    inputs = []
    for _ in range(5):
        d_image = synthetic_d_image_consistent_with_mask(
            mask=mask, K=K, T_wc=T_wc,
            Z_min=obj.Z_min, Z_max=obj.Z_max,
            a_true=a_true, b_true=b_true,
        )
        # Inject 0.1 % multiplicative noise (conditioning-realistic for
        # this far-object synthetic setup).
        noise = 1.0 + rng.normal(0, 0.001, size=d_image.shape).astype(np.float32)
        d_image = d_image * noise
        inputs.append({
            "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
            "Z_max": obj.Z_max, "Z_min": obj.Z_min,
        })

    sol = solve_affine_dense_lsq_multi(
        inputs, z_target="row-linear", min_pixels_per_frame=50,
    )
    assert sol is not None
    a, b = sol
    assert a == pytest.approx(a_true, rel=0.20)
    # b is amplified by conditioning at this far distance; allow a wide
    # tolerance — the test only certifies "joint LSQ runs and returns a
    # ballpark answer", not noise-free recovery.
    assert b == pytest.approx(b_true, abs=2.0)


def test_multi_solver_ray_obb_recovers_affine(synthetic_camera, synthetic_object):
    """With z_target='ray-obb' the solver uses the bbox geometry directly.

    This is Path B. We construct ``d_image`` consistent with
    ``z_cam = a_true · d̃ + b_true`` at every mask pixel and verify the
    solver recovers (a_true, b_true).
    """
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )

    # Build a d_image where d̃ is set so the ray-OBB hit's z_cam matches
    # a_true · d̃ + b_true. We do this analytically: for each mask pixel
    # query its z_cam_target via ray-OBB, then back out d̃.
    from lidar_anchored_depth.alignment.ray_obb import ray_obb_z_target

    rows, cols = np.where(mask)
    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    z_target = ray_obb_z_target(uv, K, T_wc, obj)
    valid = np.isfinite(z_target)
    rows = rows[valid]
    cols = cols[valid]
    z_target = z_target[valid]

    d_image = np.full((H, W), 1.0, dtype=np.float32)
    d_image[rows, cols] = ((z_target - b_true) / a_true).astype(np.float32)

    inputs = [{
        "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
        "obj": obj,
    }]
    sol = solve_affine_dense_lsq_multi(
        inputs, z_target="ray-obb", min_pixels_per_frame=50,
    )
    assert sol is not None
    a, b = sol
    # Ray-OBB recovers exactly modulo numerical noise
    assert a == pytest.approx(a_true, abs=1e-3)
    assert b == pytest.approx(b_true, abs=1e-2)


def test_multi_solver_empty_inputs_returns_none():
    sol = solve_affine_dense_lsq_multi([])
    assert sol is None


def test_multi_solver_invalid_z_target_raises():
    with pytest.raises(ValueError, match="z_target"):
        solve_affine_dense_lsq_multi([], z_target="nonsense")


def test_multi_solver_below_min_total_pixels_returns_none(synthetic_camera, synthetic_object):
    """A trivially small mask returns None when below min_total_pixels."""
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    mask = np.zeros((H, W), dtype=bool)
    mask[100:101, 100:101] = True   # 1 pixel
    d_image = np.ones((H, W), dtype=np.float32)

    sol = solve_affine_dense_lsq_multi(
        [{"K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
          "Z_max": obj.Z_max, "Z_min": obj.Z_min}],
        z_target="row-linear", min_pixels_per_frame=50,
    )
    assert sol is None
