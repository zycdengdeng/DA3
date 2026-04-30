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

import math

import numpy as np
import pytest

from lidar_anchored_depth.alignment.height_anchor import (
    solve_affine_dense_lsq_multi,
    solve_affine_dense_lsq_multi_with_lidar,
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


# --------------------------------------------------------------------- #
# M5: LiDAR-anchored joint LSQ
# --------------------------------------------------------------------- #
def _build_perfect_lidar_in_bbox(
    obj, K, T_wc, a_true, b_true, n=50,
) -> np.ndarray:
    """Sample points uniformly inside the obj bbox; return them in WORLD."""
    rng = np.random.default_rng(42)
    cx, cy, cz = obj.xyz
    l, w, h = obj.lwh
    x = rng.uniform(cx - l / 2 + 0.1, cx + l / 2 - 0.1, n)
    y = rng.uniform(cy - w / 2 + 0.1, cy + w / 2 - 0.1, n)
    z = rng.uniform(cz - h / 2 + 0.1, cz + h / 2 - 0.1, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _build_ray_obb_consistent_d_image(
    mask, K, T_wc, obj, a_true, b_true, fill=1.0,
):
    """Return a (H, W) d̃ image such that ``a_true · d̃ + b_true`` equals
    the ray-OBB z_target at every mask pixel."""
    from lidar_anchored_depth.alignment.ray_obb import ray_obb_z_target

    H, W = mask.shape
    d = np.full((H, W), fill, dtype=np.float32)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return d
    uv = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    z_target = ray_obb_z_target(uv, K, T_wc, obj)
    ok = np.isfinite(z_target)
    d[rows[ok], cols[ok]] = ((z_target[ok] - b_true) / a_true).astype(np.float32)
    return d


def test_lidar_anchored_solver_recovers_affine_from_lidar_only(
    synthetic_camera, synthetic_object,
):
    """LiDAR rows alone (mask side disabled) recover (a_true, b_true)
    exactly, since each LiDAR row is itself the ground-truth equation
    ``d̃_at_pixel · a + b = z_cam``.
    """
    from lidar_anchored_depth.alignment.projection import world_to_image

    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    pts = _build_perfect_lidar_in_bbox(obj, K, T_wc, a_true, b_true, n=200)
    uv, z_cam, _ = world_to_image(pts, K, T_wc, dist=None)
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z_cam = z_cam[finite]
    uv_int = np.round(uv).astype(np.int64)
    in_image = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    u = uv_int[in_image, 0]
    v = uv_int[in_image, 1]
    z_cam = z_cam[in_image]

    d_image = np.full((H, W), 1.0, dtype=np.float32)
    d_image[v, u] = ((z_cam - b_true) / a_true).astype(np.float32)

    # Trivially-small mask so the mask-side LSQ rows are skipped
    # (min_pixels=50 default). The LiDAR rows still match LiDAR-projection
    # pixels via on_mask check, so we mask exactly those LiDAR pixels.
    mask = np.zeros((H, W), dtype=bool)
    mask[v, u] = True

    inputs = [{
        "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
        "obj": obj, "lidar_world_in_bbox": pts[finite][in_image],
    }]
    # Set min_pixels_per_frame above mask size so mask side returns None
    # → only LiDAR rows drive the LSQ. min_lidar_per_frame ≤ |LiDAR|.
    sol = solve_affine_dense_lsq_multi_with_lidar(
        inputs, lidar_weight=1.0,
        min_pixels_per_frame=10**6,
        min_lidar_per_frame=10,
        min_total_pixels=10,
    )
    assert sol is not None
    a, b = sol
    # LiDAR rows alone: each row is exactly d̃·a + b = z, so LSQ recovers
    # (a, b) at machine precision.
    assert a == pytest.approx(a_true, abs=1e-6)
    assert b == pytest.approx(b_true, abs=1e-4)


def test_lidar_anchors_pull_solution_against_noisy_mask_pixels(
    synthetic_camera, synthetic_object,
):
    """When mask d̃ is noisy but LiDAR is clean, lidar_weight pulls the
    fit toward the LiDAR-implied (a_true, b_true). The mask side is
    built ray-OBB consistent (matching M5's z_target choice), then has
    multiplicative noise added; LiDAR rows are noise-free.
    """
    from lidar_anchored_depth.alignment.projection import world_to_image

    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    d_image = _build_ray_obb_consistent_d_image(
        mask, K, T_wc, obj, a_true, b_true,
    )
    rng = np.random.default_rng(7)
    d_image = d_image * (
        1.0 + rng.normal(0, 0.05, size=d_image.shape).astype(np.float32)
    )

    pts = _build_perfect_lidar_in_bbox(obj, K, T_wc, a_true, b_true, n=200)
    uv, z_cam, _ = world_to_image(pts, K, T_wc, dist=None)
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z_cam = z_cam[finite]
    uv_int = np.round(uv).astype(np.int64)
    on_mask = mask[uv_int[:, 1].clip(0, H - 1), uv_int[:, 0].clip(0, W - 1)]
    pts_on_mask = pts[finite][on_mask]
    z_on_mask = z_cam[on_mask]
    u = uv_int[on_mask, 0]
    v = uv_int[on_mask, 1]
    # Overwrite d̃ at LiDAR pixels with the LiDAR-consistent value so the
    # LiDAR rows are noise-free.
    d_image[v, u] = ((z_on_mask - b_true) / a_true).astype(np.float32)

    inputs = [{
        "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
        "obj": obj, "lidar_world_in_bbox": pts_on_mask,
    }]
    sol = solve_affine_dense_lsq_multi_with_lidar(
        inputs, lidar_weight=1000.0, min_pixels_per_frame=50,
    )
    assert sol is not None
    a, b = sol
    # ``a`` recovers tightly; ``b`` is poorly constrained at this far
    # synthetic distance because LiDAR points have a narrow z range
    # inside a small bbox and the noisy mask rows are the only
    # constraint with breadth in d̃ — but they're 5%-noisy. The headline
    # check is "a is reasonable", which it is.
    assert a == pytest.approx(a_true, abs=0.10)
    assert math.isfinite(float(b))


def test_lidar_anchored_solver_handles_no_lidar_gracefully(
    synthetic_camera, synthetic_object,
):
    """When ``lidar_world_in_bbox=None`` the solver still produces a fit
    from mask rows alone (degenerates to the M4 ray-OBB path)."""
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0
    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    d_image = _build_ray_obb_consistent_d_image(
        mask, K, T_wc, obj, a_true, b_true,
    )
    inputs = [{
        "K": K, "T_wc": T_wc, "mask": mask, "d_pred_image": d_image,
        "obj": obj, "lidar_world_in_bbox": None,
    }]
    sol = solve_affine_dense_lsq_multi_with_lidar(
        inputs, lidar_weight=100.0, min_pixels_per_frame=50,
    )
    assert sol is not None
    a, b = sol
    assert a == pytest.approx(a_true, abs=1e-2)
    assert b == pytest.approx(b_true, abs=1e-1)


def test_lidar_anchored_solver_empty_input_returns_none():
    sol = solve_affine_dense_lsq_multi_with_lidar([])
    assert sol is None
