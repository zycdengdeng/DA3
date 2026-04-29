"""Pytest fixtures shared across the suite.

A small synthetic roadside scene (camera at 6 m height pitched 25° down,
LiDAR points on the ground + on a "car" stand-in) supports unit tests
for projection, ground-plane fit, and HAD across modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.data.base import DynamicObject, Frame


@pytest.fixture
def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    """A pinhole camera at world (0, 0, 6) m looking forward and down ~25°.

    Returns
    -------
    K, T_wc : (3, 3) and (4, 4) float64.
    """
    fx = fy = 1500.0
    cx, cy = 960.0, 540.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Camera at world (0, 0, 6), pitched 25° down (rotation about world X).
    # Camera frame = OpenCV (X right, Y down, Z forward).
    # World frame = right-handed, Z up. World forward (the direction the
    # camera points along its Z axis, projected to ground) is +Y in world.
    pitch_deg = 25.0
    th = np.deg2rad(pitch_deg)

    # T_wc rotation, columns = camera-frame basis vectors expressed in world:
    #   camera X (right)   →  ( 1,        0,        0)
    #   camera Y (down)    →  ( 0, -sin θ,  -cos θ)   (down + slightly back)
    #   camera Z (forward) →  ( 0,  cos θ,  -sin θ)   (forward + slightly down)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -np.sin(th), np.cos(th)],
            [0.0, -np.cos(th), -np.sin(th)],
        ],
        dtype=np.float64,
    )
    t = np.array([0.0, 0.0, 6.0], dtype=np.float64)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = t
    return K, T_wc


@pytest.fixture
def synthetic_frame(synthetic_camera) -> Frame:
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # A scattering of ground points (Z = 0) and one "car" at horizontal
    # distance 30 m, 1.5 m tall (Z in [0, 1.5]).
    rng = np.random.default_rng(0)
    ground = np.stack(
        [
            rng.uniform(-15.0, 15.0, 2000),
            rng.uniform(5.0, 80.0, 2000),
            np.zeros(2000),
        ],
        axis=1,
    )
    car_x = rng.uniform(-1.0, 1.0, 200)
    car_y = rng.uniform(29.0, 31.0, 200)
    car_z = rng.uniform(0.0, 1.5, 200)
    car = np.stack([car_x, car_y, car_z], axis=1)

    lidar = np.concatenate([ground, car], axis=0).astype(np.float32)

    return Frame(
        frame_id="synthetic_0",
        image=image,
        K=K,
        T_wc=T_wc,
        lidar_world=lidar,
        gt_depth=None,
        sam_masks=None,
        meta={"is_synthetic": True},
    )


# --------------------------------------------------------------------- #
# Building blocks for HAD / AA-HAD tests
#   (helper functions live in tests/_synthetic.py)
# --------------------------------------------------------------------- #
@pytest.fixture
def synthetic_object() -> DynamicObject:
    """A 4 m × 2 m × 1.5 m "car" centred at world (0, 30, 0.75)."""
    return DynamicObject(
        id=1,
        label="Car",
        xyz=np.array([0.0, 30.0, 0.75], dtype=np.float64),
        lwh=np.array([4.0, 2.0, 1.5], dtype=np.float64),
        yaw=0.0,
    )


@pytest.fixture
def synthetic_object_lidar(synthetic_object) -> np.ndarray:
    """A dense LiDAR cloud on the surfaces of synthetic_object."""
    rng = np.random.default_rng(42)
    n = 500
    obj = synthetic_object
    # Sample uniformly inside the bbox volume
    cx, cy, cz = obj.xyz
    l, w, h = obj.lwh
    x = rng.uniform(cx - l / 2, cx + l / 2, n)
    y = rng.uniform(cy - w / 2, cy + w / 2, n)
    z = rng.uniform(cz - h / 2, cz + h / 2, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)
