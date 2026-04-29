"""Pinhole projection geometry — pure NumPy core for all alignment modules.

Conventions
-----------
- World frame: right-handed, Z-up (= THICV-R2A VirtualLidar).
- Camera frame: OpenCV (X right, Y down, Z forward).
- ``T_wc``: camera-to-world (4×4 SE(3)). ``P_world = T_wc @ [P_cam; 1]``.

Distortion model: pinhole + 5-coef plumb-bob (k1, k2, p1, p2, k3) when a
``dist`` array is passed and ``cv2`` is available; otherwise a plain
``K @ P_cam / z`` projection is used. Fisheye is intentionally not
supported by this project.

This module is the canonical home for all geometry primitives shared
across the alignment / viz subpackages.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 needed for plumb-bob projection only
    import cv2  # type: ignore[import-not-found]

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.data.calibration import euler_zyx_to_R


# --------------------------------------------------------------------- #
# SE(3) helpers
# --------------------------------------------------------------------- #
def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) 4×4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def world_to_camera(points_world: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform ``(N, 3)`` world-frame points into the camera frame."""
    if points_world.size == 0:
        return points_world.reshape(0, 3).astype(np.float64)
    T_cw = invert_T(np.asarray(T_wc, dtype=np.float64))
    P = np.asarray(points_world, dtype=np.float64)
    P_h = np.hstack([P, np.ones((P.shape[0], 1), dtype=np.float64)])
    return (T_cw @ P_h.T).T[:, :3]


def camera_to_world(points_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform ``(N, 3)`` camera-frame points into the world frame."""
    if points_cam.size == 0:
        return points_cam.reshape(0, 3).astype(np.float64)
    P = np.asarray(points_cam, dtype=np.float64)
    T = np.asarray(T_wc, dtype=np.float64)
    P_h = np.hstack([P, np.ones((P.shape[0], 1), dtype=np.float64)])
    return (T @ P_h.T).T[:, :3]


# --------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------- #
def project_camera_to_image(
    points_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    z_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixel coordinates.

    Returns
    -------
    uv : (M, 2) projected pixel coords for points with ``z > z_min``.
    in_front : (N,) bool mask. ``M == in_front.sum()``.
    """
    P = np.asarray(points_cam, dtype=np.float64)
    if P.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    z = P[:, 2]
    in_front = z > z_min
    if not in_front.any():
        return np.zeros((0, 2), dtype=np.float64), in_front

    front = P[in_front]

    if dist is not None and len(dist) >= 4 and HAS_CV2:
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        if d.size == 4:
            d = np.concatenate([d, np.zeros(1, dtype=np.float64)])
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        uv, _ = cv2.projectPoints(
            front.reshape(-1, 1, 3),
            rvec, tvec,
            np.asarray(K, dtype=np.float64),
            d,
        )
        uv = uv.reshape(-1, 2)
    else:
        proj = (np.asarray(K, dtype=np.float64) @ front.T).T
        uv = proj[:, :2] / proj[:, 2:3]

    return uv, in_front


def world_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    z_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot world→image projection.

    Returns
    -------
    uv : (M, 2) pixel coords for points in front of the camera.
    z_cam : (M,) camera-frame depth of those points.
    in_front : (N,) bool mask aligned with the input.
    """
    P_cam = world_to_camera(points_world, T_wc)
    uv, in_front = project_camera_to_image(P_cam, K, dist, z_min=z_min)
    z_cam = P_cam[in_front, 2]
    return uv, z_cam, in_front


# --------------------------------------------------------------------- #
# Pixel ↔ ray
# --------------------------------------------------------------------- #
def pixel_to_camera_ray(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Inverse-K backproject pixels to camera-frame rays with ``z=1``.

    Returns ``(N, 3)`` directions ``(x_n, y_n, 1)``. Distortion is
    ignored (pass undistorted pixel coordinates if needed).
    """
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    uv1 = np.hstack([uv_arr, np.ones((uv_arr.shape[0], 1), dtype=np.float64)])
    rays = (K_inv @ uv1.T).T
    return rays


def pixel_ray_to_world(
    uv: np.ndarray, K: np.ndarray, T_wc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject pixels to world-frame rays.

    Returns
    -------
    origin : (3,) camera centre in world frame, shared by all rays.
    directions : (N, 3) world-frame ray directions (NOT unit-normalized;
        the third camera-frame component is fixed at 1).
    """
    rays_cam = pixel_to_camera_ray(uv, K)
    R = np.asarray(T_wc, dtype=np.float64)[:3, :3]
    origin = np.asarray(T_wc, dtype=np.float64)[:3, 3]
    dirs_world = (R @ rays_cam.T).T
    return origin, dirs_world


# --------------------------------------------------------------------- #
# Bbox geometry
# --------------------------------------------------------------------- #
BBOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
)


def bbox_3d_corners(
    cx: float,
    cy: float,
    cz: float,
    length: float,
    width: float,
    height: float,
    yaw: float,
    roll: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Return the 8 corners of an oriented 3D bbox in world coordinates.

    Bottom face = corners 0..3 (z = cz - h/2), top face = 4..7. Length
    along bbox-local X, width along Y, height along Z. Yaw rotates about
    world Z; roll about local X; pitch about local Y (ZYX intrinsic,
    matching the THICV-R2A annotation convention).
    """
    R = euler_zyx_to_R(roll, pitch, yaw)
    dx, dy, dz = length / 2.0, width / 2.0, height / 2.0
    corners_local = np.array(
        [
            [-dx, -dy, -dz],
            [+dx, -dy, -dz],
            [+dx, +dy, -dz],
            [-dx, +dy, -dz],
            [-dx, -dy, +dz],
            [+dx, -dy, +dz],
            [+dx, +dy, +dz],
            [-dx, +dy, +dz],
        ],
        dtype=np.float64,
    )
    return (R @ corners_local.T).T + np.array([cx, cy, cz], dtype=np.float64)


def bbox_corners_from_object(obj: DynamicObject) -> np.ndarray:
    """Convenience wrapper around :func:`bbox_3d_corners`."""
    return bbox_3d_corners(
        float(obj.xyz[0]), float(obj.xyz[1]), float(obj.xyz[2]),
        float(obj.lwh[0]), float(obj.lwh[1]), float(obj.lwh[2]),
        obj.yaw, obj.roll, obj.pitch,
    )


def project_3d_bbox(
    obj: DynamicObject,
    K: np.ndarray,
    T_wc: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    z_min: float = 0.1,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Project the 8 corners of a :class:`DynamicObject` to image.

    Returns
    -------
    uv8 : (8, 2) projected corners, or ``None`` if any corner lies
        behind the camera (partial-frustum clipping is intentionally
        not implemented here).
    P_cam8 : (8, 3) camera-frame corners (always returned).
    """
    corners_world = bbox_corners_from_object(obj)
    P_cam = world_to_camera(corners_world, T_wc)
    if not (P_cam[:, 2] > z_min).all():
        return None, P_cam
    uv, _ = project_camera_to_image(P_cam, K, dist, z_min=0.0)
    if uv.shape[0] != 8:
        return None, P_cam
    return uv, P_cam


def bbox_uv_aabb(uv8: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding rectangle ``(u_min, v_min, u_max, v_max)``."""
    return (
        float(uv8[:, 0].min()),
        float(uv8[:, 1].min()),
        float(uv8[:, 0].max()),
        float(uv8[:, 1].max()),
    )


# --------------------------------------------------------------------- #
# Height-anchored geometry primitives
# --------------------------------------------------------------------- #
def world_z_at_pixel_unit_depth(
    uv: np.ndarray, K: np.ndarray, T_wc: np.ndarray
) -> tuple[np.ndarray, float]:
    """Decompose ``Z_world(u, v, z_cam) = α(u,v) · z_cam + β`` for HAD.

    For a pixel ``(u, v)``, the world-Z component of a 3D point at
    camera depth ``z_cam`` is **linear** in ``z_cam``::

        P_cam   = z_cam · K^{-1} · [u, v, 1]^T   (with z=z_cam at the third comp)
        P_world = T_wc · [P_cam; 1]
        Z_world = α(u, v) · z_cam + β

    where:
        α(u, v) = T_wc[2,0] · x_n + T_wc[2,1] · y_n + T_wc[2,2]
        β       = T_wc[2, 3]

    (Here ``[x_n, y_n, 1]^T = K^{-1} · [u, v, 1]^T``.) This decomposition
    is the foundation of HAD's closed-form solver.

    Returns
    -------
    alpha : (N,) per-pixel slope of Z_world w.r.t. camera depth.
    beta  : scalar (independent of pixel).
    """
    rays = pixel_to_camera_ray(uv, K)  # (N, 3) with rays[:, 2] == 1
    T = np.asarray(T_wc, dtype=np.float64)
    alpha = T[2, 0] * rays[:, 0] + T[2, 1] * rays[:, 1] + T[2, 2] * rays[:, 2]
    beta = float(T[2, 3])
    return alpha, beta
