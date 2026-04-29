"""Project world-frame geometry onto images and draw debug overlays.

Used by ``scripts/preview_frame.py`` and (later) any frame-level
visualization in evaluation reports.

Geometry conventions
--------------------
- World frame: right-handed, Z-up (= THICV-R2A VirtualLidar).
- Camera frame: OpenCV (X right, Y down, Z forward).
- ``T_wc``: camera-to-world (4×4 SE(3)). ``P_world = T_wc @ [P_cam; 1]``.

Projection model
----------------
Pinhole + plumb-bob distortion (cv2.projectPoints) when a 5-coef
distortion vector is provided; otherwise a plain ``K @ P_cam / z``.
Fisheye is **not** supported here (the project ignores fisheye streams).
"""

from __future__ import annotations

import numpy as np

try:  # cv2 is required for distortion-aware projection and drawing
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.data.calibration import euler_zyx_to_R


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #
def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def world_to_camera(points_world: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform ``(N, 3)`` world-frame points into the camera frame.

    Parameters
    ----------
    points_world : (N, 3) array
    T_wc : (4, 4) SE(3); camera-to-world.

    Returns
    -------
    points_cam : (N, 3) array, camera frame.
    """
    if points_world.size == 0:
        return points_world.reshape(0, 3).astype(np.float64)
    T_cw = _invert_T(np.asarray(T_wc, dtype=np.float64))
    P = np.asarray(points_world, dtype=np.float64)
    P_h = np.hstack([P, np.ones((P.shape[0], 1), dtype=np.float64)])
    return (T_cw @ P_h.T).T[:, :3]


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
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        # plumb-bob: cv2.projectPoints expects 5 coefs (k1, k2, p1, p2, k3)
        if d.size == 4:
            d = np.concatenate([d, np.zeros(1, dtype=np.float64)])
        uv, _ = cv2.projectPoints(
            front.reshape(-1, 1, 3),
            rvec,
            tvec,
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

    Bottom face = corners 0..3 (z = cz - h/2), top face = 4..7. Length is
    along the bbox-local X axis, width along Y, height along Z. Yaw
    rotates about world Z; roll about local X; pitch about local Y
    (ZYX intrinsic, matching the THICV-R2A annotation convention).
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
        float(obj.xyz[0]),
        float(obj.xyz[1]),
        float(obj.xyz[2]),
        float(obj.lwh[0]),
        float(obj.lwh[1]),
        float(obj.lwh[2]),
        obj.yaw,
        obj.roll,
        obj.pitch,
    )


# --------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------- #
def draw_lidar_overlay(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    radius: int = 1,
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> np.ndarray:
    """Color LiDAR points by depth (jet) and draw onto ``image``.

    ``depth_min`` / ``depth_max`` clip the colormap range; default uses
    the per-frame min / max. Image is treated as BGR (cv2 convention).

    Returns a copy of the input image with points drawn.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for draw_lidar_overlay")

    out = image.copy()
    if uv.shape[0] == 0:
        return out

    h, w = out.shape[:2]
    d = np.asarray(depth, dtype=np.float64)
    lo = float(d.min()) if depth_min is None else float(depth_min)
    hi = float(d.max()) if depth_max is None else float(depth_max)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colors = colors.reshape(-1, 3)

    for (u, v), c in zip(uv, colors):
        ui, vi = int(round(float(u))), int(round(float(v)))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(out, (ui, vi), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def draw_bbox_3d_overlay(
    image: np.ndarray,
    obj: DynamicObject,
    K: np.ndarray,
    T_wc: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label_text: bool = True,
) -> tuple[np.ndarray, bool]:
    """Draw a single 3D bbox's edges onto ``image``.

    Returns
    -------
    image_drawn : the image with the bbox drawn (copy).
    drawn : True iff all 8 corners projected in front of the camera and
        at least one edge fell inside the image.

    The bbox is skipped (no draw, ``drawn=False``) when any corner is
    behind the image plane — partial-bbox clipping is more involved than
    we need for a debug overlay.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for draw_bbox_3d_overlay")

    corners_world = bbox_corners_from_object(obj)
    P_cam = world_to_camera(corners_world, T_wc)
    if not (P_cam[:, 2] > 0.1).all():
        return image.copy(), False

    uv, _ = project_camera_to_image(P_cam, K, dist, z_min=0.0)
    if uv.shape[0] != 8:
        return image.copy(), False

    out = image.copy()
    h, w = out.shape[:2]
    drew_any = False
    for i, j in BBOX_EDGES:
        p1 = (int(round(float(uv[i, 0]))), int(round(float(uv[i, 1]))))
        p2 = (int(round(float(uv[j, 0]))), int(round(float(uv[j, 1]))))
        # Skip edges with both endpoints out of frame
        if (
            (p1[0] < -2000 or p1[0] > w + 2000 or p1[1] < -2000 or p1[1] > h + 2000)
            and (p2[0] < -2000 or p2[0] > w + 2000 or p2[1] < -2000 or p2[1] > h + 2000)
        ):
            continue
        cv2.line(out, p1, p2, color, thickness)
        drew_any = True

    if drew_any and label_text:
        anchor = (
            int(round(float(uv[4, 0]))),
            int(round(float(uv[4, 1]))) - 4,
        )
        if 0 <= anchor[0] < w and 0 <= anchor[1] < h:
            cv2.putText(
                out,
                f"{obj.label}#{obj.id}",
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return out, drew_any


# --------------------------------------------------------------------- #
# Class → color (deterministic palette for repeatable figures)
# --------------------------------------------------------------------- #
_CLASS_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "Car": (0, 255, 0),
    "Suv": (0, 200, 0),
    "Truck": (0, 165, 255),
    "Bus": (0, 100, 255),
    "Huge_vehicle": (0, 60, 200),
    "Pedestrian": (255, 200, 0),
    "Pedestrian_else": (255, 150, 0),
    "Non_motor_rider": (255, 0, 200),
    "Motor_rider": (255, 0, 150),
    "Motorcycle": (200, 0, 255),
    "Bicycle": (150, 0, 255),
    "Tricycle": (100, 50, 255),
    "Bollards": (200, 200, 200),
    "Crash_bucket": (180, 180, 180),
    "Cone": (160, 160, 160),
}


def color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic BGR color for a class label (default: yellow)."""
    return _CLASS_COLOR_BGR.get(label, (0, 255, 255))
