"""Drawing helpers for LiDAR / 3D-bbox debug overlays.

Geometry primitives (``world_to_camera``, ``project_camera_to_image``,
``world_to_image``, ``bbox_3d_corners``, ``BBOX_EDGES``, ...) are
re-exported from :mod:`lidar_anchored_depth.alignment.projection` and
canonicalized there. This module owns only the cv2-based drawing and
the per-class color palette.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 is required for drawing
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

from lidar_anchored_depth.alignment.projection import (
    BBOX_EDGES,
    bbox_3d_corners,
    bbox_corners_from_object,
    project_3d_bbox,
    project_camera_to_image,
    world_to_camera,
    world_to_image,
)
from lidar_anchored_depth.data.base import DynamicObject

__all__ = [
    "BBOX_EDGES",
    "bbox_3d_corners",
    "bbox_corners_from_object",
    "color_for_label",
    "draw_bbox_3d_overlay",
    "draw_lidar_overlay",
    "project_3d_bbox",
    "project_camera_to_image",
    "world_to_camera",
    "world_to_image",
]


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
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for draw_bbox_3d_overlay")

    uv8, _ = project_3d_bbox(obj, K, T_wc, dist, z_min=0.1)
    if uv8 is None:
        return image.copy(), False

    out = image.copy()
    h, w = out.shape[:2]
    drew_any = False
    for i, j in BBOX_EDGES:
        p1 = (int(round(float(uv8[i, 0]))), int(round(float(uv8[i, 1]))))
        p2 = (int(round(float(uv8[j, 0]))), int(round(float(uv8[j, 1]))))
        if (
            (p1[0] < -2000 or p1[0] > w + 2000 or p1[1] < -2000 or p1[1] > h + 2000)
            and (p2[0] < -2000 or p2[0] > w + 2000 or p2[1] < -2000 or p2[1] > h + 2000)
        ):
            continue
        cv2.line(out, p1, p2, color, thickness)
        drew_any = True

    if drew_any and label_text:
        anchor = (
            int(round(float(uv8[4, 0]))),
            int(round(float(uv8[4, 1]))) - 4,
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
