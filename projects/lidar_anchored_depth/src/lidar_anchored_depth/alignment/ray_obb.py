"""Ray ↔ oriented-bbox geometry — used by HAD's path-B z_target.

The dense LSQ in :func:`alignment.height_anchor.solve_affine_dense_lsq`
originally assumed each mask pixel's world Z varied linearly with image
row. That holds for head-on frontal views but breaks down for side /
oblique views where columns also matter — leading to per-camera (a, b)
bias that manifests as ghost-shells when accumulating across cameras.

This module replaces the row-linear assumption with the **exact**
geometric z_target: for every mask pixel, intersect its camera ray
with the V2X 3D bbox (a 6-face oriented box) and take the camera-axis
depth of the camera-facing intersection (the "near" hit). Per-pixel
geometric correctness across any view direction.

Slab algorithm
--------------
We work in the bbox-local frame (X = length, Y = width, Z = height).
Transform the world ray into the local frame; the OBB then becomes an
axis-aligned box ``[-l/2, l/2] × [-w/2, w/2] × [-h/2, h/2]``. For each
axis ``k``::

    t_lo_k = (-half_k - origin_local_k) / dir_local_k
    t_hi_k = ( half_k - origin_local_k) / dir_local_k
    swap so t_near_k <= t_far_k

The box hit is ``[max(t_near_k), min(t_far_k)]`` if non-empty AND
``> 0`` (in front of the ray origin = camera).

Z_cam from t
------------
Pixels backproject as ``rays_cam = K^{-1} · [u, v, 1]`` with
``rays_cam[:, 2] == 1``. The world parametric ray is
``P_world(t) = origin_world + t · dirs_world`` where
``dirs_world = R_wc · rays_cam``. Because ``R_wc`` is a rotation, and
``camera_Z_axis_in_world = R_wc · (0,0,1)``, we have
``dirs_world · camera_Z_axis = rays_cam[2] = 1``. So the world
parametric ``t`` equals the camera-axis depth ``z_cam``. We can return
``t_near`` directly as ``z_cam_target``.
"""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.alignment.projection import pixel_to_camera_ray
from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.data.calibration import euler_zyx_to_R


def ray_obb_z_target(
    pixel_uv: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    obj: DynamicObject,
    *,
    expand: float = 0.0,
) -> np.ndarray:
    """Per-pixel camera-axis depth at the V2X bbox's near surface.

    Pixels whose ray misses the bbox (or hits it behind the camera) get
    ``NaN``.

    Parameters
    ----------
    pixel_uv : (N, 2) pixel coordinates.
    K, T_wc : pinhole calibration of the camera the pixels came from.
    obj : :class:`DynamicObject` providing the oriented bbox.
    expand : meters added to every half-extent — useful to mirror the
        same expand the LiDAR-side membership test uses, so HAD's
        z_target and the LiDAR ground truth share the same surface.

    Returns
    -------
    z_target : (N,) float64 — camera-axis depth at the bbox-near hit.
        ``NaN`` for pixels that miss or whose ray is parallel to all
        three axis pairs (the latter is degenerate and shouldn't arise
        for finite intrinsics).
    """
    uv = np.asarray(pixel_uv, dtype=np.float64).reshape(-1, 2)
    if uv.size == 0:
        return np.zeros(0, dtype=np.float64)

    rays_cam = pixel_to_camera_ray(uv, K)  # (N, 3) with rays_cam[:, 2] == 1
    R_wc = np.asarray(T_wc, dtype=np.float64)[:3, :3]
    origin_world = np.asarray(T_wc, dtype=np.float64)[:3, 3]
    dirs_world = (R_wc @ rays_cam.T).T  # (N, 3)

    # Transform into bbox-local frame
    R_obj = euler_zyx_to_R(obj.roll, obj.pitch, obj.yaw)
    R_obj_T = R_obj.T
    origin_local = R_obj_T @ (origin_world - obj.xyz)  # (3,)
    dirs_local = (R_obj_T @ dirs_world.T).T  # (N, 3)

    half = (np.asarray(obj.lwh, dtype=np.float64) / 2.0) + expand  # (3,)

    # Slab method per axis
    eps = 1e-12
    t_near = np.full(uv.shape[0], -np.inf, dtype=np.float64)
    t_far = np.full(uv.shape[0], +np.inf, dtype=np.float64)
    valid = np.ones(uv.shape[0], dtype=bool)

    for k in range(3):
        d_k = dirs_local[:, k]
        o_k = origin_local[k]
        h_k = half[k]
        parallel = np.abs(d_k) < eps
        # Parallel ray: must already lie within the slab, else miss
        miss_parallel = parallel & ((o_k < -h_k) | (o_k > +h_k))
        valid &= ~miss_parallel
        # For non-parallel, compute t_lo / t_hi
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (-h_k - o_k) / d_k
            t_hi = (+h_k - o_k) / d_k
        swap = t_lo > t_hi
        t_near_k = np.where(swap, t_hi, t_lo)
        t_far_k = np.where(swap, t_lo, t_hi)
        # Only update for non-parallel rays
        t_near = np.where(parallel, t_near, np.maximum(t_near, t_near_k))
        t_far = np.where(parallel, t_far, np.minimum(t_far, t_far_k))

    valid &= t_near <= t_far
    valid &= t_near > 0.0  # near hit in front of camera

    out = np.full(uv.shape[0], np.nan, dtype=np.float64)
    out[valid] = t_near[valid]
    return out
