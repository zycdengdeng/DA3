"""Ground-plane RANSAC fit and per-pixel ground depth (B5).

A single RANSAC plane is fit to the LiDAR ground returns; per-pixel
ground depth is then obtained analytically by intersecting each
back-projection ray with this plane.

This is the straightforward ground-only baseline. For the full HAD
pipeline, the 4-LiDAR stratification + 300 m extent will eventually push
us toward a learned ground field MLP (Stage 2C+), but this baseline is
necessary for ablations and as a fallback when the MLP is not trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.alignment.projection import pixel_ray_to_world


@dataclass(slots=True)
class GroundPlane:
    """World-frame plane ``n · X + d = 0`` with unit normal ``n``.

    Convention: the normal points "up" in world frame (i.e., its Z
    component is non-negative).
    """

    normal: np.ndarray  # (3,) unit vector
    offset: float  # d in n·X + d = 0
    inlier_count: int
    inlier_ratio: float
    residual_std: float


# --------------------------------------------------------------------- #
# RANSAC
# --------------------------------------------------------------------- #
def fit_ground_plane_ransac(
    points_world: np.ndarray,
    *,
    distance_threshold: float = 0.20,
    n_iterations: int = 2000,
    seed: int = 0,
    refine_with_inliers: bool = True,
    min_inliers_to_refine: int = 50,
) -> GroundPlane:
    """RANSAC 3-point plane fit.

    Parameters
    ----------
    points_world : (N, 3)
        World-frame points (LiDAR), already filtered to candidate ground
        returns (e.g., ``z < median(z) + ε`` or simply "not in any V2X
        bbox"). The filtering policy is the caller's responsibility.
    distance_threshold : float, meters
        Inlier threshold on perpendicular distance from the plane.
    n_iterations : int
        Number of random minimal fits.
    seed : int
        RNG seed.
    refine_with_inliers : bool
        After the best inlier set is found, run a least-squares plane
        refit on those inliers (PCA on centered points). Recommended.
    min_inliers_to_refine : int
        Skip refit if fewer inliers; falls back to the minimal-fit plane.
    """
    P = np.asarray(points_world, dtype=np.float64)
    if P.shape[0] < 3:
        raise ValueError(
            f"fit_ground_plane_ransac: need >= 3 points, got {P.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    n = P.shape[0]
    best_n = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_offset = 0.0
    best_inliers: np.ndarray | None = None

    for _ in range(n_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = P[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-12:
            continue
        normal = normal / nrm
        # Ensure normal "up" (Z >= 0)
        if normal[2] < 0:
            normal = -normal
        offset = -float(normal @ p0)
        # Distance from plane (signed)
        dist = np.abs(P @ normal + offset)
        inliers = dist < distance_threshold
        cnt = int(inliers.sum())
        if cnt > best_n:
            best_n = cnt
            best_normal = normal
            best_offset = offset
            best_inliers = inliers

    if best_inliers is None or best_n < 3:
        raise RuntimeError(
            "fit_ground_plane_ransac: failed to find any plane "
            f"(best inlier count = {best_n})"
        )

    if refine_with_inliers and best_n >= min_inliers_to_refine:
        # Total least squares: SVD on centered inliers
        Q = P[best_inliers]
        centroid = Q.mean(axis=0)
        Qc = Q - centroid
        # Smallest singular vector is the plane normal
        _, _, vh = np.linalg.svd(Qc, full_matrices=False)
        n_refit = vh[-1]
        if n_refit[2] < 0:
            n_refit = -n_refit
        n_refit = n_refit / float(np.linalg.norm(n_refit))
        d_refit = -float(n_refit @ centroid)
        # Re-evaluate inliers
        dist2 = np.abs(P @ n_refit + d_refit)
        inliers2 = dist2 < distance_threshold
        cnt2 = int(inliers2.sum())
        if cnt2 >= best_n:
            best_normal = n_refit
            best_offset = d_refit
            best_inliers = inliers2
            best_n = cnt2

    # Residual stats on inliers
    res = P[best_inliers] @ best_normal + best_offset
    residual_std = float(np.std(res))
    return GroundPlane(
        normal=best_normal.astype(np.float64),
        offset=float(best_offset),
        inlier_count=best_n,
        inlier_ratio=best_n / n,
        residual_std=residual_std,
    )


# --------------------------------------------------------------------- #
# Per-pixel ground depth
# --------------------------------------------------------------------- #
def ray_plane_intersection(
    origin: np.ndarray,
    directions: np.ndarray,
    plane: GroundPlane,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Solve ``t`` such that ``n · (origin + t · dir) + d = 0``.

    Parameters
    ----------
    origin : (3,) world-frame ray origin (camera centre).
    directions : (N, 3) world-frame ray directions.
    plane : :class:`GroundPlane`.

    Returns
    -------
    t : (N,) parameter along each ray. Negative or NaN values mean the
        ray does not hit the plane in front of the camera.
    """
    n = plane.normal.astype(np.float64)
    d = float(plane.offset)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)

    denom = dirs @ n  # (N,)
    num = -(o @ n + d)  # scalar
    t = np.full(dirs.shape[0], np.nan, dtype=np.float64)
    valid = np.abs(denom) > eps
    t[valid] = num / denom[valid]
    return t


def ground_depth_map(
    plane: GroundPlane,
    K: np.ndarray,
    T_wc: np.ndarray,
    image_hw: tuple[int, int],
    *,
    non_ground_mask: np.ndarray | None = None,
    max_depth: float = 300.0,
) -> np.ndarray:
    """Per-pixel metric depth from camera-ray ↔ ground-plane intersection.

    Parameters
    ----------
    plane : :class:`GroundPlane`
        Fitted in the world frame.
    K : (3, 3) intrinsics.
    T_wc : (4, 4) camera-to-world.
    image_hw : (H, W).
    non_ground_mask : optional (H, W) bool, ``True`` = pixel is NOT
        ground (e.g. inside an instance mask). These pixels return NaN.
    max_depth : pixels with depth > max_depth or behind the camera are
        set to NaN.

    Returns
    -------
    depth : (H, W) float32, in meters. NaN where invalid.
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)

    origin, dirs = pixel_ray_to_world(uv, K, T_wc)
    t = ray_plane_intersection(origin, dirs, plane)

    # Convert t (along the world-frame direction) into camera-frame depth.
    # Camera depth = z component of P_cam = z component of T_cw · P_world.
    # P_world = origin + t · dir_world.
    # P_cam = T_cw · [P_world; 1]. We need P_cam[2].
    T_cw = np.linalg.inv(np.asarray(T_wc, dtype=np.float64))
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    P_world = origin[None, :] + t[:, None] * dirs  # (N, 3)
    z_cam = (P_world @ R_cw[2, :]) + t_cw[2]  # row 2 of R_cw

    bad = (
        np.isnan(z_cam)
        | (z_cam <= 0.0)
        | (z_cam > max_depth)
        | np.isnan(t)
    )
    z_cam[bad] = np.nan

    depth = z_cam.reshape(H, W).astype(np.float32)
    if non_ground_mask is not None:
        if non_ground_mask.shape != (H, W):
            raise ValueError(
                f"non_ground_mask shape {non_ground_mask.shape} != ({H}, {W})"
            )
        depth[non_ground_mask] = np.nan
    return depth
