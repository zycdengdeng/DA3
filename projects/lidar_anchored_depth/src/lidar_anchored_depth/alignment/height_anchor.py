"""Height-Anchored Depth (HAD) — closed-form per-instance solver.

The geometry derivation lives in ``docs/method.md`` §3. The crux is that
for a pixel ``(u, v)`` and DA3 relative depth ``d̃`` the world-frame Z
component of the back-projected 3D point is **linear** in the affine
parameters ``(a, b)`` (with ``z_cam = a · d̃ + b``), so two height
anchors produce a closed-form ``(a, b)``.

Two anchor sources are supported:

- **M1 HAD-mask** (this module): estimate ``(Z_min, Z_max)`` from the
  LiDAR points falling inside the SAM mask via percentile clipping.
- **M2 HAD-bbox / M3 AA-HAD** (:mod:`alignment.bbox_anchor`): take
  ``(Z_min, Z_max)`` directly from the V2X 3D-bbox annotation.

This module owns the shared closed-form solver
(:func:`solve_affine_from_height_anchors`) and the M1 mask-driven path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    world_to_image,
    world_z_at_pixel_unit_depth,
)


# --------------------------------------------------------------------- #
# Closed-form solver — shared by M1 / M2 / M3
# --------------------------------------------------------------------- #
def solve_affine_from_height_anchors(
    K: np.ndarray,
    T_wc: np.ndarray,
    uv_top: np.ndarray,           # (2,) pixel
    uv_bot: np.ndarray,           # (2,) pixel
    d_pred_top: float,            # DA3 relative depth at top pixel
    d_pred_bot: float,            # DA3 relative depth at bottom pixel
    Z_max: float,                 # world Z at top of object
    Z_min: float,                 # world Z at bottom of object
) -> tuple[float, float] | None:
    """Solve ``(a, b)`` from two height anchors. Closed-form, no iteration.

    Math (see ``docs/method.md`` §3, ``world_z_at_pixel_unit_depth``)::

        Z_world(u, v, z_cam) = α(u,v) · z_cam + β
        z_cam(u, v)         = a · d̃(u, v) + b
        ⇒ Z_world = α · (a · d̃ + b) + β

    Two anchors → 2×2 linear system in ``(a, b)``::

        α_top · d̃_top · a + α_top · b = Z_max - β
        α_bot · d̃_bot · a + α_bot · b = Z_min - β

    Returns ``(a, b)`` on success, or ``None`` when the system is
    singular. **Sensitive to single-pixel ``d̃`` noise** — use
    :func:`solve_affine_dense_lsq` on real data when a SAM mask is
    available; reserve this 2-anchor form for synthetic / theoretical
    work.
    """
    uv = np.asarray([uv_top, uv_bot], dtype=np.float64)
    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)
    a_top, a_bot = float(alphas[0]), float(alphas[1])
    if abs(a_top) < 1e-9 or abs(a_bot) < 1e-9:
        return None

    A = np.array(
        [
            [a_top * d_pred_top, a_top],
            [a_bot * d_pred_bot, a_bot],
        ],
        dtype=np.float64,
    )
    rhs = np.array([Z_max - beta, Z_min - beta], dtype=np.float64)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    sol = np.linalg.solve(A, rhs)
    return float(sol[0]), float(sol[1])


def solve_affine_dense_lsq(
    K: np.ndarray,
    T_wc: np.ndarray,
    mask: np.ndarray,
    d_pred_image: np.ndarray,
    Z_max: float,
    Z_min: float,
    *,
    robust_lo_pct: float = 0.0,
    robust_hi_pct: float = 100.0,
    min_pixels: int = 50,
) -> tuple[float, float] | None:
    """Robust HAD solver using **all** mask pixels (super-determined LSQ).

    Replaces the 2-anchor closed-form (which reads ``d̃`` at exactly two
    pixels and is fully exposed to single-pixel noise) with a stacked
    LSQ over every mask pixel. The assumption is that within a mask, the
    object's surface world-Z varies linearly with image row:

        Z_target(v) = Z_max + (v − v_top_extent) / (v_bot_extent − v_top_extent) · (Z_min − Z_max)

    where ``v_top_extent`` and ``v_bot_extent`` are the **true** mask
    bounds (``rows.min(), rows.max()``) — they define the interpolation
    line. The optional ``robust_lo_pct`` / ``robust_hi_pct`` percentiles
    additionally **drop** rows outside the band from the LSQ for
    robustness to a few stray edge pixels (default no drop). For each
    surviving mask pixel:

        α(u,v) · d̃(u,v) · a  +  α(u,v) · b  =  Z_target(v) - β

    All such rows are stacked into a tall ``(N, 2) · (a, b) = N``
    least-squares system and solved with ``np.linalg.lstsq``. Robust to
    ``d̃`` noise (averaged over thousands of pixels) and to SAM masks
    that are tighter than the bbox AABB (since the interpolation
    extents are re-estimated per mask).

    Parameters
    ----------
    K, T_wc : pinhole calibration.
    mask : (H, W) bool — the SAM (or AABB) mask of one instance.
    d_pred_image : (H, W) float — full-image relative depth (DA3 output).
    Z_max, Z_min : world-Z extents of the object (from V2X bbox).
    robust_lo_pct, robust_hi_pct : drop rows outside this percentile
        band from the LSQ (DOES NOT change the interpolation; the linear
        interp always uses the true ``rows.min/max``). Defaults
        ``0 / 100`` = no drop. Try ``5 / 95`` on noisy real-data SAM
        masks if some pixels at the silhouette boundary are unreliable.
    min_pixels : reject masks with fewer pixels than this (returns
        ``None``).

    Returns
    -------
    (a, b) on success, or ``None`` if the mask is too small / d̃
    invalid / system rank-deficient.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    H, W = mask.shape
    if d_pred_image.shape != (H, W):
        raise ValueError(
            f"d_pred_image shape {d_pred_image.shape} must match mask {mask.shape}"
        )
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    if rows.size < min_pixels:
        return None

    # Interpolation extent always uses the true mask bounds.
    v_top_extent = float(rows.min())
    v_bot_extent = float(rows.max())
    span = v_bot_extent - v_top_extent
    if span < 2.0:
        return None  # mask too thin vertically

    # Optional row-band filter for robustness against silhouette edge noise.
    if robust_lo_pct > 0.0 or robust_hi_pct < 100.0:
        v_lo = float(np.percentile(rows, robust_lo_pct))
        v_hi = float(np.percentile(rows, robust_hi_pct))
        in_band = (rows >= v_lo) & (rows <= v_hi)
        rows = rows[in_band]
        cols = cols[in_band]
        if rows.size < min_pixels:
            return None

    d_vals = d_pred_image[rows, cols].astype(np.float64)
    finite = np.isfinite(d_vals) & (d_vals > 0)
    if int(finite.sum()) < min_pixels:
        return None
    rows_f = rows[finite].astype(np.float64)
    cols_f = cols[finite].astype(np.float64)
    d_vals = d_vals[finite]

    # Per-pixel α(u, v) and β
    uv = np.stack([cols_f, rows_f], axis=1)
    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)
    safe = np.abs(alphas) > 1e-9
    if int(safe.sum()) < min_pixels:
        return None
    rows_f = rows_f[safe]
    alphas = alphas[safe]
    d_vals = d_vals[safe]

    # Linear-Z-vs-v interpolation using the TRUE mask extent (no clamp,
    # no drift from percentile-defined v_top to true row.min).
    frac = (rows_f - v_top_extent) / span                # in [0, 1] for kept rows
    z_target = Z_max + frac * (Z_min - Z_max)

    # Build the LSQ system: A · [a; b] = rhs
    A = np.stack([alphas * d_vals, alphas], axis=1)  # (N, 2)
    rhs = z_target - beta                              # (N,)

    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if sol.shape[0] != 2:
        return None
    return float(sol[0]), float(sol[1])


# --------------------------------------------------------------------- #
# Joint LSQ across many (mask, image) pairs of the same camera
# --------------------------------------------------------------------- #
def _build_lsq_rows_for_one_frame(
    K: np.ndarray,
    T_wc: np.ndarray,
    mask: np.ndarray,
    d_pred_image: np.ndarray,
    Z_max: float,
    Z_min: float,
    *,
    obj=None,  # DynamicObject for ray-OBB; None → row-linear z_target
    bbox_expand: float = 0.0,
    robust_lo_pct: float = 0.0,
    robust_hi_pct: float = 100.0,
    min_pixels: int = 50,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the LSQ rows ``A_rows · [a; b] = rhs_rows`` for one frame.

    Returns ``(A_rows, rhs_rows)`` or ``None`` when the frame yields too
    few usable pixels. Internal helper for the per-camera joint solver.

    When ``obj`` is provided, ``z_target`` per pixel comes from
    :func:`alignment.ray_obb.ray_obb_z_target` (Path B — geometrically
    exact). Otherwise the row-linear interpolation is used (current
    Path A baseline).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    H, W = mask.shape
    if d_pred_image.shape != (H, W):
        return None
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    if rows.size < min_pixels:
        return None

    if robust_lo_pct > 0.0 or robust_hi_pct < 100.0:
        v_lo = float(np.percentile(rows, robust_lo_pct))
        v_hi = float(np.percentile(rows, robust_hi_pct))
        in_band = (rows >= v_lo) & (rows <= v_hi)
        rows = rows[in_band]
        cols = cols[in_band]
        if rows.size < min_pixels:
            return None

    d_vals = d_pred_image[rows, cols].astype(np.float64)
    finite = np.isfinite(d_vals) & (d_vals > 0)
    if int(finite.sum()) < min_pixels:
        return None
    rows_f = rows[finite].astype(np.float64)
    cols_f = cols[finite].astype(np.float64)
    d_vals = d_vals[finite]

    uv = np.stack([cols_f, rows_f], axis=1)
    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)
    safe = np.abs(alphas) > 1e-9
    if int(safe.sum()) < min_pixels:
        return None
    rows_f = rows_f[safe]
    cols_f = cols_f[safe]
    alphas = alphas[safe]
    d_vals = d_vals[safe]
    uv = np.stack([cols_f, rows_f], axis=1)

    # ---- z_target: row-linear (default) or ray-OBB (Path B) ----
    if obj is None:
        # Row-linear interpolation using the TRUE mask extent
        v_top_extent = float(rows.min())
        v_bot_extent = float(rows.max())
        span = v_bot_extent - v_top_extent
        if span < 2.0:
            return None
        frac = (rows_f - v_top_extent) / span
        z_target_world = Z_max + frac * (Z_min - Z_max)  # this is a world-Z target
        # Convert to "z_cam target" via Z_world = α · z_cam + β
        # We want the LSQ to produce a, b such that α · (a · d̃ + b) + β ≈ z_target_world
        # ⇒ rows: α · d̃ · a + α · b = z_target_world - β
        rhs_rows = z_target_world - beta
    else:
        from lidar_anchored_depth.alignment.ray_obb import ray_obb_z_target
        z_cam_target = ray_obb_z_target(uv, K, T_wc, obj, expand=bbox_expand)
        ok = np.isfinite(z_cam_target)
        if int(ok.sum()) < min_pixels:
            return None
        d_vals = d_vals[ok]
        z_cam_target = z_cam_target[ok]
        # Path B: z_cam_target is in CAMERA-axis space directly (it's the
        # ray's t to the bbox-near hit, with rays_cam[2] == 1 → t == z_cam).
        # HAD model: z_cam = a · d̃ + b. So each pixel adds:
        #     d̃ · a + 1 · b = z_cam_target
        A_rows = np.stack([d_vals, np.ones_like(d_vals)], axis=1)
        rhs_rows = z_cam_target
        return A_rows, rhs_rows

    A_rows = np.stack([alphas * d_vals, alphas], axis=1)
    return A_rows, rhs_rows


def solve_affine_dense_lsq_multi(
    inputs: list[dict],
    *,
    z_target: str = "row-linear",
    robust_lo_pct: float = 0.0,
    robust_hi_pct: float = 100.0,
    min_pixels_per_frame: int = 50,
    min_total_pixels: int = 200,
) -> tuple[float, float] | None:
    """Joint dense-LSQ across multiple frames sharing the same camera.

    Solves a *single* ``(a, b)`` from the stacked LSQ rows of every
    input frame. The intended use is "one (a, b) per (camera, object)"
    where many frames of the same object as seen by the same camera
    pool their pixel constraints — averages out per-frame DA3 noise +
    per-frame SAM mask jitter.

    Parameters
    ----------
    inputs : list of dicts. Each dict has::
        {
          'K': (3, 3),
          'T_wc': (4, 4),
          'mask': (H, W) bool,
          'd_pred_image': (H, W) float,
          'Z_max': float, 'Z_min': float,   # only used when z_target='row-linear'
          'obj': DynamicObject,             # only used when z_target='ray-obb'
        }
    z_target : 'row-linear' (Path A only) or 'ray-obb' (Path A+B).
    robust_lo_pct, robust_hi_pct : per-frame row percentile band filter.
    min_pixels_per_frame : drop frames with fewer usable pixels than this.
    min_total_pixels : refuse to solve if the stack has fewer rows total.

    Returns
    -------
    (a, b) on success, else ``None``.
    """
    if z_target not in ("row-linear", "ray-obb"):
        raise ValueError(f"z_target must be 'row-linear' or 'ray-obb'; got {z_target!r}")

    A_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    for entry in inputs:
        if z_target == "row-linear":
            built = _build_lsq_rows_for_one_frame(
                K=entry["K"], T_wc=entry["T_wc"],
                mask=entry["mask"], d_pred_image=entry["d_pred_image"],
                Z_max=entry["Z_max"], Z_min=entry["Z_min"],
                obj=None,
                robust_lo_pct=robust_lo_pct, robust_hi_pct=robust_hi_pct,
                min_pixels=min_pixels_per_frame,
            )
        else:  # 'ray-obb'
            built = _build_lsq_rows_for_one_frame(
                K=entry["K"], T_wc=entry["T_wc"],
                mask=entry["mask"], d_pred_image=entry["d_pred_image"],
                Z_max=entry.get("Z_max", 0.0), Z_min=entry.get("Z_min", 0.0),
                obj=entry["obj"],
                bbox_expand=entry.get("bbox_expand", 0.0),
                robust_lo_pct=robust_lo_pct, robust_hi_pct=robust_hi_pct,
                min_pixels=min_pixels_per_frame,
            )
        if built is None:
            continue
        A_rows, rhs_rows = built
        A_blocks.append(A_rows)
        rhs_blocks.append(rhs_rows)

    if not A_blocks:
        return None
    A = np.vstack(A_blocks)
    rhs = np.concatenate(rhs_blocks)
    if A.shape[0] < min_total_pixels:
        return None
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if sol.shape[0] != 2:
        return None
    return float(sol[0]), float(sol[1])


def solve_affine_dense_lsq_multi_with_lidar(
    inputs: list[dict],
    *,
    lidar_weight: float = 100.0,
    min_pixels_per_frame: int = 50,
    min_lidar_per_frame: int = 5,
    min_total_pixels: int = 200,
) -> tuple[float, float] | None:
    """Joint per-camera LSQ that combines mask pixels with LiDAR anchors.

    M5: stacks two row types into one big LSQ system, both expressed in
    camera-axis depth space (so they're dimensionally compatible):

      Mask row     (Path A+B):  d̃_mask · a + b = z_cam_target_OBB
      LiDAR row    (Path C):    d̃_lidar · a + b = z_cam_lidar  (× sqrt(w))

    The mask side uses the geometrically-exact ``ray_obb_z_target``
    (Path B) to define the per-pixel target depth. The LiDAR side
    contributes the actually-measured camera depth at the projection of
    each in-bbox LiDAR point. Each LiDAR row is scaled by
    ``sqrt(lidar_weight)`` so that a single LiDAR sample counts as
    ``lidar_weight`` mask pixels in the LSQ residual sum (default 100).

    Why this is better than mask-only or LiDAR-only:
      - Mask-only (Path A+B): bbox surface is GEOMETRIC prior; per-pixel
        d̃ noise propagates → ~0.4 m chamfer at 30-100 m range.
      - LiDAR-only (B4 baseline): centimeter-precise but sparse — a few
        dozen LiDAR points per object can't constrain (a, b) with
        sub-meter precision under d̃ noise.
      - Combined: mask provides density, LiDAR provides absolute scale.
        Expected chamfer 0.15-0.20 m on real data.

    Parameters
    ----------
    inputs : list of dicts. Each dict has::
        {
          'K': (3, 3),
          'T_wc': (4, 4),
          'mask': (H, W) bool,
          'd_pred_image': (H, W) float,
          'obj': DynamicObject,                     # for ray-OBB
          'lidar_world_in_bbox': (M, 3) or None,    # LiDAR points already
              # filtered to lie inside this object's 3D bbox; if None,
              # this frame contributes only mask rows.
          'dist': optional distortion vector,
          'bbox_expand': optional float for ray-OBB expand,
        }
    lidar_weight : per-LiDAR-row weight in the LSQ residual (default 100).
        Roughly: how many mask pixels does one LiDAR point count as.
    min_pixels_per_frame : drop frames whose mask side yields fewer
        usable pixels than this.
    min_lidar_per_frame : drop frames whose LiDAR side yields fewer
        in-mask LiDAR projections than this.
    min_total_pixels : require this many total LSQ rows to attempt.

    Returns
    -------
    (a, b) on success, else ``None``.
    """
    A_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    n_lidar_used = 0
    n_mask_used = 0

    weight_sqrt = float(np.sqrt(max(lidar_weight, 0.0)))

    for entry in inputs:
        # ---- Mask side: ray-OBB z_target (Path B) -----------------
        built = _build_lsq_rows_for_one_frame(
            K=entry["K"], T_wc=entry["T_wc"],
            mask=entry["mask"], d_pred_image=entry["d_pred_image"],
            Z_max=entry.get("Z_max", 0.0), Z_min=entry.get("Z_min", 0.0),
            obj=entry["obj"],
            bbox_expand=entry.get("bbox_expand", 0.0),
            min_pixels=min_pixels_per_frame,
        )
        if built is not None:
            A_mask, rhs_mask = built
            A_blocks.append(A_mask)
            rhs_blocks.append(rhs_mask)
            n_mask_used += A_mask.shape[0]

        # ---- LiDAR side: actually-measured z_cam at projected LiDAR -
        lidar_world = entry.get("lidar_world_in_bbox")
        if lidar_world is None or lidar_world.shape[0] < min_lidar_per_frame:
            continue
        if weight_sqrt <= 0.0:
            continue

        K = entry["K"]
        T_wc = entry["T_wc"]
        dist = entry.get("dist")
        d_image = entry["d_pred_image"]
        mask = entry["mask"]
        H, W = d_image.shape

        uv, z_cam_lidar, _ = world_to_image(lidar_world, K, T_wc, dist)
        if uv.size == 0:
            continue
        finite = np.isfinite(uv).all(axis=1)
        uv = uv[finite]
        z_cam_lidar = z_cam_lidar[finite]
        if uv.size == 0:
            continue
        uv_int = np.round(uv).astype(np.int64)
        in_image = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
            & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
        )
        u = uv_int[in_image, 0]
        v = uv_int[in_image, 1]
        z_cam_lidar = z_cam_lidar[in_image].astype(np.float64)
        if u.size < min_lidar_per_frame:
            continue
        # Restrict to LiDAR points that project ONTO the SAM mask of
        # this object, so we read d̃ at the right surface (not the
        # background behind it).
        on_mask = mask[v, u]
        u = u[on_mask]
        v = v[on_mask]
        z_cam_lidar = z_cam_lidar[on_mask]
        if u.size < min_lidar_per_frame:
            continue
        d_at = d_image[v, u].astype(np.float64)
        valid = np.isfinite(d_at) & (d_at > 0) & np.isfinite(z_cam_lidar) & (z_cam_lidar > 0)
        if int(valid.sum()) < min_lidar_per_frame:
            continue
        d_at = d_at[valid]
        z_cam_lidar = z_cam_lidar[valid]

        # LSQ rows: d̃ · a + 1 · b = z_cam_lidar, scaled by sqrt(weight).
        A_lidar = weight_sqrt * np.stack(
            [d_at, np.ones_like(d_at)], axis=1
        )
        rhs_lidar = weight_sqrt * z_cam_lidar
        A_blocks.append(A_lidar)
        rhs_blocks.append(rhs_lidar)
        n_lidar_used += int(valid.sum())

    if not A_blocks:
        return None
    A = np.vstack(A_blocks)
    rhs = np.concatenate(rhs_blocks)
    if A.shape[0] < min_total_pixels:
        return None
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if sol.shape[0] != 2:
        return None
    return float(sol[0]), float(sol[1])


# --------------------------------------------------------------------- #
# Mask geometry helpers
# --------------------------------------------------------------------- #
def mask_top_bottom_pixels(
    mask: np.ndarray, *, take_centroid_of_row: bool = True
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the topmost and bottommost pixel of a binary mask.

    The horizontal coordinate is the row centroid (averaged column
    indices) by default — robust to a ragged mask edge. Returns
    ``None`` if the mask is empty.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    v_top = int(rows.min())
    v_bot = int(rows.max())
    if take_centroid_of_row:
        u_top = float(cols[rows == v_top].mean())
        u_bot = float(cols[rows == v_bot].mean())
    else:
        u_top = float(cols[rows == v_top].min())
        u_bot = float(cols[rows == v_bot].max())
    return (
        np.array([u_top, v_top], dtype=np.float64),
        np.array([u_bot, v_bot], dtype=np.float64),
    )


# --------------------------------------------------------------------- #
# M1: HAD-mask (height anchors derived from LiDAR-in-mask)
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class HeightInterval:
    """Height interval derived from LiDAR points inside an instance mask."""

    Z_min: float
    Z_max: float
    n_points: int
    confidence: float  # in [0, 1]


def height_interval_from_lidar_in_mask(
    lidar_world: np.ndarray,
    pixel_uv: np.ndarray,
    in_front: np.ndarray,
    mask: np.ndarray,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_points: int = 5,
) -> HeightInterval | None:
    """Estimate ``(Z_min, Z_max)`` for one instance from LiDAR inside its mask.

    Parameters
    ----------
    lidar_world : (N, 3)
        Full LiDAR cloud in world frame.
    pixel_uv : (M, 2)
        Projection of the in-front LiDAR points to image (output of
        ``world_to_image``).
    in_front : (N,) bool
        Mask indicating which of the original LiDAR points were in front
        of the camera (i.e. lined up with ``pixel_uv``).
    mask : (H, W) bool
        Instance mask in the same image as ``pixel_uv``.

    Returns
    -------
    :class:`HeightInterval` or ``None`` if too few points inside the mask.

    Confidence heuristic
    --------------------
    ``confidence = clip(n_points / 50, 0, 1) * (1 - 0.5 · clip((Z_max - Z_min) / 5 - 1, 0, 1))``
    — favors masks with many points and tight intervals.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    H, W = mask.shape
    uv_int = np.round(pixel_uv).astype(np.int64)
    in_image = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    if not in_image.any():
        return None

    # Map back to original LiDAR indices (in-front + in-image)
    mask_vals = np.zeros(uv_int.shape[0], dtype=bool)
    mask_vals[in_image] = mask[uv_int[in_image, 1], uv_int[in_image, 0]]
    if mask_vals.sum() < min_points:
        return None

    # Recover the indices in the ORIGINAL lidar_world array
    front_indices = np.flatnonzero(in_front)
    selected_orig_indices = front_indices[mask_vals]
    Z = lidar_world[selected_orig_indices, 2].astype(np.float64)
    if Z.size < min_points:
        return None

    z_lo = float(np.percentile(Z, percentile_lo))
    z_hi = float(np.percentile(Z, percentile_hi))
    n_pts = int(Z.size)

    confidence_count = float(np.clip(n_pts / 50.0, 0.0, 1.0))
    height_extent = z_hi - z_lo
    looseness = float(np.clip((height_extent / 5.0) - 1.0, 0.0, 1.0))
    confidence = confidence_count * (1.0 - 0.5 * looseness)

    return HeightInterval(
        Z_min=z_lo, Z_max=z_hi, n_points=n_pts, confidence=confidence
    )


@dataclass(slots=True)
class InstanceAnchor:
    """One HAD anchor pair (top + bot) plus its closed-form ``(a, b)``."""

    instance_id: int
    uv_top: np.ndarray  # (2,)
    uv_bot: np.ndarray
    d_pred_top: float
    d_pred_bot: float
    Z_min: float
    Z_max: float
    a: float
    b: float
    confidence: float


@dataclass(slots=True)
class HADResult:
    """Per-frame HAD output (used by M1 and M2/M3)."""

    aligned_depth: np.ndarray  # (H, W) float32 metric depth, NaN elsewhere
    coverage_mask: np.ndarray  # (H, W) bool
    anchors: list[InstanceAnchor] = field(default_factory=list)
    n_masks_attempted: int = 0
    n_masks_solved: int = 0


def had_mask(
    d_pred_image: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    masks: np.ndarray,
    lidar_world: np.ndarray,
    *,
    dist: np.ndarray | None = None,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_lidar_per_mask: int = 5,
) -> HADResult:
    """M1 HAD-mask: per-instance height-anchored solve, anchors from LiDAR.

    For each mask: take its top / bottom pixels, estimate
    ``(Z_min, Z_max)`` from the LiDAR points falling inside it (5 / 95
    percentile by default), and run the shared closed-form solver.

    Parameters
    ----------
    d_pred_image : (H, W) float, DA3 relative depth.
    K, T_wc, dist : pinhole calibration (distortion may be ``None``).
    masks : (K, H, W) bool, instance masks.
    lidar_world : (N, 3) world-frame LiDAR.
    """
    H, W = d_pred_image.shape
    aligned = np.full((H, W), np.nan, dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)

    # Project all LiDAR once
    uv_all, _, in_front = world_to_image(lidar_world, K, T_wc, dist)

    anchors: list[InstanceAnchor] = []
    n_attempt = 0
    n_solved = 0

    for k in range(masks.shape[0]):
        n_attempt += 1
        mask_k = masks[k]
        if not mask_k.any():
            continue
        topbot = mask_top_bottom_pixels(mask_k)
        if topbot is None:
            continue
        uv_top, uv_bot = topbot

        height_int = height_interval_from_lidar_in_mask(
            lidar_world=lidar_world,
            pixel_uv=uv_all,
            in_front=in_front,
            mask=mask_k,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            min_points=min_lidar_per_mask,
        )
        if height_int is None:
            continue

        d_top = float(d_pred_image[int(uv_top[1]), int(uv_top[0])])
        d_bot = float(d_pred_image[int(uv_bot[1]), int(uv_bot[0])])
        if abs(d_top - d_bot) < 1e-9:
            continue

        sol = solve_affine_from_height_anchors(
            K=K, T_wc=T_wc,
            uv_top=uv_top, uv_bot=uv_bot,
            d_pred_top=d_top, d_pred_bot=d_bot,
            Z_max=height_int.Z_max, Z_min=height_int.Z_min,
        )
        if sol is None:
            continue
        a, b = sol

        # Apply to mask
        aligned[mask_k] = (a * d_pred_image[mask_k] + b).astype(np.float32)
        coverage |= mask_k
        n_solved += 1
        anchors.append(
            InstanceAnchor(
                instance_id=k,
                uv_top=uv_top, uv_bot=uv_bot,
                d_pred_top=d_top, d_pred_bot=d_bot,
                Z_min=height_int.Z_min, Z_max=height_int.Z_max,
                a=a, b=b, confidence=height_int.confidence,
            )
        )

    return HADResult(
        aligned_depth=aligned,
        coverage_mask=coverage,
        anchors=anchors,
        n_masks_attempted=n_attempt,
        n_masks_solved=n_solved,
    )
