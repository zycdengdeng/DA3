"""Inject per-object reconstructions back into the world scene.

The ``run_object_accumulation.py`` pipeline produces, per V2X bbox.id,
two clouds in the *object-local* frame:

    <scene>_obj{id}_lidar.ply    # accumulated LiDAR returns
    <scene>_obj{id}_aa_had.ply   # accumulated AA-HAD predictions

The per-object track is the high-quality dynamic-object reconstruction
(median chamfer 0.27–0.38 m, vs 1+ m if we tried to predict each car
frame-by-frame in the static branch). To put dynamic objects back into
the static scene cleanly:

    1. Pick a *representative* timestamp per object — the ts where its
       per-object cloud is most supported, or a user-specified anchor.
    2. Look up that ts's V2X bbox pose (roll, pitch, yaw, xyz).
    3. Apply ``object_local_to_world`` with that pose to the per-object
       cloud — yields a "snapshot" of the object at that moment.
    4. Concatenate all snapshots into the scene cloud.

The result replaces the ghosted/scattered per-frame static-branch
reconstruction of dynamic objects with the well-reconstructed
per-object cloud, placed at its correct world pose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.data.roadside_v2x import RoadsideV2XLoader
from lidar_anchored_depth.pipeline.lidar_completion import lidar_priority_fill
from lidar_anchored_depth.reconstruction.io import read_ply_xyz, read_ply_xyz_rgb
from lidar_anchored_depth.reconstruction.object_local import object_local_to_world


# Vehicle classes whose left-right symmetry across the object-local
# y-axis (= bbox lateral axis) holds well enough that mirroring the
# observed half across the symmetry plane is a sound geometric prior.
_DEFAULT_MIRROR_CLASSES = frozenset({"Car", "Suv", "Bus", "Truck"})


def _mirror_object_local(points: np.ndarray, axis: str) -> np.ndarray:
    """Reflect ``points`` across the given axis in object-local frame."""
    if points.size == 0:
        return points
    out = points.copy()
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    out[:, idx] *= -1.0
    return out


def fuse_object_clouds(
    aa_had_local: np.ndarray | None,
    aa_had_colors: np.ndarray | None,
    lidar_local: np.ndarray | None,
    *,
    voxel_size: float = 0.05,
    max_dist_to_lidar: float | None = 0.5,
    mirror_axis: str | None = None,
    object_lidar_color: tuple[int, int, int] = (160, 160, 160),
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Combine per-object LiDAR (primary) and AA-HAD (fill) in
    object-local frame, with optional left-right symmetry mirror.

    Steps
    -----
    1. Optionally mirror both LiDAR and AA-HAD across ``mirror_axis``
       (concatenated to the originals, doubling coverage on the
       unobserved side).
    2. Run :func:`lidar_priority_fill` on the (possibly mirrored)
       clouds: LiDAR voxels win, AA-HAD only fills empty voxels,
       AA-HAD points farther than ``max_dist_to_lidar`` from any
       LiDAR neighbour are dropped (sanity).

    Returns ``(xyz, rgb, source)`` where source ``0`` = LiDAR,
    ``1`` = AA-HAD, all in object-local frame.
    """
    aa_xyz = (
        aa_had_local.astype(np.float64) if aa_had_local is not None and aa_had_local.size else
        np.zeros((0, 3), dtype=np.float64)
    )
    aa_rgb = (
        aa_had_colors.astype(np.uint8) if aa_had_colors is not None and aa_had_colors.size else
        None
    )
    lid_xyz = (
        lidar_local.astype(np.float64) if lidar_local is not None and lidar_local.size else
        np.zeros((0, 3), dtype=np.float64)
    )

    if mirror_axis is not None:
        if aa_xyz.shape[0] > 0:
            aa_xyz = np.concatenate([aa_xyz, _mirror_object_local(aa_xyz, mirror_axis)], axis=0)
            if aa_rgb is not None:
                aa_rgb = np.concatenate([aa_rgb, aa_rgb], axis=0)
        if lid_xyz.shape[0] > 0:
            lid_xyz = np.concatenate([lid_xyz, _mirror_object_local(lid_xyz, mirror_axis)], axis=0)

    xyz, rgb, source = lidar_priority_fill(
        lid_xyz, aa_xyz, aa_rgb,
        voxel_size=voxel_size,
        max_dist_to_lidar=max_dist_to_lidar,
        lidar_color=object_lidar_color,
    )
    return xyz, rgb, source


_OBJ_PATTERN = re.compile(
    r"^(?P<stem>.+)_obj(?P<id>\d+)_(?P<kind>lidar|aa_had|aa_had_icp)\.ply$"
)


@dataclass(slots=True)
class ObjectClouds:
    """Per-object accumulated clouds in object-local frame."""

    obj_id: int
    aa_had_local: np.ndarray | None  # (N, 3) or None
    aa_had_colors: np.ndarray | None  # (N, 3) uint8 or None
    lidar_local: np.ndarray | None  # (N, 3) or None
    aa_had_path: Path | None = None
    lidar_path: Path | None = None


def discover_object_clouds(
    recon_dir: Path,
    *,
    prefer_icp: bool = True,
) -> dict[int, ObjectClouds]:
    """Walk a recon directory and load every available per-object PLY.

    ``prefer_icp`` (default ``True``) selects ``aa_had_icp.ply`` over
    ``aa_had.ply`` when both exist for the same object — that file is
    the post-ICP refinement and has the lowest chamfer to LiDAR.
    """
    out: dict[int, ObjectClouds] = {}

    aa_files: dict[int, Path] = {}
    aa_icp_files: dict[int, Path] = {}
    lidar_files: dict[int, Path] = {}
    for ply in sorted(recon_dir.glob("*_obj*_*.ply")):
        m = _OBJ_PATTERN.match(ply.name)
        if m is None:
            continue
        oid = int(m.group("id"))
        kind = m.group("kind")
        if kind == "aa_had":
            aa_files[oid] = ply
        elif kind == "aa_had_icp":
            aa_icp_files[oid] = ply
        elif kind == "lidar":
            lidar_files[oid] = ply

    all_ids = set(aa_files) | set(aa_icp_files) | set(lidar_files)
    for oid in sorted(all_ids):
        aa_path = (
            aa_icp_files.get(oid)
            if prefer_icp and oid in aa_icp_files
            else aa_files.get(oid)
        )
        aa_pts: np.ndarray | None = None
        aa_col: np.ndarray | None = None
        if aa_path is not None and aa_path.is_file():
            aa_pts, aa_col = read_ply_xyz_rgb(aa_path)
            if aa_col is not None and aa_col.size == 0:
                aa_col = None
        lidar_path = lidar_files.get(oid)
        lidar_pts: np.ndarray | None = None
        if lidar_path is not None and lidar_path.is_file():
            lidar_pts = read_ply_xyz(lidar_path)
        out[oid] = ObjectClouds(
            obj_id=oid,
            aa_had_local=aa_pts,
            aa_had_colors=aa_col,
            lidar_local=lidar_pts,
            aa_had_path=aa_path,
            lidar_path=lidar_path,
        )
    return out


def build_object_pose_cache(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_for_discovery: str = "0",
) -> dict[int, list[tuple[int, DynamicObject]]]:
    """Walk every (ts, cam_for_discovery) frame in the scene ONCE and
    collect, per V2X obj_id, the list of ``(ts, DynamicObject)`` pairs.

    Use the returned dict with :func:`inject_object_snapshots` via the
    ``ts_cache=`` argument: it skips the per-object inner loop that
    otherwise reloads every frame for every object on every call,
    which dominates the runtime of video-mode rendering.
    """
    scene = next((s for s in loader.scenes if s.scene_id == scene_id), None)
    if scene is None:
        return {}
    cache: dict[int, list[tuple[int, DynamicObject]]] = {}
    for ts in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts, cam_for_discovery)
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        for obj in frame.dynamic_objects or []:
            cache.setdefault(int(obj.id), []).append((int(ts), obj))
    return cache


def _pick_object_ts_cached(
    cache: dict[int, list[tuple[int, DynamicObject]]],
    obj_id: int,
    *,
    anchor_ts_ms: int | None = None,
    strict_anchor: bool = False,
) -> tuple[int, DynamicObject] | None:
    """O(1) cache lookup version of ``_pick_object_ts``."""
    matches = cache.get(int(obj_id))
    if not matches:
        return None
    if anchor_ts_ms is not None:
        for t, obj in matches:
            if t == int(anchor_ts_ms):
                return t, obj
        if strict_anchor:
            return None
    matches_sorted = sorted(matches, key=lambda p: p[0])
    return matches_sorted[len(matches_sorted) // 2]


def _angle_lerp(a: float, b: float, alpha: float) -> float:
    """Linearly interpolate between two angles (radians) along the
    shortest direction; result is in ``[-pi, pi]`` modulo wrap."""
    import math
    diff = ((b - a) + math.pi) % (2.0 * math.pi) - math.pi
    return float(a + alpha * diff)


def _pick_object_ts_video(
    cache: dict[int, list[tuple[int, DynamicObject]]],
    obj_id: int,
    current_ts_ms: int,
    *,
    max_extrap_ms: int = 200,
    max_gap_ms: int = 500,
) -> tuple[int, DynamicObject] | None:
    """Video-mode picker: linearly interpolate the object's pose
    between its two surrounding annotated ts; return ``None`` when

    * ``current_ts_ms`` is outside the object's annotation window
      plus ``max_extrap_ms`` tolerance, or
    * ``current_ts_ms`` falls inside a gap between two consecutive
      annotations that are farther apart than ``max_gap_ms``. Such a
      gap usually means V2X tracking lost the object — interpolating
      across it would slide a parked car across the scene to where
      a re-acquired track happens to be (the user's reported "car
      teleporting back to the intersection").

    This replaces the older "fall back to median ts" behaviour, which
    made sparsely-annotated objects look stationary at one fixed
    position across the whole video and never disappear.
    """
    import bisect
    from dataclasses import replace

    matches = cache.get(int(obj_id))
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda p: p[0])
    timestamps = [m[0] for m in matches_sorted]
    current_ts_ms = int(current_ts_ms)

    if current_ts_ms < timestamps[0] - int(max_extrap_ms):
        return None
    if current_ts_ms > timestamps[-1] + int(max_extrap_ms):
        return None

    idx = bisect.bisect_left(timestamps, current_ts_ms)

    # Exact match
    if idx < len(timestamps) and timestamps[idx] == current_ts_ms:
        return current_ts_ms, matches_sorted[idx][1]

    # Within the extrapolation tolerance before first annotation
    if idx == 0:
        return current_ts_ms, matches_sorted[0][1]
    # ... or after last
    if idx >= len(timestamps):
        return current_ts_ms, matches_sorted[-1][1]

    t_prev = timestamps[idx - 1]
    t_next = timestamps[idx]

    # Tracker-loss detection: if the bracketing annotations are far
    # apart, we are inside an annotation gap that almost certainly
    # came from V2X failing to track this object across that span.
    # Don't render — the object truly isn't visible there.
    if (t_next - t_prev) > int(max_gap_ms):
        return None

    obj_prev = matches_sorted[idx - 1][1]
    obj_next = matches_sorted[idx][1]
    alpha = (current_ts_ms - t_prev) / max(t_next - t_prev, 1)

    xyz_prev = np.asarray(obj_prev.xyz, dtype=np.float64)
    xyz_next = np.asarray(obj_next.xyz, dtype=np.float64)
    interp_xyz = ((1.0 - alpha) * xyz_prev + alpha * xyz_next)
    if hasattr(obj_prev.xyz, "tolist"):
        interp_xyz = np.asarray(interp_xyz, dtype=xyz_prev.dtype)

    interp_yaw = _angle_lerp(float(obj_prev.yaw), float(obj_next.yaw), alpha)
    interp_pitch = _angle_lerp(float(obj_prev.pitch), float(obj_next.pitch), alpha)
    interp_roll = _angle_lerp(float(obj_prev.roll), float(obj_next.roll), alpha)

    interp_obj = replace(
        obj_prev,
        xyz=interp_xyz,
        yaw=interp_yaw,
        pitch=interp_pitch,
        roll=interp_roll,
    )
    return current_ts_ms, interp_obj


def _pick_object_ts(
    loader: RoadsideV2XLoader,
    scene_id: str,
    obj_id: int,
    cam_for_discovery: str,
    *,
    anchor_ts_ms: int | None = None,
    strict_anchor: bool = False,
) -> tuple[int, DynamicObject] | None:
    """Pick a representative timestamp for an object.

    Strategy: prefer the user-specified ``anchor_ts_ms`` if the object
    is annotated there. If ``strict_anchor`` is ``True`` and the
    anchor ts is not annotated for this object, return ``None`` (drop
    the object). Otherwise fall back to the median ts.
    """
    scene = next((s for s in loader.scenes if s.scene_id == scene_id), None)
    if scene is None:
        return None
    matches: list[tuple[int, DynamicObject]] = []
    for ts in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts, cam_for_discovery)
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        for obj in frame.dynamic_objects or []:
            if int(obj.id) == int(obj_id):
                matches.append((int(ts), obj))
                break
    if not matches:
        return None
    if anchor_ts_ms is not None:
        for t, obj in matches:
            if t == int(anchor_ts_ms):
                return t, obj
        if strict_anchor:
            return None
    matches.sort(key=lambda p: p[0])
    return matches[len(matches) // 2]


def inject_object_snapshots(
    loader: RoadsideV2XLoader,
    scene_id: str,
    object_clouds: dict[int, ObjectClouds],
    *,
    cam_for_discovery: str = "0",
    anchor_ts_ms: int | None = None,
    strict_anchor: bool = False,
    use_lidar: bool = False,
    fallback_color: tuple[int, int, int] = (220, 220, 100),
    object_fusion: str = "lidar-priority",
    object_voxel_size: float = 0.05,
    object_max_dist_to_lidar: float | None = 0.5,
    mirror_axis: str | None = None,
    mirror_classes: frozenset[str] | None = None,
    static_obj_ids: set[int] | None = None,
    ts_cache: dict[int, list[tuple[int, "DynamicObject"]]] | None = None,
    video_anchor_ts_ms: int | None = None,
    video_max_extrap_ms: int = 200,
    video_max_gap_ms: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Place per-object clouds at their representative ts pose.

    Parameters
    ----------
    object_clouds : output of :func:`discover_object_clouds`.
    cam_for_discovery : camera id used to query V2X annotations
        (annotations are camera-independent, so any camera works as
        long as it is sampled).
    anchor_ts_ms : optional global anchor; objects annotated at this
        ts are placed at that pose, others fall back to their median ts
        (or are dropped if ``strict_anchor=True``).
    strict_anchor : if ``True`` AND ``anchor_ts_ms`` is set, drop any
        object that is not annotated at the anchor ts. Yields a
        single-frame snapshot of the intersection (no temporal smear)
        at the cost of fewer visible objects. Static objects in
        ``static_obj_ids`` are NEVER dropped (they don't move).
    use_lidar : (legacy) when ``object_fusion == 'aa-had-only'``, also
        place the per-object LiDAR cloud alongside the AA-HAD cloud.
        Ignored for ``lidar-priority`` and ``lidar-only`` modes.
    fallback_color : RGB used when the per-object PLY has no colour
        (e.g., a LiDAR-only object).
    object_fusion : one of ``'lidar-priority'`` (default; LiDAR wins
        per-voxel, AA-HAD fills empty), ``'lidar-only'`` (drop AA-HAD
        entirely; clean but only the side LiDAR could see),
        ``'aa-had-only'`` (legacy: only the AA-HAD cloud, with shells).
    object_voxel_size : voxel size used for the per-object LiDAR-
        priority fusion in object-local frame (default 5 cm).
    object_max_dist_to_lidar : drop AA-HAD object-local points whose
        distance to nearest LiDAR neighbour exceeds this (default
        0.5 m; tighter than the static-scene 2 m because objects are
        physically small).
    mirror_axis : if set (``'x'``, ``'y'``, ``'z'``), mirror the per-
        object cloud across that axis in object-local frame to fill
        the unobserved side. Default ``None`` = no mirror.
    mirror_classes : set of class labels for which mirroring is
        applied. When ``None`` defaults to the four-wheel vehicle
        classes (Car, Suv, Bus, Truck). Pedestrians, riders, etc.
        are NOT mirrored.
    static_obj_ids : optional set of obj_ids treated as static.
        Static objects bypass the strict-anchor drop (they appear in
        the snapshot even if not annotated at the anchor ts).
    """
    pts_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    log: list[dict] = []

    if mirror_classes is None:
        mirror_classes = _DEFAULT_MIRROR_CLASSES
    static_set = set(static_obj_ids or [])

    for obj_id, clouds in object_clouds.items():
        is_static = int(obj_id) in static_set
        if video_anchor_ts_ms is not None:
            # Video mode: interpolate pose between the two annotated
            # ts that bracket video_anchor_ts_ms; drop the object
            # entirely if that ts is outside its annotation window
            # (avoids the old "stuck at median pose, then disappears"
            # behaviour the user reported).
            if ts_cache is None:
                # Build a tiny per-call cache as a fallback.
                ts_cache_local: dict[int, list[tuple[int, DynamicObject]]] = {}
                scene = next(
                    (s for s in loader.scenes if s.scene_id == scene_id), None,
                )
                if scene is not None:
                    for ts in scene.timestamps_ms:
                        try:
                            idx = loader.find_frame_idx(scene_id, ts, cam_for_discovery)
                        except ValueError:
                            continue
                        frame = loader.get_frame(idx)
                        for obj in frame.dynamic_objects or []:
                            if int(obj.id) == int(obj_id):
                                ts_cache_local.setdefault(int(obj_id), []).append(
                                    (int(ts), obj),
                                )
                                break
                pick = _pick_object_ts_video(
                    ts_cache_local, obj_id, int(video_anchor_ts_ms),
                    max_extrap_ms=video_max_extrap_ms,
                    max_gap_ms=video_max_gap_ms,
                )
            else:
                pick = _pick_object_ts_video(
                    ts_cache, obj_id, int(video_anchor_ts_ms),
                    max_extrap_ms=video_max_extrap_ms,
                    max_gap_ms=video_max_gap_ms,
                )
        elif ts_cache is not None:
            pick = _pick_object_ts_cached(
                ts_cache, obj_id,
                anchor_ts_ms=anchor_ts_ms,
                strict_anchor=strict_anchor and not is_static,
            )
        else:
            pick = _pick_object_ts(
                loader, scene_id, obj_id, cam_for_discovery,
                anchor_ts_ms=anchor_ts_ms,
                strict_anchor=strict_anchor and not is_static,
            )
        if pick is None:
            reason = (
                "not_at_anchor_ts"
                if strict_anchor and anchor_ts_ms is not None
                else "no_v2x_match"
            )
            log.append({"obj_id": obj_id, "ts_ms": None, "n_pts": 0,
                        "source": reason})
            continue
        ts_ms, obj = pick
        label = getattr(obj, "label", "Unknown")
        do_mirror = (mirror_axis is not None) and (label in mirror_classes)

        if object_fusion == "lidar-priority":
            xyz_local, rgb_local, source = fuse_object_clouds(
                aa_had_local=clouds.aa_had_local,
                aa_had_colors=clouds.aa_had_colors,
                lidar_local=clouds.lidar_local,
                voxel_size=object_voxel_size,
                max_dist_to_lidar=object_max_dist_to_lidar,
                mirror_axis=mirror_axis if do_mirror else None,
            )
            sources_used = ["lidar-priority"]
            if do_mirror:
                sources_used.append(f"mirror-{mirror_axis}")
            n_total = int(xyz_local.shape[0])
            if n_total == 0:
                log.append({"obj_id": int(obj_id), "ts_ms": int(ts_ms),
                            "n_pts": 0, "source": "empty",
                            "label": label, "is_static": is_static})
                continue
            P_world = object_local_to_world(xyz_local, obj)
            pts_chunks.append(P_world.astype(np.float32))
            if rgb_local is not None:
                rgb_chunks.append(rgb_local.astype(np.uint8))
            else:
                rgb_chunks.append(
                    np.full((P_world.shape[0], 3), fallback_color, dtype=np.uint8)
                )
            log.append({
                "obj_id": int(obj_id), "ts_ms": int(ts_ms),
                "n_pts": n_total, "source": "+".join(sources_used),
                "label": label, "is_static": is_static,
                "n_lidar": int((source == 0).sum()),
                "n_aahad_fill": int((source == 1).sum()),
            })
            continue

        # Legacy paths (lidar-only / aa-had-only) ----------------------
        sources: list[str] = []
        n_total = 0
        if object_fusion == "aa-had-only" and clouds.aa_had_local is not None and clouds.aa_had_local.size > 0:
            aa_local = clouds.aa_had_local
            aa_col = clouds.aa_had_colors
            if do_mirror:
                aa_local = np.concatenate(
                    [aa_local, _mirror_object_local(aa_local, mirror_axis)], axis=0,
                )
                if aa_col is not None:
                    aa_col = np.concatenate([aa_col, aa_col], axis=0)
            P_world = object_local_to_world(aa_local, obj)
            pts_chunks.append(P_world.astype(np.float32))
            if aa_col is not None:
                rgb_chunks.append(aa_col.astype(np.uint8))
            else:
                rgb_chunks.append(
                    np.full((P_world.shape[0], 3), fallback_color, dtype=np.uint8)
                )
            sources.append("aa_had")
            n_total += int(P_world.shape[0])

        if (
            (object_fusion == "lidar-only" or (object_fusion == "aa-had-only" and use_lidar))
            and clouds.lidar_local is not None and clouds.lidar_local.size > 0
        ):
            lid_local = clouds.lidar_local
            if do_mirror:
                lid_local = np.concatenate(
                    [lid_local, _mirror_object_local(lid_local, mirror_axis)], axis=0,
                )
            P_world_l = object_local_to_world(lid_local, obj)
            pts_chunks.append(P_world_l.astype(np.float32))
            rgb_chunks.append(
                np.full((P_world_l.shape[0], 3), fallback_color, dtype=np.uint8)
            )
            sources.append("lidar")
            n_total += int(P_world_l.shape[0])

        if do_mirror and sources:
            sources.append(f"mirror-{mirror_axis}")
        log.append({
            "obj_id": int(obj_id),
            "ts_ms": int(ts_ms),
            "n_pts": n_total,
            "source": "+".join(sources) if sources else "empty",
            "label": label,
            "is_static": is_static,
        })

    if pts_chunks:
        return (
            np.concatenate(pts_chunks, axis=0),
            np.concatenate(rgb_chunks, axis=0),
            log,
        )
    return (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.uint8),
        log,
    )


__all__ = [
    "ObjectClouds",
    "build_object_pose_cache",
    "discover_object_clouds",
    "fuse_object_clouds",
    "inject_object_snapshots",
]
