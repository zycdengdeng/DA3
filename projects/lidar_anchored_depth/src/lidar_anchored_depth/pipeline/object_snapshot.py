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
from lidar_anchored_depth.reconstruction.io import read_ply_xyz, read_ply_xyz_rgb
from lidar_anchored_depth.reconstruction.object_local import object_local_to_world


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
        at the cost of fewer visible objects.
    use_lidar : if ``True``, also place the per-object LiDAR cloud
        alongside the AA-HAD cloud (in case the per-object LiDAR is
        denser than the static-branch dynamic LiDAR).
    fallback_color : RGB used when the per-object PLY has no colour
        (e.g., a LiDAR-only object).

    Returns
    -------
    points_world : (N, 3) float32
    colors_rgb : (N, 3) uint8
    log : list of per-object dicts ``{obj_id, ts_ms, n_pts, source}``
    """
    pts_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    log: list[dict] = []

    for obj_id, clouds in object_clouds.items():
        pick = _pick_object_ts(
            loader, scene_id, obj_id, cam_for_discovery,
            anchor_ts_ms=anchor_ts_ms,
            strict_anchor=strict_anchor,
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

        sources: list[str] = []
        n_total = 0
        if clouds.aa_had_local is not None and clouds.aa_had_local.size > 0:
            P_world = object_local_to_world(clouds.aa_had_local, obj)
            pts_chunks.append(P_world.astype(np.float32))
            if clouds.aa_had_colors is not None:
                rgb_chunks.append(clouds.aa_had_colors.astype(np.uint8))
            else:
                rgb_chunks.append(
                    np.full((P_world.shape[0], 3), fallback_color, dtype=np.uint8)
                )
            sources.append("aa_had")
            n_total += int(P_world.shape[0])

        if use_lidar and clouds.lidar_local is not None and clouds.lidar_local.size > 0:
            P_world_l = object_local_to_world(clouds.lidar_local, obj)
            pts_chunks.append(P_world_l.astype(np.float32))
            rgb_chunks.append(
                np.full((P_world_l.shape[0], 3), fallback_color, dtype=np.uint8)
            )
            sources.append("lidar")
            n_total += int(P_world_l.shape[0])

        log.append({
            "obj_id": int(obj_id),
            "ts_ms": int(ts_ms),
            "n_pts": n_total,
            "source": "+".join(sources) if sources else "empty",
            "label": getattr(obj, "label", "Unknown"),
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
    "discover_object_clouds",
    "inject_object_snapshots",
]
