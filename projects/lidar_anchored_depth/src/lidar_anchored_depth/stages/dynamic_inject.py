"""``inject`` stage: place per-object accumulated snapshots into the
static cloud at one anchor timestamp.

Auto-discovers the input static cloud from
``<output.root>/<scene>/complete/latest/hybrid.ply`` unless
:attr:`InjectConfig.static_ply` overrides.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.configs.stages.inject import InjectConfig
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.engine.discovery import resolve_upstream_dir
from lidar_anchored_depth.pipeline.object_snapshot import (
    build_object_pose_cache,
    discover_object_clouds,
    inject_object_snapshots,
)
from lidar_anchored_depth.pipeline.v2x_static_classifier import (
    classify_v2x_objects,
    split_static_dynamic_ids,
)
from lidar_anchored_depth.reconstruction import (
    read_ply_xyz_rgb,
    voxel_downsample_robust,
    write_ply_xyz,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts
from lidar_anchored_depth.stages.dense_completion import _resolve_scene_id


def _discover_static_ply(cfg: InjectConfig | object, root: Path, scene: str) -> Path:
    """Resolve the upstream static cloud, falling back to the convention
    ``<root>/<scene>/complete/latest/hybrid.ply``."""
    explicit = getattr(cfg, "static_ply", None)
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"--static-ply {path} does not exist")
        return path
    latest = OutputManager.resolve_latest(root, scene, "complete")
    if latest is None:
        raise SystemExit(
            f"no upstream `complete` run found under {root}/{scene}/complete/. "
            f"Run `lad complete --scene.scene {scene}` first, or pass "
            f"--static-ply explicitly."
        )
    candidate = latest / "hybrid.ply"
    if not candidate.is_file():
        raise SystemExit(
            f"upstream {latest} has no hybrid.ply "
            f"(was --include-lidar-priority on?)"
        )
    return candidate


class DynamicInjectStage(Stage[InjectConfig]):
    """Layer 5: snapshot injection at a single anchor ts."""

    name = "inject"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out_dir = self.output_dir
        t_total = time.time()

        static_ply = _discover_static_ply(
            cfg, cfg.output.root, cfg.scene.scene,
        )
        print(f"[inject] static cloud <- {static_ply}")
        static_xyz, static_rgb = read_ply_xyz_rgb(static_ply)
        print(f"  loaded {static_xyz.shape[0]} static points")

        loader = RoadsideV2XLoader(
            data_root=cfg.scene.data_root,
            scenes=[cfg.scene.scene] if "_" in cfg.scene.scene else None,
            min_num_points=cfg.scene.loader_min_points,
            labels_source=cfg.scene.dynamic_labels_source,
        )
        if "_" not in cfg.scene.scene:
            loader.scene_filter = [cfg.scene.scene]
        scene_id = _resolve_scene_id(loader, cfg.scene.scene)
        cam_for_discovery = cfg.scene.cams[0]

        if cfg.recon_dir is None:
            recon_dir = resolve_upstream_dir(
                cfg.output.root, cfg.scene.scene, "object-accum",
                flag_hint="recon-dir",
            )
            print(f"[auto] recon-dir <- {recon_dir}")
        else:
            recon_dir = Path(cfg.recon_dir)
            if not recon_dir.is_dir():
                raise SystemExit(f"--recon-dir {recon_dir} not a directory")
            print(f"[inject] recon_dir = {recon_dir}")
        clouds = discover_object_clouds(recon_dir, prefer_icp=True)
        motion = classify_v2x_objects(
            loader, scene_id,
            cam_for_discovery=cam_for_discovery,
            drift_threshold_m=cfg.motion_drift_threshold_m,
        )
        static_ids, dyn_ids = split_static_dynamic_ids(motion)
        print(
            f"  motion: {len(static_ids)} static (always-on), "
            f"{len(dyn_ids)} dynamic (anchor only)"
        )

        t_cache = time.time()
        ts_cache = build_object_pose_cache(
            loader, scene_id, cam_for_discovery=cam_for_discovery,
        )
        print(
            f"  built object pose cache: {len(ts_cache)} obj_ids "
            f"({time.time() - t_cache:.1f}s)"
        )

        dyn_xyz, dyn_rgb, inj_log = inject_object_snapshots(
            loader, scene_id, clouds,
            cam_for_discovery=cam_for_discovery,
            anchor_ts_ms=int(cfg.anchor_ts_ms),
            strict_anchor=cfg.strict_anchor,
            object_fusion=cfg.object_fusion,
            object_voxel_size=cfg.object_voxel_size,
            object_max_dist_to_lidar=cfg.object_max_dist_to_lidar,
            mirror_axis=cfg.mirror_axis,
            mirror_classes=(
                frozenset(cfg.mirror_classes) if cfg.mirror_classes else None
            ),
            static_obj_ids=static_ids,
            ts_cache=ts_cache,
        )
        n_dyn = int(dyn_xyz.shape[0])
        n_kept = sum(1 for r in inj_log if r["n_pts"] > 0)
        print(f"  injected {n_dyn} pts across {n_kept}/{len(inj_log)} objs")

        # Stack on the static cloud + voxel-down to keep the output tidy.
        if n_dyn > 0:
            full_xyz = np.concatenate([static_xyz, dyn_xyz], axis=0)
            full_rgb = np.concatenate([static_rgb, dyn_rgb], axis=0)
        else:
            full_xyz = static_xyz
            full_rgb = static_rgb

        out_xyz, out_rgb = voxel_downsample_robust(
            full_xyz, cfg.object_voxel_size, colors=full_rgb,
        )
        hybrid_with_dyn = out_dir / "hybrid_with_dyn.ply"
        write_ply_xyz(
            hybrid_with_dyn, out_xyz, out_rgb, binary=cfg.ply_binary,
        )
        print(f"  hybrid_with_dyn voxels = {out_xyz.shape[0]}")

        summary = {
            "scene_id": scene_id,
            "anchor_ts_ms": int(cfg.anchor_ts_ms),
            "static_ply_in": str(static_ply),
            "n_static_in": int(static_xyz.shape[0]),
            "n_dynamic_points": n_dyn,
            "n_objs_kept": n_kept,
            "n_objs_total": len(inj_log),
            "n_voxels_out": int(out_xyz.shape[0]),
            "wall_time_s": round(time.time() - t_total, 2),
            "object_snapshots": inj_log,
        }
        return StageArtifacts(
            output_dir=out_dir,
            files={
                "hybrid_with_dyn": hybrid_with_dyn,
                "static_ply_in": Path(static_ply),
            },
            summary=summary,
        )
