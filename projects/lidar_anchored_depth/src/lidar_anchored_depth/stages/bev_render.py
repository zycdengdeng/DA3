"""``render-bev`` stage: per-ts BEV PNGs for video assembly.

The static cloud is pre-voxelised ONCE outside the per-ts loop —
otherwise ``np.unique`` on a 50 M-point cloud is re-run every frame
(45 min/frame in early iterations). Per-frame we only voxel-down
the (typically <1 M) dynamic part and concatenate before rasterising.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig
from lidar_anchored_depth.data import RoadsideV2XLoader
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
    voxel_downsample,
    voxel_downsample_robust,
    write_ply_xyz,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts
from lidar_anchored_depth.stages.dense_completion import _resolve_scene_id
from lidar_anchored_depth.stages.dynamic_inject import _discover_static_ply
from lidar_anchored_depth.viz.bev import auto_bev_range, render_bev


class BevRenderStage(Stage[BevRenderConfig]):
    """Render per-ts BEV PNGs by re-injecting dynamic objects with
    linear pose interpolation between annotated bracketing ts."""

    name = "render-bev"

    def run(self) -> StageArtifacts:
        try:
            from PIL import Image as PILImage
        except ModuleNotFoundError as e:
            raise SystemExit("render-bev requires Pillow") from e

        cfg = self.cfg
        out_dir = self.output_dir
        t_total = time.time()

        # ---- static cloud (auto-discover or override) ----
        static_ply = _discover_static_ply(
            cfg, cfg.output.root, cfg.scene.scene,
        )
        print(f"[render-bev] static cloud <- {static_ply}")
        static_xyz, static_rgb = read_ply_xyz_rgb(static_ply)
        print(f"  loaded {static_xyz.shape[0]} static points")

        # ---- loader + scene (10 Hz dynamic labels for smooth video) ----
        loader = RoadsideV2XLoader(
            data_root=cfg.scene.data_root,
            scenes=[cfg.scene.scene] if "_" in cfg.scene.scene else None,
            min_num_points=cfg.scene.loader_min_points,
            labels_source=cfg.scene.dynamic_labels_source,
        )
        if "_" not in cfg.scene.scene:
            loader.scene_filter = [cfg.scene.scene]
        scene_id = _resolve_scene_id(loader, cfg.scene.scene)
        scene = next(s for s in loader.scenes if s.scene_id == scene_id)
        cam_for_discovery = cfg.scene.cams[0]

        # ---- per-object pose cache + motion classification ----
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
        clouds = discover_object_clouds(recon_dir, prefer_icp=True)
        motion = classify_v2x_objects(
            loader, scene_id,
            cam_for_discovery=cam_for_discovery,
            drift_threshold_m=cfg.motion_drift_threshold_m,
        )
        static_ids, _ = split_static_dynamic_ids(motion)
        ts_cache = build_object_pose_cache(
            loader, scene_id, cam_for_discovery=cam_for_discovery,
        )
        print(
            f"  motion: {len(static_ids)} static, pose cache "
            f"{len(ts_cache)} obj_ids"
        )

        # ---- BEV viewport ----
        x_range = cfg.x_range
        y_range = cfg.y_range
        if x_range is None or y_range is None:
            auto_x, auto_y = auto_bev_range(static_xyz, pad_m=cfg.pad_m)
            x_range = x_range or auto_x
            y_range = y_range or auto_y
        print(
            f"  BEV  x={x_range}  y={y_range}  "
            f"size={cfg.image_size}"
        )

        # ---- pre-voxel static once ----
        bev_dir = out_dir / "bev_frames"
        bev_dir.mkdir(parents=True, exist_ok=True)
        ply_dir: Path | None = None
        if cfg.write_plys:
            ply_dir = out_dir / "frame_plys"
            ply_dir.mkdir(parents=True, exist_ok=True)

        print("  pre-voxelising static cloud (one-time)...")
        t_pv = time.time()
        static_v_xyz, static_v_rgb = voxel_downsample_robust(
            static_xyz, cfg.voxel_size, colors=static_rgb,
            color_method=cfg.robust_color,
        )
        print(
            f"    static {static_xyz.shape[0]} -> "
            f"{static_v_xyz.shape[0]} voxels  ({time.time() - t_pv:.1f}s)"
        )

        ts_list = list(scene.timestamps_ms)[::max(1, cfg.ts_stride)]
        print(f"  rendering {len(ts_list)} frames (stride={cfg.ts_stride})")
        mirror_classes = (
            frozenset(cfg.mirror_classes) if cfg.mirror_classes else None
        )

        t_video = time.time()
        for i, ts in enumerate(ts_list):
            dyn_xyz_t, dyn_rgb_t, _ = inject_object_snapshots(
                loader, scene_id, clouds,
                cam_for_discovery=cam_for_discovery,
                object_fusion=cfg.object_fusion,
                object_voxel_size=cfg.object_voxel_size,
                object_max_dist_to_lidar=cfg.object_max_dist_to_lidar,
                mirror_axis=cfg.mirror_axis,
                mirror_classes=mirror_classes,
                static_obj_ids=static_ids,
                ts_cache=ts_cache,
                video_anchor_ts_ms=int(ts),
                video_max_extrap_ms=cfg.max_extrap_ms,
                video_max_gap_ms=cfg.max_gap_ms,
            )

            if dyn_xyz_t.shape[0] > 0:
                dyn_v_xyz, dyn_v_rgb = voxel_downsample(
                    dyn_xyz_t, cfg.voxel_size, colors=dyn_rgb_t,
                )
                full_xyz = np.concatenate([static_v_xyz, dyn_v_xyz], axis=0)
                full_rgb = np.concatenate([static_v_rgb, dyn_v_rgb], axis=0)
            else:
                full_xyz = static_v_xyz
                full_rgb = static_v_rgb

            bev_img = render_bev(
                full_xyz, full_rgb,
                x_range=x_range, y_range=y_range,
                image_hw=tuple(cfg.image_size),
            )
            PILImage.fromarray(bev_img).save(bev_dir / f"bev_{i:04d}.png")
            if ply_dir is not None:
                write_ply_xyz(
                    ply_dir / f"frame_{i:04d}.ply",
                    full_xyz, full_rgb, binary=True,
                )

            if (i + 1) % 10 == 0 or i == len(ts_list) - 1:
                elapsed = time.time() - t_video
                print(
                    f"  frame {i + 1:>4}/{len(ts_list)}  ts={ts}  "
                    f"N={full_xyz.shape[0]}  "
                    f"({elapsed:.1f}s, {(i + 1) / elapsed:.2f} fps)"
                )

        wall = time.time() - t_total
        print()
        print(f"  ↳ {len(ts_list)} BEV PNGs in {bev_dir}")
        print("  ↳ to assemble video:")
        print(
            f"     ffmpeg -framerate 10 -i {bev_dir}/bev_%04d.png "
            f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
            f"{out_dir}/bev_video.mp4"
        )

        summary = {
            "scene_id": scene_id,
            "static_ply_in": str(static_ply),
            "n_frames": len(ts_list),
            "ts_stride": cfg.ts_stride,
            "x_range": list(x_range),
            "y_range": list(y_range),
            "image_size": list(cfg.image_size),
            "wall_time_s": round(wall, 2),
        }
        return StageArtifacts(
            output_dir=out_dir,
            files={
                "bev_dir": bev_dir,
                "static_ply_in": Path(static_ply),
            },
            summary=summary,
        )
