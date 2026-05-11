# CLI reference

`lad` is the single top-level command, with subcommands for each
stage and for the multi-stage orchestrator. Every flag comes from a
typed `@dataclass`, so `lad <sub> --help` is the authoritative source
for the current field list. This page calls out only the flags users
ask about most often.

## Common conventions

* List flags use **spaces**, not commas:
  `--runtime.gpu-ids 0 1 2 3` (not `0,1,2,3`).
* Dotted names traverse nested dataclasses:
  `--scene.scene 008`, `--complete.scene.data-root /mnt/...`.
* Boolean flags have explicit `--flag` / `--no-flag` forms.
* Each stage writes into `outputs/<scene>/<stage>/<run_id>/` and refreshes
  the sibling `latest` symlink on success.

## `lad complete`

Runs the per-(cam, ts) point-completion refine + Layer 4 LiDAR-priority
voxel fuse. Outputs `refined.ply`, `baseline.ply`, `hybrid.ply`.

```bash
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --scene.cams 0 3 6 9 \
    --runtime.gpu-ids 0 1 2 3 \
    --d-paths-glob 'preview/da3/008_*_d.npz' \
    --sam-mask-dir preview/sam/ \
    --segformer-dir preview/segformer/ \
    --calib-json preview/scene_recon/008_*_static_calib.json \
    --residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --object-anchor-ts-ms 1742879650704 \
    --robust-color median \
    --output.root outputs
```

### Most-asked flags

| Flag                           | Default     | Notes                                                                |
| ------------------------------ | ----------- | -------------------------------------------------------------------- |
| `--runtime.gpu-ids`            | (serial)    | Set to multi-value (e.g. `0 1 2 3`) to shard refine across GPUs.     |
| `--robust-color`               | `median`    | Per-voxel colour aggregation when multiple frames hit one voxel.    |
| `--lidar-skip-ground`          | on          | Drops LiDAR points in the ground band so road density is uniform.   |
| `--include-lidar-priority`     | on          | Stage 4 fuse. Set `--no-include-lidar-priority` to skip it.         |
| `--colorize-lidar`             | on          | Re-colour LiDAR-source points from anchor cameras.                  |
| `--object-anchor-ts-ms`        | required for colorize | ts of the anchor cameras used for image RGB transfer.   |
| `--scene.static-labels-source` | `merged_pcd`| Hand-labeled bboxes for static cull. Fall back to `interpolation` if scene has no hand-labels. |
| `--max-points-per-frame`       | 12000       | Per-frame cap before the residual flow head (memory + KNN cost).    |
| `--aahad-pixel-stride`         | 4           | Stride on depth pixels when building the Layer 1+3 prior.           |

## `lad inject`

Places per-object accumulated PLYs into the static cloud at one anchor ts.

```bash
lad inject \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --recon-dir preview/recon_3I_full/ \
    --anchor-ts-ms 1742879650704 \
    --mirror-axis y
```

Auto-reads `outputs/<scene>/complete/latest/hybrid.ply` unless you pass
`--static-ply <path>` explicitly.

| Flag                       | Default            | Notes                                                  |
| -------------------------- | ------------------ | ------------------------------------------------------ |
| `--recon-dir`              | (required)         | Dir of per-object PLYs from `run_object_accumulation.py`. |
| `--anchor-ts-ms`           | (required)         | ms timestamp to place all objects at.                   |
| `--object-fusion`          | `lidar-priority`   | Per-object fusion mode.                                 |
| `--mirror-axis`            | None               | Mirror per-object cloud across this local-frame axis (handles single-side observations). |
| `--motion-drift-threshold-m` | 0.5              | Above this drift between min/max ts, object is "dynamic"; below it's "static" (always-on). |

## `lad render-bev`

Renders per-ts BEV PNGs with linear pose-interpolated dynamic objects.

```bash
lad render-bev \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --recon-dir preview/recon_3I_full/ \
    --ts-stride 1 \
    --image-size 1024 1024 \
    --max-extrap-ms 200 \
    --max-gap-ms 500
```

At the end of the run the stage prints the `ffmpeg` one-liner to
assemble the PNGs into an mp4.

| Flag                  | Default   | Notes                                                          |
| --------------------- | --------- | -------------------------------------------------------------- |
| `--ts-stride`         | 1         | Render every Nth ts. Use 2-3 for fast preview.                 |
| `--max-extrap-ms`     | 200       | Drop objects whose video ts is outside the annotation window by more than this. |
| `--max-gap-ms`        | 500       | Drop objects whose bracketing annotated ts are farther apart than this (V2X tracker loss). |
| `--write-plys`        | off       | Also write per-frame PLY (debug; disk-heavy).                  |
| `--robust-color`      | `median`  | Per-voxel colour for the one-time static pre-voxelisation.     |
| `--image-size`        | `1024 1024` | Output PNG (H, W).                                            |
| `--x-range` / `--y-range` | auto | World metres; auto-computed from the static cloud if omitted.  |

## `lad pipeline`

Runs `complete` → `inject` → `render-bev` back-to-back. Set
`scene` / `output` / `runtime` once on the `--complete.*` block; the
orchestrator propagates them.

```bash
lad pipeline \
    --complete.scene.scene 008 \
    --complete.scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.d-paths-glob 'preview/da3/008_*_d.npz' \
    --complete.calib-json preview/scene_recon/008_*_static_calib.json \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --complete.object-anchor-ts-ms 1742879650704 \
    --inject.recon-dir preview/recon_3I_full/ \
    --inject.anchor-ts-ms 1742879650704 \
    --render-bev.recon-dir preview/recon_3I_full/
```

Subset / resume:

```bash
# Re-render BEV without recomputing the static cloud
lad pipeline --stages render-bev ...

# Continue after an inject failure
lad pipeline --from-stage inject ...
```

## `lad info` / `lad version`

Smoke-test the CLI:

```bash
lad version             # prints the package version
lad info                # prints config tree for the args you passed
```

## Tab completion (optional)

`tyro` exposes a `--tyro-write-completion` hook for shells that support
it; see [`tyro` docs](https://brentyi.github.io/tyro/) for installation.
