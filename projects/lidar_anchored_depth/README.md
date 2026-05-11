# LAD-DA3 — LiDAR-Anchored Depth for Foundation Depth Models

> Roadside / V2X depth reconstruction on top of
> [Depth Anything 3 (DA3)](../../README.md): anchor a foundation depth
> model on **sparse LiDAR object heights**, refine with a point-cloud
> flow head, and fuse multi-camera multi-frame returns into a dense
> 3D scene with smooth dynamic objects.

**Status:** under active development. The pipeline is feature-complete
end-to-end; tuning + paper write-up in progress.

## What you get

A typed multi-stage CLI (`lad`) that turns a roadside V2X capture into:

* **`refined.ply`** — Stage 2 dense point-completion (DA3 + AA-HAD +
  trained residual flow head).
* **`hybrid.ply`** — Stage 4 LiDAR-priority fusion: cm-precise LiDAR
  wins per voxel, refined fills the rest. Ground band of the LiDAR
  is dropped before fusion so the road density is uniform (no scan-ring
  artefacts).
* **`hybrid_with_dyn.ply`** — Stage 5 per-object accumulated dynamic
  objects placed at one anchor timestamp.
* **`bev_frames/*.png`** — per-ts BEV PNGs with linear-pose-interpolated
  dynamic objects; pipe into `ffmpeg` for video.

Each stage writes into a timestamped directory and refreshes a sibling
`latest` symlink, so downstream stages discover their input by
convention.

## Install

```bash
pip install -e projects/lidar_anchored_depth
```

This installs the `lad` CLI:

```bash
lad --help
# Subcommands: info, version, complete, inject, render-bev, pipeline
```

## Quickstart — one full pipeline run

```bash
lad pipeline \
    --complete.scene.scene 008 \
    --complete.scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.d-paths-glob 'preview/da3/008_*_d.npz' \
    --complete.sam-mask-dir preview/sam/ \
    --complete.segformer-dir preview/segformer/ \
    --complete.calib-json preview/scene_recon/008_*_static_calib.json \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --complete.object-anchor-ts-ms 1742879650704 \
    --inject.recon-dir preview/recon_3I_full/ \
    --inject.anchor-ts-ms 1742879650704 \
    --render-bev.recon-dir preview/recon_3I_full/
```

Outputs land at `outputs/008/{complete,inject,render-bev}/<timestamp>/`,
each with `config.json`, `summary.json`, and the stage's artefacts.

To run the stages individually (e.g. iterating on render parameters):

```bash
lad complete    --scene.scene 008 --runtime.gpu-ids 0 1 2 3 ...
lad inject      --scene.scene 008 --recon-dir preview/recon_3I_full/ \
                --anchor-ts-ms 1742879650704
lad render-bev  --scene.scene 008 --recon-dir preview/recon_3I_full/
```

`inject` and `render-bev` auto-discover the upstream
`outputs/<scene>/complete/latest/hybrid.ply`.

## Method (HAD — Height-Anchored Depth)

```
   Roadside RGB  ─►  DA3  ─►  d̃  (relative depth)
                                    │
                  SAM  ─►  {m_i}    │
                            │       │
   LiDAR (world) ─►  per-mask Z range [Z_min^i, Z_max^i]
                            │       │
                            ▼       ▼
                  closed-form solve (a_i, b_i) →  AA-HAD calibration
                            │
                            ▼
                  metric depth z(u,v)  +  ground-plane snap
                            │
                            ▼
                  point-completion network (DGCNN + CFM)
                            │
                            ▼
                  LiDAR-priority voxel fuse  ⇄  per-object snapshots
                            │
                            ▼
                  dense 3D scene + dynamic BEV video
```

Per instance: 2 anchors, 1 linear system, no iteration. See
[`docs/method.md`](docs/method.md) §3 for the geometry; the layered
pipeline (Layer 1–5) is documented in
[`docs/architecture.md`](docs/architecture.md).

## Architecture

```
src/lidar_anchored_depth/
├── configs/          @dataclass configs (tyro-friendly)
│   ├── base.py       SceneConfig / OutputConfig / RuntimeConfig
│   └── stages/       per-stage configs (complete, inject, render-bev, pipeline)
├── stages/           Stage classes, one per `lad <subcommand>`
│   ├── base.py       Stage ABC + StageArtifacts
│   ├── dense_completion.py
│   ├── dynamic_inject.py
│   └── bev_render.py
├── pipelines/        multi-stage orchestrators (`lad pipeline`)
├── engine/           OutputManager: <root>/<scene>/<stage>/<run>/ + latest symlink
├── cli/              tyro-based `lad` entry point
├── alignment/        HAD / AA-HAD geometry, baselines
├── data/             V2X loader, calibration, ego-id lookup
├── pipeline/         primitives (depth_completion, colorize, ground grid, object_snapshot)
├── reconstruction/   PLY I/O, voxel down, ICP, chamfer
├── segmentation/     SAM / SegFormer I/O
├── models/           DGCNN + CFM residual head
├── viz/              BEV renderer, debug overlays
├── eval/             metrics + per-split runner
└── analysis/         error-propagation derivations
```

### Dual labels source

The V2X dataset ships two bbox folders:

* `road_labels/interpolation_labels/<ts>.json` — 10 Hz interpolated;
  smooth for video but jitter at bbox edges leaks dynamic-object
  LiDAR into the "static" cloud.
* `road_labels/merged_pcd_all/<ts>.json` — 1 Hz hand-labeled; precise
  bbox cull, used for static accumulation.

`SceneConfig` defaults split the two:

| Stage / use                          | Default source            |
| ------------------------------------ | ------------------------- |
| Static cloud + ground grid           | `merged_pcd` (1 Hz)       |
| Per-frame refine (dyn LiDAR cull)    | `interpolation` (10 Hz)   |
| Per-object accumulation              | `interpolation` (10 Hz)   |
| Snapshot inject + BEV video          | `interpolation` (10 Hz)   |

Fall back to `--scene.static-labels-source interpolation` when a scene
has no hand-labels.

## Reproducing paper results

See [`docs/reproduce.md`](docs/reproduce.md).

## Citation

To be filled once the manuscript is finalized — see `CITATION.cff`.

## License

Inherits the parent DA3 repository license (Apache-2.0). Contributions
welcome via PR.
