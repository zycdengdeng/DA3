# Architecture

LAD-DA3 is organised as a multi-stage CLI (`lad`) where every stage
reads its input from disk (or by convention from the previous stage's
`latest` symlink) and writes typed artefacts into its own output
directory. There is no shared in-memory state between stages, so any
stage can be re-run in isolation.

## Pipeline overview

```
       External inputs                                Layered processing
       ──────────────                                 ──────────────────
   ┌───────────────────┐
   │  RGB images (cam) │ ──► run_da3_inference.py ──►  DA3 depth (.npz)
   │  /preview/da3/    │
   └───────────────────┘
   ┌───────────────────┐
   │  RGB images       │ ──► run_sam_inference.py  ──►  SAM dyn masks (.npz)
   └───────────────────┘
   ┌───────────────────┐
   │  RGB images       │ ──► run_segformer_inf.py  ──►  SegFormer class (.npz)
   └───────────────────┘
   ┌───────────────────┐
   │  Hand-labeled     │
   │  bboxes (1 Hz)    │ ◄── road_labels/merged_pcd_all/<ts>.json
   ├───────────────────┤
   │  Interpolated     │
   │  bboxes (10 Hz)   │ ◄── road_labels/interpolation_labels/<ts>.json
   └───────────────────┘
   ┌───────────────────┐
   │  Per-object PLYs  │ ──► run_object_accumulation.py
   │  /preview/recon_  │     (per-object accumulated dense cloud)
   │   3I_full/        │
   └───────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  lad complete                                                │
   │  Stage 2: per-(scene, cam, ts) Layer 1+3 prior → CFM refine  │
   │  Stage 4: LiDAR-priority voxel fuse (ground-band skipped)    │
   │  ─────────────────────────────────────────────────────────── │
   │  Outputs:                                                    │
   │    refined.ply  baseline.ply  hybrid.ply                     │
   └──────────────────────────────────────────────────────────────┘
                   │
                   ▼  (auto-discover via outputs/<scene>/complete/latest/)
   ┌──────────────────────────────────────────────────────────────┐
   │  lad inject       │   lad render-bev                         │
   │  ───────────────  │   ─────────────────────────────────────  │
   │  Stage 5 at one   │   Stage 5 per ts (linear pose interp +   │
   │  anchor ts. Out:  │   gap detection). Top-down z-buffer.     │
   │  hybrid_with_     │   Out: bev_frames/bev_*.png              │
   │   dyn.ply         │   →  ffmpeg one-liner printed.           │
   └──────────────────────────────────────────────────────────────┘
```

## Layered method

Numbering matches the docstrings inside the codebase.

| Layer | Where                                          | Purpose                                                                                |
| ----- | ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1     | `alignment/`                                   | DA3 → camera-axis depth via SAM-instance height anchors (HAD).                         |
| 2     | `alignment/aa_had.py`                          | AA-HAD per-camera linear calibration ``(a, b)``.                                       |
| 3     | `pipeline/ground_height_grid.py`               | LiDAR-derived 2.5D ground height grid + snap.                                          |
| 4     | `pipeline/lidar_completion.py`                 | LiDAR-priority voxel fuse: cm-precise LiDAR wins per voxel, refined fills the rest.    |
| 5     | `pipeline/object_snapshot.py`                  | Per-object accumulated dense reconstructions, placed at the anchor / video ts.         |
| A1    | `segmentation/` + `models/point_residual.*`    | SegFormer per-pixel class as a segment feature for the residual flow head.             |

## Output convention

```
outputs/<scene>/
├── complete/
│   ├── 2026-05-11_14-30-00/
│   │   ├── refined.ply
│   │   ├── baseline.ply
│   │   ├── hybrid.ply
│   │   ├── config.json     ← exact config that produced these artefacts
│   │   └── summary.json    ← voxel + frame counts, wall time
│   └── latest -> 2026-05-11_14-30-00/
├── inject/   ...same layout...
└── render-bev/ ...
```

* `OutputManager` (in `engine/runner.py`) provisions each run dir,
  dumps `config.json`, and refreshes the `latest` symlink on success.
* Downstream stages call `OutputManager.resolve_latest(root, scene,
  stage)` to find their input — falling back to lexicographic sort of
  subdirs on filesystems without symlink support.

## Configs

Every config is a frozen `@dataclass` consumed by `tyro` at the CLI
boundary. The shared blocks live in `configs/base.py`:

* `SceneConfig` — data root, scene id, camera ids, plus the dual
  labels-source switch (see below).
* `OutputConfig` — output root, run id, latest-symlink behaviour.
* `RuntimeConfig` — `gpu_ids` for sharding + `seed`.

Per-stage configs (`configs/stages/{complete,inject,render_bev,pipeline}.py`)
compose these three and add stage-specific knobs.

## Multi-GPU sharding

`lad complete` shards the per-(cam, ts) refine loop across GPUs:

```bash
lad complete --runtime.gpu-ids 0 1 2 3 ...
```

The parent never imports torch; each worker sets
`CUDA_VISIBLE_DEVICES` *before* its first torch import so that `cuda:0`
inside the child resolves to its assigned card. Per-frame work units
get a deterministic seed derived from `(scene, cam, ts)`, so the
refined PLY is bit-for-bit identical across worker counts.

`inject` and `render-bev` are CPU-only stages — they ignore the GPU
list.

## Dual labels source

The V2X dataset ships two bbox folders per scene:

* `road_labels/interpolation_labels/<ts>.json` — 10 Hz interpolated.
* `road_labels/merged_pcd_all/<ts>.json` — 1 Hz hand-labeled.

`SceneConfig.static_labels_source` (default `"merged_pcd"`) drives the
static-cloud accumulator: hand-labeled 1 Hz bboxes give a tight cull,
eliminating the "trail of car points along the moving trajectory"
artefact caused by interpolation jitter.

`SceneConfig.dynamic_labels_source` (default `"interpolation"`) drives
the per-frame refine and the dynamic snapshot inject — smoothness
matters more there than 0.1 m bbox precision.

When a scene has no hand-labels, pass `--scene.static-labels-source
interpolation` on the CLI.

## Determinism

* Per-(scene, cam, ts) torch + numpy seeds in `refine_one_frame` make
  refined artefacts reproducible across worker counts.
* `OutputManager` writes `config.json` so any run can be replayed by
  reading back the dataclass.

## Where to look first

| If you want to…                            | Read                                          |
| ------------------------------------------ | --------------------------------------------- |
| Understand the method                      | `docs/method.md`                              |
| See how a stage runs                       | `src/.../stages/dense_completion.py`          |
| Trace per-frame work                       | `src/.../pipeline/depth_completion.py`        |
| See how multi-GPU sharding works           | same file, ``run_refine_parallel`` + workers  |
| Add a new stage                            | implement `Stage` ABC + register in `cli/__init__.py` |
