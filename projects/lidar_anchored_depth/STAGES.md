# Rollout plan

The project is built incrementally. Each stage is one (or a small number of)
commits. Reviewers can stop after any stage and have something coherent.

## Stage 1 — Skeleton & methodology *(this commit)*

- Project layout under `projects/lidar_anchored_depth/`
- `pyproject.toml` with editable install, dev / sam / viz extras
- Method note `docs/method.md` — full HAD geometry derivation, baselines,
  ablations
- Data card `docs/data.md` — Frame schema, calibration conventions,
  dataset-adapter contract
- Reproducibility doc `docs/reproduce.md`
- `Frame` and `BaseDataset` (real, with validation + tests)
- Stub modules for `alignment`, `segmentation`, `pipeline`, `eval`,
  `analysis`, `viz`, `cli` — all importable, with their public API
  pinned by signatures and dataclasses
- Real depth metrics + their unit tests
- Hydra config skeleton (`default`, `data/*`, `model/*`, `method/*`,
  `experiment/*`)
- pytest configured; smoke tests pass on the real bits

## Stage 2 — Core algorithms

- `alignment.projection` — port from prior `utils/lidar_alignment.py`
  (NumPy + Torch differentiable variants), plus `project_3d_bbox`
- `alignment.global_scale` — B1, B2, B3 (median, LSQ, RANSAC)
- `alignment.ground_plane` — RANSAC plane fit + per-pixel ground depth
- `alignment.region_affine` — B4 (per-mask affine on z)
- `alignment.height_anchor` — **HAD-mask** (M1, mask-derived heights)
- `alignment.bbox_anchor` — **HAD-bbox / AA-HAD** (M2, M3) — V2X bbox
  height + (vx, vy) motion compensation; bbox-uv-to-mask IoU matching
- `alignment.adaptive` — fusion (M4 component)
- Unit tests against synthetic ground truth from `conftest.py` (extended
  with synthetic dynamic-object scenarios)
- Type-correct, mypy-clean

## Stage 3 — Data adapters & end-to-end pipeline

- `data.roadside_v2x` — implement parsing for the V2X annotation format
  (`docs/data.md` §3): per-timestamp JSON, scene calibration, multi-camera
  emission, occlusion / `num_points` filtering
- `data.car_road` — port from prior branch, slim
- `data.self_data` — adapter for the lidar-calibration snapshot
- `data.generic` — for arbitrary roadside data; manifest-driven
- `data.pseudo_gt` — multi-sweep accumulation for evaluation GT
- `pipeline.roadside.RoadsidePipeline` — wires it all together; AA-HAD
  is the default method
- `cli.infer`, `cli.evaluate`, `cli.ablate`
- Integration test: B0–B6 + M1–M4 all run on synthetic frame end-to-end

## Stage 4 — Analysis, ablations, paper figures

- `analysis.error_propagation` — closed-form curves + sensitivity figure
- Ablation runners: anchor primitive, LiDAR sparsity, SAM quality, range
- `cli.reproduce_paper` — single command per table/figure
- Notebooks for exploration & qualitative results
- README results tables filled

## Stage 5 — Release polish

- CI (lint + tests on PR)
- README badges, demo gif
- CITATION.cff with real authors
- Tagged release, archived artefact
