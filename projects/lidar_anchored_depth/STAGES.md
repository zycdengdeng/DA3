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
  (NumPy + Torch differentiable variants)
- `alignment.global_scale` — B1, B2, B3 (median, LSQ, RANSAC)
- `alignment.ground_plane` — RANSAC plane fit + per-pixel ground depth
- `alignment.region_affine` — B4 (per-mask affine on z)
- `alignment.height_anchor` — **HAD** (M1)
- `alignment.adaptive` — fusion (M3 component)
- Unit tests against synthetic ground truth from `conftest.py`
- Type-correct, mypy-clean

## Stage 3 — Data adapters & end-to-end pipeline

- `data.car_road` — port from prior branch, slim
- `data.self_data` — adapter for the lidar-calibration snapshot
- `data.generic` — for the user's external roadside data; manifest-driven
- `data.pseudo_gt` — multi-sweep accumulation for evaluation GT
- `pipeline.roadside.RoadsidePipeline` — wires it all together
- `cli.infer`, `cli.evaluate`, `cli.ablate`
- Integration test: B0–B5 + M1–M3 all run on synthetic frame end-to-end

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
