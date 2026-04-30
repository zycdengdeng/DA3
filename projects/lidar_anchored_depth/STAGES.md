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

## Stage 1.2 — Dataset alignment *(this commit)*

- `docs/dataset_guide.md` — verbatim copy of the user's THICV-R2A guide
- `data.calibration` — parse `support_info/calib.json` into typed
  `SceneCalibration`, including the world frame (= VirtualLidar),
  per-camera intrinsics + distortion + `T_wc`, per-LiDAR `T_wL`, and
  ZYX-Euler helper
- `data.carid_lookup` — parse `support_info/carid.json` for ego-id
  per-scene
- `data.roadside_v2x` rewritten with the real format:
  - `pinhole{N}/` ↔ `cam{3,6,9,0}_*.png` mapping pinned at module level
  - ZYX Euler annotation parsing (not Rodrigues)
  - `(scene_id, instance_id)` composite key for cross-scene uniqueness
  - `STATIC_FIXTURE_CLASSES = {Bollards, Crash_bucket, Cone}` lifted
    out of the per-frame dynamic stream into a scene-static table
  - 500 ms PCD-timestamp tolerance constant
- `scripts/sanity_check_dataset.py` — runs the §11 verification list
  (data root, scene layout, annotations, calibration, pinhole mapping,
  carid) once on a fresh server mount
- `method.md` updated: framing is "roadside reconstruction" (mainline);
  R2A control-signal feeding is application section only; THICV-R2A
  test-bed and per-scene-overfit rationale documented

Stage 1.2 ships only public APIs and dataclasses — actual file I/O
(PCD reading, image decoding, full scene indexing) waits for Stage 3
once the data is mounted on the build host.

## Stage 3C — Per-object temporal accumulation (this commit)

The per-frame headline (Stage 3A/B) gave us a per-object RMSE number.
Stage 3C complements that with a **multi-frame, multi-camera per-object
reconstruction**: we accumulate every LiDAR return that landed inside
one V2X bbox across every frame the object appears in (object-local
frame), do the same for AA-HAD's per-pixel unprojection through SAM
masks, voxel-downsample, and report the symmetric Chamfer distance
between the two clouds. This is the project's headline qualitative
**and** quantitative figure: a dense per-object reconstruction with a
single number that says "metric depth error vs LiDAR ground truth".

Modules added under `src/lidar_anchored_depth/reconstruction/`:
- `io` — tiny in-house PLY writer / reader (XYZ + optional RGB),
  ASCII or binary little-endian. No open3d.
- `chamfer` — symmetric Chamfer distance via `scipy.spatial.cKDTree`,
  returns scalar + per-direction stats (mean / median / p95).
- `object_local` — `world_to_object_local` / `object_local_to_world`
  using the V2X bbox pose, plus `voxel_downsample` (centroid + colors).
- `unproject` — `depth_to_world_points`: backproject a (H, W) metric
  depth + RGB image through the camera ray to a colored world cloud,
  optionally restricted by a SAM mask.

Scripts:
- `scripts/run_object_accumulation.py` — main Stage 3C entry point.
  `--bbox-id N` for one object or `--all-bboxes` for a sweep. Writes
  `<scene>_obj<id>_lidar.ply` (GT) and `<scene>_obj<id>_aa_had.ply`
  (predicted, with RGB) plus a JSON summary with per-object Chamfer.
- `scripts/aggregate_object_chamfer.py` — population-level summary
  across many JSONs (median chamfer, per-class breakdown, chamfer vs
  number of frames).

22 new tests in `tests/test_reconstruction.py`. 146 total tests pass.

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

- `data.roadside_v2x` — implement file I/O on the THICV-R2A mount:
  - `_build_index` walks `/mnt/car_road_data_TianJin/` to enumerate
    scenes, label JSONs, pinhole image paths, and merged-PCD paths
  - `get_frame` decodes one (scene, ts, cam) into a populated `Frame`
  - static-fixture pass on first encounter per scene
  - PCD I/O via `open3d` (or `pypcd`); image I/O via `cv2`
- `data.pseudo_gt` — multi-session LiDAR accumulation for dense GT
  (the rig is fixed across all 89 sessions)
- `pipeline.roadside.RoadsidePipeline` — wires DA3 + SAM + AA-HAD +
  ground-field + multi-camera fusion; AA-HAD is the default method
- `cli.infer`, `cli.evaluate`, `cli.ablate`
- Integration test: B0–B6 + M1–M4 all run on a synthetic frame
  end-to-end; THICV-R2A smoke run gated on `--data-root` being present

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
