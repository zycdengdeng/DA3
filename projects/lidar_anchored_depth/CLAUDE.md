# LAD-DA3 — project notes for Claude

## Target server (wzh@admin)

- CPU: 100+ threads (2500+ kernel threads visible in htop, ~200 user tasks)
- RAM: ~1 TB (24 G in use baseline)
- GPU: 8 × NVIDIA A100 80 GB
- Conda env: `lad` (Python 3.11)
- Repo path on server: `/mnt/zyc_wzh/DA3_lad/projects/lidar_anchored_depth`
- Dataset path on server: `/mnt/car_road_data_TianJin`

## Performance preferences

The user has ample CPU + GPU. Default to parallel execution for any
embarrassingly-parallel CPU work, **as long as it does not change the
numerical result**:

- per-object loops (accumulation, ICP refinement) → joblib / `multiprocessing.Pool`
- per-camera loops (static recon unproject, calibration) → thread/process pool
- per-frame data loading (DA3 + LiDAR + V2X) → `concurrent.futures.ThreadPoolExecutor`
  (loader is I/O-bound on PCD parse + V2X JSON, so threads suffice — no GIL contention)
- GPU work that can shard across cards (DA3 inference batch, SAM inference) →
  per-GPU process via `CUDA_VISIBLE_DEVICES`
- voxel-grid downsample / KDTree-based chamfer → already C-extension, leave
  single-threaded unless we see a wall-clock issue

Things to NOT parallelize:
- per-frame solver math that uses a shared RNG (B3-RANSAC seed must stay
  reproducible — split work, not the seed)
- training loops (single-GPU is fine; the head is tiny)
- anything where ordering matters (e.g., `find_frame_idx` index build)

When refactoring an existing script for parallelism, expose
`--workers N` (default `0` = serial, for debugging) and keep the serial
path as the fallback.

## Currently parallelisable scripts (not yet retrofitted)

- `scripts/run_object_accumulation.py` — per-object loop over objects
- `scripts/refine_aa_had_with_icp.py` — per-object ICP
- `scripts/run_static_scene_recon.py` — per-camera unproject + calibrate
- `scripts/run_da3_inference.py` — could shard across 8 GPUs
- `scripts/run_sam_inference.py` — same
- `scripts/train_sigma_head.py` — pair collection is per-(scene, ts, cam),
  fully parallelisable

Retrofit on demand; don't pre-emptively touch scripts the user is happy
with.
