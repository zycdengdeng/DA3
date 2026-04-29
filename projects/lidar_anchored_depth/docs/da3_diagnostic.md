# DA3 Diagnostic — picking the right depth model for HAD

> **Question this answers**: Does HAD's affine assumption
> ``z = a · d̃ + b`` hold for our DA3 outputs on roadside data, or do we
> need to switch to ``1/z = a · d̃ + b`` (disparity space) or a higher-order
> ``z = a · d̃² + b · d̃ + c``? The answer determines the form of the
> closed-form solver and ultimately the per-instance HAD recovery
> accuracy.

This doc walks through running the diagnostic on the THICV-R2A test bed.

## 0. Prereqs

Stage 2C requires DA3 + torch + matplotlib in the project's conda env:

```bash
conda activate lad
cd /mnt/zyc_wzh/DA3_lad

pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install -e .                                 # parent DA3 (editable)
pip install matplotlib

# HuggingFace mirror for the model weights download
export HF_ENDPOINT=https://hf-mirror.com
```

## 1. Run DA3 on a frame

`scripts/run_da3_inference.py` loads one (scene, ts, cam) frame via the
THICV-R2A loader and saves the relative depth as ``*_d.npz``:

```bash
cd projects/lidar_anchored_depth

PYTHONPATH=src python scripts/run_da3_inference.py \
    --data-root /mnt/car_road_data_TianJin \
    --scene 008 \
    --cam all \
    --output preview/da3/
```

Expected stdout (4 cameras, one timestamp):
```
[load] depth-anything/DA3Mono-Large
  ↳ loaded in 12.3s on cuda
[scene] 008_car0325_road0327_t8  ts=174287...  cams=['0','3','6','9']

  cam0  ->  008_..._cam0_d.npz    d̃ range [0.123, 4.56]  median 1.89   DA3 inference 410 ms
  cam3  ->  ...
  cam6  ->  ...
  cam9  ->  ...

done.
```

Sizes: each ``.npz`` is ~5 MB (1080×1920 float32 + conf).

> **Picking a model**: the default is ``DA3Mono-Large`` which the DA3
> README describes as "directly predicts depth, resulting in superior
> geometric accuracy" — best fit for HAD's linearity assumption. Try
> ``DA3-Giant`` for the strongest backbone, or
> ``DA3NESTED-GIANT-LARGE-1.1`` for a metric-tuned head.

## 2. Run the diagnostic

```bash
PYTHONPATH=src python scripts/depth_model_diagnostic.py \
    --data-root /mnt/car_road_data_TianJin \
    --scene 008 --timestamp 1742879642908 --cam 3 \
    --d-path 'preview/da3/008_*_ts1742879642908_cam3_d.npz' \
    --output preview/diag/
```

Stdout looks like:
```
[diag] 008_car0325_road0327_t8  ts=174287...  cam3  model=depth-anything/DA3Mono-Large
  N pairs (after filter): 24871
  d̃ range  : [0.412, 5.184]   median 1.876
  z range  : [3.214, 187.342]  median 38.412

  range          n      linear     inverse   quadratic
  [0-20]      8123       0.42        0.31       0.32
  [20-50]    11234       1.85        1.91       1.04
  [50-100]    4801       6.21        4.74       3.18
  [100-200]    713      14.92       11.85       7.96

  overall RMSE (m):
    linear      4.187    z = a · d̃ + b
    inverse     3.518    1/z = a · d̃ + b
    quadratic   2.142    z = a · d̃² + b · d̃ + c

  VERDICT: quadratic (z = a · d̃² + b · d̃ + c) — RMSE=2.142 m,
           margin over runner-up = 39.0% (high confidence)

  ↳ figure: preview/diag/008_..._cam3_diag.png
  ↳ json  : preview/diag/008_..._cam3_diag.json
```

The figure has two panels:
- **Left**: scatter (d̃, z) with all three fitted curves overlaid.
- **Right**: residuals (z_pred − z) vs true z, per model.

The JSON next to the PNG is the canonical quantitative artifact: it
goes into the paper's ablation table and into ``method.md`` updates.

## 3. Interpreting the verdict

Three outcomes determine the path forward:

### 3.1 ``linear`` wins (HAD's assumption holds)
Best case: HAD's closed-form solver as written is correct. Move on to
Stage 3 (end-to-end pipeline).

### 3.2 ``inverse`` wins (DA3 outputs disparity)
HAD's solver should be re-derived in 1/z space:

    1/z_world(u, v, z_cam) = 1 / (α(u,v) · z_cam + β)
    z_cam(u, v)            = a · d̃(u, v) + b
    ⇒ closed-form is no longer linear in (a, b)

Two practical fixes:
1. **Pre-invert**: pass ``1/z_lidar`` and ``d̃`` to the same linear
   solver (a small wrapper, no new math).
2. **Predict 1/z directly**: change the project's "metric depth"
   contract to "metric inverse-depth"; many downstream consumers
   prefer this anyway (constant disparity ≡ uniform precision).

### 3.3 ``quadratic`` (or worse) wins (DA3 has non-linear distortion)
The cleanest response is the **DA3 calibration head ``g_φ``** sketched in
``docs/method.md`` §4.2: a small per-camera-per-scene MLP that learns
``g_φ(d̃, u, v) → d̃'`` such that ``z = a · g_φ(d̃) + b`` fits at scale.
Stage 2C+ would add ``models/g_phi.py`` and an EM-style training loop.

If the margin between quadratic and linear is small (< 5 %), still
prefer linear — fewer parameters, closed-form solver, more interpretable.

## 4. Robustness — repeat across cameras and timestamps

The verdict on a single frame is noisy. Repeat over the 4 pinholes and
~10 timestamps, then aggregate the JSONs. Bash one-liner for an
overview:

```bash
ls preview/diag/*.json | xargs -I{} python -c "
import json, sys; d = json.load(open(sys.argv[1])); v = d['verdict']
print(f'{d[\"cam_id\"]} {d[\"ts_ms\"]} -> {v[\"best\"]:<10} RMSE {v[\"rmse_m\"]:5.2f} m  ({v[\"confidence\"]})')
" {}
```

If 90 %+ of frames vote the same model, that's the project's depth
model going forward.

## 5. Reporting

The diagnostic JSONs are the artifacts of record. Keep them under
``preview/diag/`` (gitignored) until the paper figures are stable;
copy the chosen scatter PNG into ``docs/figs/`` for the paper draft.
