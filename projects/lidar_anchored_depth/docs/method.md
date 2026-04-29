# Height-Anchored Depth (HAD)

> Status: working draft, v0.1
> Owner: project team
> Scope: methodology note for the LAD-DA3 paper. Companion: `docs/data.md`, `docs/reproduce.md`.

## 1. Motivation

Foundation depth models (DA3, Marigold, Metric3D, Depth-Anything-V2) deliver
remarkable **relative** depth in the wild, but their **metric** outputs
exhibit strong **scale drift** on out-of-distribution scenes — most notably
roadside V2X cameras mounted at 5–10 m height with 15–45° down-tilt.

The **deliverable** of this work is a dense, metric, colored point-cloud
**reconstruction of a roadside intersection** from 4 wide-baseline pinhole
cameras + 4 LiDARs + V2X 3D-detection annotations. The classical
roadside-reconstruction problem statement is well established and the
target market (HD maps, traffic monitoring, V2X-perception) is mature; we
adopt this framing as our paper's primary one. (One downstream
application — feeding dense control signals into a Roadside-to-Agent
video diffusion pipeline — is discussed only as motivation in the
application section.)

Wide-baseline roadside is an **adversarial** regime for existing dense
reconstruction stacks: 90°+ baseline angles between adjacent cameras
collapse SIFT/ORB feature matching, so COLMAP / MVS / 3D-Gaussian-Splatting
all fail to bootstrap. Naïvely fusing per-camera foundation depth fails
too, because each camera self-normalizes its relative depth and the four
clouds do not link up. We sidestep correspondence entirely by anchoring
each camera independently to the same world frame.

Existing test-time corrections for the metric-depth subproblem fall into
two camps:

1. **Global scale alignment.** Solve a single scalar `s` (or affine
   `(a, b)`) so that `s · d_pred ≈ d_lidar` over all overlapping pixels,
   typically via least-squares + RANSAC.
2. **Region-wise alignment.** Use SAM masks to fit per-region affines on
   `(d_pred, d_lidar)` pairs.

Both treat **camera-axis depth `z`** as the primitive and use **per-pixel
LiDAR `z` measurements** as anchors. We argue this is the *wrong* primitive
for roadside cameras and propose **height** (world-frame Z) as the right
one.

### 1.1 Test bed: THICV-R2A

We evaluate on the THICV-R2A dataset (Tsinghua intersection, 89 ~22-second
sessions over ~9 days, 12,891 roadside frames, 1.08 M annotated objects
across 20 classes). The rig is **one fixed intersection** with 4 pinhole
cameras (cam0 / 3 / 6 / 9), 4 LiDARs (merged in world frame), and 3D-bbox
annotations per frame. The full schema is in `docs/dataset_guide.md`.

Two properties of this setup matter for design:

- **Single-intersection × multi-session**: per-scene overfitting (NeRF /
  3DGS style) is *encouraged* — the static background is identical
  across all 89 sessions, so any learned scene-specific component (the
  ground field MLP, σ-uncertainty head) can pool training data across
  the full 0.49-hour corpus.
- **Static fixtures are pre-segmented**: 17 % of annotations
  (`Bollards`, `Crash_bucket`, `Cone`) are scene-static; we anchor them
  once per `(scene, instance_id)` and reuse the anchor across all
  sessions, instead of re-solving an affine per frame.

## 2. Why height, not depth, is the right primitive for roadside

### 2.1 The roadside geometry

Let camera be mounted at world height `h_c` (≈5–10 m) with down-tilt
`θ ∈ [15°, 45°]`. For a point on the ground at world horizontal distance
`X` from the camera base:

- `z` (camera-axis depth) ≈ `√(X² + h_c²) · cos(θ + atan(h_c/X) - π/2)`
- `Z` (world height of point) = `0` (exact, by definition of ground plane)
- `Z` (world height of object top) = physical height of object (1.5 m car,
  1.7 m person, etc.) — **independent of `X`**.

### 2.2 Sensitivity to single-pixel error

For a single-pixel image-row error `Δv`, the back-projected depth error
scales as:

```
Δz / z   ≈   z / (f_y · h_eff)
```

where `f_y` is focal length and `h_eff` is the *effective* baseline given
by camera tilt. At `z = 100 m`, `f_y = 1500 px`, `h_eff = 5 m`: a 1-px
error gives **Δz ≈ 1.3 m** at 100 m. RANSAC on `z`-anchors does not
help — the underlying signal is degenerate.

In contrast, for **world height `Z`**, the same single-pixel error gives:

```
ΔZ   ≈   z · Δv / f_y      (small-angle, near-vertical edges)
```

`Z` does *not* multiply `z`-error onto itself — it is conditioned by the
height of the object, which is a fixed physical quantity. **Height is
intrinsically better-conditioned at long range.**

A formal error-propagation derivation lives in
`src/lidar_anchored_depth/analysis/error_propagation.py`; we reproduce
its main result as Figure X of the paper.

### 2.3 LiDAR-side reliability

LiDAR ranging error is ≈ ±2 cm + 0.1 % range (Velodyne-class) and is
*independent* of camera-LiDAR calibration in the **world-Z direction** for
roadside rigs (LiDAR mounted near the camera, both pre-calibrated to a
common world frame). Per-point `Z` is therefore one of the most reliable
quantities available; per-point `z` (camera depth) is degraded by
calibration error proportional to range.

## 3. Method: Height-Anchored Depth (HAD)

### 3.0 Empirical justification of the linear assumption

HAD assumes the foundation depth model's relative output ``d̃`` is linear
in metric depth: ``z = a · d̃ + b``. We tested this on the THICV-R2A
test bed (8 frames, scene 008, 2 timestamps × 4 pinhole cameras) by
fitting three candidate models on all (d̃, z_lidar) pairs and comparing
RMSE in z-space:

| Model | Median RMSE (m) | Worst RMSE (m) |
|---|---:|---:|
| linear ``z = a · d̃ + b`` | 14.7 | 16.9 |
| inverse ``1/z = a · d̃ + b`` | 1576 | 4448 |
| quadratic ``z = a · d̃² + b · d̃ + c`` | 14.6 | 16.9 |

Two findings drive the design:

1. **DA3 is depth-like, not disparity.** The inverse fit's RMSE is
   ≈100× the linear fit's. The 1/z fit reliably blows up near
   ``d̃ ≈ 5–6`` (where the fitted ``a · d̃ + b`` crosses zero) — clear
   evidence that the inverse model is mis-specified.
2. **Quadratic offers no real improvement.** Across all 8 frames
   the margin between quadratic and linear is < 1 % (median 0.13 %).
   The extra parameter is statistical noise; the linear form survives.

So ``z = a · d̃ + b`` is empirically validated as the right form for our
deployment. (No g_φ calibration MLP is needed — see ``docs/method.md``
§9 for what would have triggered one.)

The 11–17 m global-affine RMSE per camera is *much larger* than what we
expect per-instance: each SAM mask covers a narrow z range (1–2 m for a
ground vehicle), so a *local* affine is well-conditioned. Closing this
gap — global ≈ 15 m → per-instance ≈ 1 m — is the central claim of the
HAD ablation table.

The diagnostic walkthrough is in ``docs/da3_diagnostic.md``; raw scatter
plots and JSON summaries live under ``preview/diag/``.

### 3.1 Inputs

For a single roadside frame:

- `I ∈ ℝ^{H×W×3}` — RGB image
- `K ∈ ℝ^{3×3}` — camera intrinsics
- `T_wc ∈ SE(3)` — camera-to-world rigid transform (so `T_cw = T_wc^{-1}`
  takes world points to camera frame)
- `P_lidar ∈ ℝ^{N×3}` — LiDAR points in world frame
- `M = {m_i}_{i=1..K}` — SAM masks, each `m_i ∈ {0,1}^{H×W}` for one
  instance
- `d̃ ∈ ℝ^{H×W}_+` — DA3 relative depth prediction (up to global affine)

### 3.2 Per-instance height extraction

For each mask `m_i`:

1. **Project LiDAR into the image:**
   `(u, v, z_cam) = π(K · T_cw · [P_lidar; 1])` for the points whose
   projection lies inside `m_i` and whose `z_cam > 0`.
2. **Lift the LiDAR points back to world Z:** `Z_p = (T_wc · [P_lidar; 1])[2]`.
   (Trivial — `Z_p` is the third coordinate of the LiDAR point in world frame.)
3. **Robustly estimate the instance height interval `[Z_i^min, Z_i^max]`.**
   Use a 5/95-percentile (or RANSAC over a 1D Gaussian fit) to reject
   spill-over points (mask boundary leaking onto background). Output the
   pair `(Z_i^min, Z_i^max)`.

### 3.3 Per-instance depth solve

For a mask, identify two anchor pixels:

- **Top pixel** `(u_t, v_t)` — the top row inside `m_i`, x-coord chosen
  along the mask centroid column.
- **Bottom pixel** `(u_b, v_b)` — the bottom row of `m_i`.

For each anchor pixel `(u, v)`, the back-projection ray in world frame is
parameterized by depth `λ`:

```
P_world(λ) = T_wc · ( λ · K^{-1} · [u, v, 1]^T )
```

Setting `P_world(λ).Z = Z_target` gives a **single linear equation in
λ** (closed-form, no iteration):

```
λ_top = (Z_i^max − t_z) / (R_3,: · K^{-1} · [u_t, v_t, 1]^T)
λ_bot = (Z_i^min − t_z) / (R_3,: · K^{-1} · [u_b, v_b, 1]^T)
```

where `T_wc = [R | t]`, `R_3,:` is the third row of `R`, and `t_z` is
the camera world height. We obtain two **metric depth anchors**
`(z_top, z_bot)` for the instance, with `z = ‖ λ · K^{-1} · [u,v,1] ‖`.

### 3.4 Per-instance affine fit on relative depth

Within each mask `m_i`, fit a 1-parameter affine on the DA3 relative
depth using the two anchors:

```
z_pred(u, v) = a_i · d̃(u, v) + b_i,        (u, v) ∈ m_i
```

where `(a_i, b_i)` is solved exactly by the linear system

```
[ d̃(u_t, v_t)   1 ] [a_i]   [z_top]
[ d̃(u_b, v_b)   1 ] [b_i] = [z_bot]
```

For instances with only a single reliable anchor (e.g., ground-occluded
bottom or framing-cropped top), fall back to a 1-parameter scale fit
through that anchor, plus a neighbor-propagated `b_i` from spatially
adjacent instances.

### 3.5 Ground plane

Ground pixels (background to all instance masks, or labeled by SAM as
"road" if available) are handled by a separate branch: fit a world-frame
ground plane `n · X + d = 0` from the LiDAR ground returns via RANSAC,
then for each pixel back-project the ray to its intersection with the
plane → metric `z` directly. This is the same procedure used by prior
roadside-reconstruction work and is treated as a strong baseline branch.

### 3.6 Fusion

The full output depth map is assembled as:

- For each instance mask: `z_pred = a_i · d̃ + b_i`
- For ground: `z_ground` from plane intersection
- Smooth seam at mask boundaries via guided filter or Poisson blending
  on `1/z` (TBD; falls under refinement)

## 4. Annotation-Aware HAD (AA-HAD) — handling LiDAR-camera time offset

### 4.1 The problem

A roadside camera and roadside LiDAR are separate sensors with separate
clocks and (often) separate sampling rates. For a fast-moving target,
even a 50 ms offset induces a ≈ 1.5 m horizontal world displacement —
enough to break the *assignment* between a SAM image-mask and the LiDAR
points that nominally lie inside it. Per-pixel `z`-anchor methods are
particularly vulnerable: a wrong assignment scrambles the depth
distribution of the very anchors used to fit `(a, b)`.

HAD is partially immune by construction: world-frame **height `Z` does
not move under small horizontal displacements** (`v_z ≈ 0` for ground
vehicles within the time scale of interest). The remaining vulnerability
is the bbox ↔ mask **assignment** itself.

### 4.2 The free lunch from V2X annotations

In an actual roadside V2X deployment, the rig is already running 3D
object detection + tracking on the LiDAR stream. The output is exactly
what we need:

```jsonc
{
  "timestamp": "1743645353317",
  "object": [
    {
      "id": 1, "label": "Car",
      "x": -61.96, "y": 14.01, "z": -1.46,        // bbox center, world frame, meters
      "length": 4.45, "width": 2.43, "height": 1.97,
      "yaw": -1.69,
      "vx": 0.0, "vy": 0.0,                         // world-frame velocity, m/s
      "occlusion": 0,                               // {0, 1, 2}
      "num_points": 1472                            // points inside the bbox
    },
    ...
  ]
}
```

Two consequences:

1. The bbox `height` field gives the **exact** world-frame height of
   the object — no need to estimate `[Z_min, Z_max]` from the points
   inside a SAM mask, no percentile / RANSAC rejection. From the bbox:
   `Z_min = z - h/2`, `Z_max = z + h/2`. This is HAD's strongest input.
2. `(vx, vy)` lets us compensate the bbox in world frame for the
   LiDAR-camera time offset *without* tracking, *without* clustering,
   *without* any LiDAR semantic segmentation.

### 4.3 AA-HAD algorithm

For a per-timestamp annotation entry with cameras `{c}`, LiDAR `P_world`,
and dynamic objects `{o_j}`:

```text
Δt  = camera_ts - lidar_ts                 (often 0; configurable per dataset)

# Dynamic / annotated branch
for each object o_j:
    if occlusion >= 2: skip
    if num_points < N_min: skip
    cx, cy = o_j.x + o_j.vx · Δt, o_j.y + o_j.vy · Δt    # XY motion-compensated
    cz, h  = o_j.z, o_j.height                            # Z untouched (v_z ≈ 0)
    Z_min, Z_max = cz - h/2, cz + h/2                     # FROM BBOX, not points
    bbox_world(t_C) = box(cx, cy, cz, l, w, h, yaw)

    for each camera c that potentially sees o_j:
        bbox_uv = project_3d_bbox(bbox_world(t_C), K_c, T_cw_c)
        mask_i  = argmax_i  IoU(bbox_uv, sam_mask_i^c)
        if IoU < τ_iou: skip                              # assignment failed

        # HAD: solve (a_i, b_i) so that v_top→Z_max and v_bot→Z_min
        a_i, b_i = solve_per_instance_affine(d̃_c, K_c, T_wc_c,
                                              mask_i, Z_min, Z_max)
        z_pred_c[mask_i] = a_i · d̃_c[mask_i] + b_i

# Static / background branch
P_static = P_world  \  ⋃_j  bbox_interior(o_j)            # remove annotated objects
# Optionally accumulate P_static across a temporal window (rig is fixed)
# Run HAD-on-mask or ground-plane fit on regions not covered above.
```

`τ_iou`, `N_min`, the temporal window for static accumulation, and the
choice of "covered region" → ground-vs-instance fall-through are all
exposed as method config knobs.

### 4.4 Why C2 is a clean second contribution

The community has *separately* invested in: (a) roadside V2X 3D
detection / tracking (mature, multiple SOTA models, real deployments),
and (b) monocular metric depth (foundation models, the Marigold / DA-V2 /
Metric3D line). Nobody — to our knowledge — has connected the two ends:
**the V2X infrastructure already produces precisely the supervision a
roadside metric-depth recovery needs.** AA-HAD is the link. It costs no
new training, no new annotation, and it generalizes to any roadside
deployment that produces 3D detection logs.

### 4.5 Failure modes & honest limits

- **Mask ambiguity**: two overlapping objects (truck behind car) project
  to overlapping bbox-uv → IoU split. Mitigation: use SAM's
  `predicted_iou` per mask plus `num_points` of the bbox; choose the
  mask whose centroid lies closest to the bbox center.
- **Cropped bbox at image border**: the height interval is no longer
  bracketed by `v_top, v_bot` from a complete mask. Mitigation:
  fall back to a single anchor (whichever extreme is in-image) plus
  the global scale prior.
- **Wrong `vx, vy` from the V2X tracker**: rare for vehicles after a
  few frames; can flag via `num_points` and `track_age` (if available).
- **Static fixtures (Bollards, Crash_bucket, Cone)**: these classes are
  annotated every frame but never move. The loader detects them by
  class membership and lifts them out of the per-frame dynamic stream
  into a scene-static `(scene_id, instance_id)` table; the affine
  anchor is solved once per fixture and reused across sessions, both
  for efficiency and to avoid frame-by-frame numerical noise.

## 5. Methods (ours), revised

| ID | Name | Description |
|----|------|-------------|
| M1 | HAD-mask | Per-instance height extracted from LiDAR-in-mask returns (§3) |
| M2 | HAD-bbox | Per-instance height taken directly from V2X bbox annotation (§4) — exact |
| M3 | AA-HAD | M2 dynamic branch + ground-plane background branch + temporal accumulation for static |
| M4 | AA-HAD + Adaptive | M3 with density-aware fusion at seams |

`M3` is the headline; `M1` is reported for ablation against
annotation-free deployment.

## 6. Baselines

| ID | Name | Description |
|----|------|-------------|
| B0 | DA3-raw | Median scaling, no LiDAR |
| B1 | Median-z | Median ratio of `z_lidar / z_pred` over all pairs |
| B2 | Global-LSQ | Single-scale closed-form LSQ |
| B3 | Global-RANSAC | LSQ with RANSAC outlier rejection |
| B4 | Region-z-affine | Per-mask `(a, b)` fit on `(d̃, z_lidar)` pairs (prior work) |
| B5 | Ground-only | Plane-back-project for ground pixels, fall back to B3 elsewhere |
| B6 | Bbox-z-anchor | Use V2X bbox center `z` as a single per-object anchor (no height); ablates "bbox without height-as-primitive" |

## 7. Evaluation protocol

Metrics (per-pixel, on pseudo-GT depth from accumulated LiDAR sweeps):

- AbsRel, SqRel, RMSE, RMSE-log
- δ < 1.25, δ < 1.25², δ < 1.25³
- Stratified by range: 0–20 m, 20–50 m, 50–100 m, > 100 m
- Stratified by class (vehicle / pedestrian / structure / ground), via SAM
  + label propagation

Point-cloud metrics (lifted depth → world points):

- Chamfer distance to LiDAR point cloud
- Voxel occupancy IoU at 0.5 m grid

Splits: scene-level train/val/test (no frame leakage). Fixed seed.

## 8. Ablations

- **Anchor primitive**: height vs. depth (HAD-mask M1 vs. B4) — the central
  scientific question
- **Annotation source**: bbox (M2) vs. mask-derived height (M1) — measures
  the gain from the V2X annotation free lunch
- **Time compensation**: AA-HAD with `(vx, vy)` correction vs. without —
  isolates the contribution of explicit motion correction
- **Ground branch**: with vs. without (M3 vs. M2)
- **LiDAR sparsity**: 100 % / 50 % / 25 % / 10 % retained points
- **SAM quality**: Ground-truth masks vs. SAM-base vs. SAM-Huge vs. no
  segmentation (degenerates to global)
- **Per-object anchors count**: 1 (top only) / 2 (top+bot) / N (all
  reliable returns within mask treated as anchors via robust fit)
- **Range stratification**: gain attributable to height-anchor specifically
  at > 50 m
- **Annotation degradation**: drop occluded objects vs. keep, vary
  `num_points` threshold, simulate noisy bbox (perturb `(x, y, yaw)`)

## 9. Open questions / future work

- **Differentiable HAD** — turning §3.4 into a loss for fine-tuning DA3
  itself on roadside data.
- **Multi-frame HAD** — accumulating instance height estimates over a
  short window (instance tracking → tighter `Z` interval).
- **No-LiDAR setting** — using class-specific height priors when LiDAR
  is unavailable but SAM is, with a learned uncertainty.

## References (placeholders)

- Fischler & Bolles, RANSAC, 1981
- Depth Anything 3 (DA3), 2025
- Segment Anything (SAM), 2023
- DAIR-V2X, 2022
- KITTI Depth Completion benchmark, 2017
- Metric3D / ZoeDepth / Marigold (foundation metric depth)
