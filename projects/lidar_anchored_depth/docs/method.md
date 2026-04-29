# Height-Anchored Depth (HAD)

> Status: working draft, v0.1
> Owner: project team
> Scope: methodology note for the LAD-DA3 paper. Companion: `docs/data.md`, `docs/reproduce.md`.

## 1. Motivation

Foundation depth models (DA3, Marigold, Metric3D, Depth-Anything-V2) deliver
remarkable **relative** depth in the wild, but their **metric** outputs
exhibit strong **scale drift** on out-of-distribution scenes — most notably
roadside V2X cameras mounted at 5–10 m height with 15–45° down-tilt.

Existing test-time corrections fall into two camps:

1. **Global scale alignment.** Solve a single scalar `s` (or affine
   `(a, b)`) so that `s · d_pred ≈ d_lidar` over all overlapping pixels,
   typically via least-squares + RANSAC.
2. **Region-wise alignment.** Use SAM masks to fit per-region affines on
   `(d_pred, d_lidar)` pairs.

Both treat **camera-axis depth `z`** as the primitive and use **per-pixel
LiDAR `z` measurements** as anchors. We argue this is the *wrong* primitive
for roadside cameras and propose **height** (world-frame Z) as the right
one.

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

## 4. Baselines

| ID | Name | Description |
|----|------|-------------|
| B0 | DA3-raw | Median scaling, no LiDAR |
| B1 | Median-z | Median ratio of `z_lidar / z_pred` over all pairs |
| B2 | Global-LSQ | Single-scale closed-form LSQ |
| B3 | Global-RANSAC | LSQ with RANSAC outlier rejection |
| B4 | Region-z-affine | Per-mask `(a, b)` fit on `(d̃, z_lidar)` pairs (prior work) |
| B5 | Ground-only | Plane-back-project for ground pixels, fall back to B3 elsewhere |

## 5. Methods (ours)

| ID | Name | Description |
|----|------|-------------|
| M1 | HAD | Per-instance height-anchored fit (§3.2–§3.4) on instances; B0 elsewhere |
| M2 | HAD + Ground | M1 plus ground-plane branch (§3.5) |
| M3 | HAD + Ground + Adaptive | M2 with density-aware fusion at seams |

## 6. Evaluation protocol

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

## 7. Ablations

- **Anchor primitive**: height vs. depth (M1 vs. B4)
- **Ground branch**: with vs. without (M2 vs. M1)
- **LiDAR sparsity**: 100 % / 50 % / 25 % / 10 % retained points
- **SAM quality**: Ground-truth masks vs. SAM-base vs. SAM-Huge vs. no segmentation (degenerates to global)
- **Number of anchors per instance**: 1 (top only) / 2 (top+bot) / N (all
  reliable LiDAR points within mask treated as anchors via robust fit)
- **Range stratification**: gain attributable to height-anchor specifically
  in the > 50 m range

## 8. Open questions / future work

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
