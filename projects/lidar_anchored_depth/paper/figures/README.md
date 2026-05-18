# Figures — capture commands

Every `\includegraphics` call in `main.tex` references a PNG in this
directory. Below is the per-figure recipe: which command produces the
underlying artefact, and what post-processing (cropping / overlay /
side-by-side) turns the raw output into a paper-ready figure.

Drop the final PNGs into this directory with the file names listed
under "**Save as**" — `main.tex` already imports them by that name.

All `cd …` commands assume the server cwd:

```bash
cd /mnt/zyc_wzh/DA3_lad/projects/lidar_anchored_depth
```

`lad` is the installed CLI (`pip install -e .`); `--scene.data-root`
is always `/mnt/car_road_data_TianJin` on this server.

---

## 1. `teaser.png`  — Fig. 1

**What it shows.** A top-down BEV of one intersection with the
\textsc{Lad-DA3} hybrid cloud + dynamic objects placed at one anchor ts.
Two zoom-ins as insets (one vehicle, one traffic-light pole).

**Capture.**

```bash
cd /mnt/zyc_wzh/DA3_lad/projects/lidar_anchored_depth

# Pick a representative ts (intersection busy, several vehicles visible)
lad render-bev \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --ts-stride 999999 \
    --image-size 2048 2048 \
    --pad-m 2.0 \
    --output.run-id teaser

# The BEV PNG you want is:
#   outputs/008/render-bev/teaser/bev_frames/bev_0000.png
```

**Post-processing.** Open the BEV PNG in any image editor. Crop two
~256×256 sub-regions around (a) a clearly-shaped vehicle and (b) a
traffic-light pole. Use a 2x2 grid layout (big BEV on the left taking
the full height; two zooms stacked on the right).

**Save as.** `paper/figures/teaser.png`

---

## 2. `pipeline.png`  — Fig. 2

**What it shows.** Block diagram of the 8 stages.

**Capture.** No command needed — this is a hand-drawn diagram. Sketch
in Excalidraw / draw.io / Keynote / Slides showing:

* 8 boxes left-to-right: **depth · mask · seg · calib · object-accum
  · complete · inject · render-bev**.
* Solid arrows: data flow between adjacent boxes.
* Dashed arrows up to a "outputs/&lt;scene&gt;/&lt;stage&gt;/latest"
  symlink overlay, with downstream boxes pulling from it.
* Labels under each box: input artefact type → output artefact type
  (e.g. depth: "RGB → `*_d.npz`").

**Save as.** `paper/figures/pipeline.png` (1500×500 px, two-column wide).

---

## 3. `008_baseline_refined_hybrid.png`  — Fig. 4

**What it shows.** Side-by-side 1×3 panel of the three PLYs the
`complete` stage writes on scene 008: `baseline.ply`, `refined.ply`,
`hybrid.ply` — all rendered from the same BEV viewpoint.

**Capture.** All three PLYs already exist after a successful
`lad complete --scene.scene 008` run:

```bash
ls outputs/008/complete/latest/{baseline,refined,hybrid}.ply
```

Render each via the BEV render helper. Easiest: open each in
CloudCompare or Open3D with a fixed top-down camera and screenshot,
OR use this small Python:

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from PIL import Image
from lidar_anchored_depth.reconstruction import read_ply_xyz_rgb
from lidar_anchored_depth.viz.bev import auto_bev_range, render_bev

OUT = "outputs/008/complete/latest"
for name in ("baseline", "refined", "hybrid"):
    xyz, rgb = read_ply_xyz_rgb(f"{OUT}/{name}.ply")
    xr, yr = auto_bev_range(xyz, pad_m=2.0)
    img = render_bev(xyz, rgb, x_range=xr, y_range=yr,
                     image_hw=(1536, 1536))
    Image.fromarray(img).save(f"paper/figures/008_{name}_bev.png")
PY
```

**Post-processing.** Image editor: concatenate the three PNGs
horizontally, add 3 text labels ("baseline", "refined", "hybrid")
under the panels.

**Save as.** `paper/figures/008_baseline_refined_hybrid.png`

---

## 4. `002_zeroshot.png`  — Fig. 5

**What it shows.** Two BEV panels stacked vertically: baseline (top)
and refined (bottom) of scene 002, where the residual head was
trained only on scene 008.

**Capture.** Same BEV render as Fig. 4 but on scene 002:

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from PIL import Image
from lidar_anchored_depth.reconstruction import read_ply_xyz_rgb
from lidar_anchored_depth.viz.bev import auto_bev_range, render_bev

OUT = "outputs/002/complete/latest"
for name in ("baseline", "refined"):
    xyz, rgb = read_ply_xyz_rgb(f"{OUT}/{name}.ply")
    xr, yr = auto_bev_range(xyz, pad_m=2.0)
    img = render_bev(xyz, rgb, x_range=xr, y_range=yr,
                     image_hw=(2048, 2048))
    Image.fromarray(img).save(f"paper/figures/002_{name}_bev.png")
PY
```

**Post-processing.** Stack the two PNGs vertically with labels.

**Save as.** `paper/figures/002_zeroshot.png`

---

## 5. `per_object.png`  — Fig. 6

**What it shows.** Three per-object accumulated PLYs (two cars + a
bus) overlaid in BEV at one anchor ts.

**Capture.** Per-object PLYs live in
`outputs/008/object-accum/latest/*_obj*.ply`. Pick 3 visually-distinct
ones (e.g. obj1 = car, obj7 = SUV, obj26 = bus) and render them
together with their bbox at the anchor pose:

```bash
PYTHONPATH=src python - <<'PY'
# Manual: pick 3 obj ids from outputs/008/object-accum/latest/ and
# project them via inject_object_snapshots at one ts. The simplest
# is to run `lad inject` and crop just the dynamic part:
PY

lad inject \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --anchor-ts-ms 1742879650704 \
    --mirror-axis y \
    --output.run-id per_object_fig
# Then BEV-render outputs/008/inject/per_object_fig/hybrid_with_dyn.ply
# (subtract the static cloud first if you want only the dyn highlights).
```

**Post-processing.** Crop the BEV to the central intersection, draw
3 dashed-arrow labels pointing at the 3 highlighted objects.

**Save as.** `paper/figures/per_object.png`

---

## 6. `bev_video_strip.png`  — Fig. 7

**What it shows.** 4 consecutive BEV frames (t, t+100ms, t+200ms,
t+300ms) showing one vehicle moving smoothly through the
intersection.

**Capture.** Run a full BEV render on scene 008, then pull 4 frames
that bracket a vehicle of interest:

```bash
lad render-bev \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --ts-stride 1 \
    --image-size 1024 1024 \
    --output.run-id video_strip
# Inspect outputs/008/render-bev/video_strip/bev_frames/
# Pick 4 frames that show the same vehicle traversing.
cp outputs/008/render-bev/video_strip/bev_frames/bev_0040.png \
   paper/figures/bev_t0.png
cp outputs/008/render-bev/video_strip/bev_frames/bev_0041.png \
   paper/figures/bev_t100.png
cp outputs/008/render-bev/video_strip/bev_frames/bev_0042.png \
   paper/figures/bev_t200.png
cp outputs/008/render-bev/video_strip/bev_frames/bev_0043.png \
   paper/figures/bev_t300.png
```

**Post-processing.** Horizontal 1×4 strip; circle the same vehicle in
each frame (small red dashed ellipse). Add timestamp labels under
each frame.

**Save as.** `paper/figures/bev_video_strip.png`

---

## 7. `ablation_skipground.png`  — Fig. 8

**What it shows.** BEV close-up of the road surface with and without
`--lidar-skip-ground`. Without: visible concentric rings around each
of the 4 LiDAR sensors. With: uniform.

**Capture.** Two `lad complete` runs differing only in this flag:

```bash
# With skip (default)
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --output.run-id ablation_skipground_on

# Without skip
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --no-lidar-skip-ground \
    --output.run-id ablation_skipground_off
```

BEV-render each `hybrid.ply` (same script as Fig. 4) and crop a
~10m × 10m road region in the centre.

**Post-processing.** Two panels side-by-side, labels
"without skip" / "with skip".

**Save as.** `paper/figures/ablation_skipground.png`

---

## 8. `ablation_labels.png`  — Fig. 9

**What it shows.** Static cloud built from interpolated 10Hz boxes
(left, showing trail artefacts) vs hand-labelled 1Hz boxes (right,
clean).

**Capture.** Two `lad complete` runs differing in
`--scene.static-labels-source`:

```bash
# 10Hz interpolation (old behaviour)
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --scene.static-labels-source interpolation \
    --output.run-id ablation_labels_interp

# 1Hz hand-labels (new default)
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --scene.static-labels-source merged_pcd \
    --output.run-id ablation_labels_handlabel
```

BEV-render each `hybrid.ply`. Zoom in on a region where a moving
vehicle (e.g. a green bus) was in the scene — the interpolation panel
should show the "trail of car-coloured pixels along the trajectory"
artefact.

**Post-processing.** Side-by-side; circle / arrow the bus trail in
the left panel; corresponding clean region on the right.

**Save as.** `paper/figures/ablation_labels.png`

---

## 9. `ablation_outlier_ref.png`  — Fig. 10

**What it shows.** BEV of the road surface near a tall structure
(traffic-light pole / wall) with and without
`outlier_reference_lidar`. Without: a visible "halo" of AA-HAD ground
points only near vertical structures (caused by the cull falsely
dropping ground AA-HAD because the nearest LiDAR is several metres
up). With: uniform road surface.

**Capture.** This branch is no longer exposed in the CLI (the
post-skip-ground KD-tree leak was patched in commit `71eef38`). To
reproduce the *before* picture:

```bash
# Temporarily revert the fix locally:
git show 71eef38^:src/lidar_anchored_depth/pipeline/lidar_completion.py \
  > /tmp/lidar_completion_buggy.py
cp src/lidar_anchored_depth/pipeline/lidar_completion.py /tmp/_save.py
cp /tmp/lidar_completion_buggy.py \
  src/lidar_anchored_depth/pipeline/lidar_completion.py

lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --output.run-id ablation_outlier_off

# Restore the fix:
cp /tmp/_save.py src/lidar_anchored_depth/pipeline/lidar_completion.py

# And the "with-fix" panel:
lad complete \
    --scene.scene 008 \
    --scene.data-root /mnt/car_road_data_TianJin \
    --complete.runtime.gpu-ids 0 1 2 3 \
    --complete.residual-checkpoint preview/point_completion_ckpt_v2/best.pt \
    --output.run-id ablation_outlier_on
```

Crop each BEV to a region around a pole and a few metres of
surrounding road.

**Post-processing.** Side-by-side with labels and pole markers.

**Save as.** `paper/figures/ablation_outlier_ref.png`

---

## Convenience: render all BEVs in one shot

Once all the `lad …` runs above are done, this Python loop renders
the BEVs for every PLY artefact at once:

```bash
PYTHONPATH=src python - <<'PY'
import os, numpy as np
from pathlib import Path
from PIL import Image
from lidar_anchored_depth.reconstruction import read_ply_xyz_rgb
from lidar_anchored_depth.viz.bev import auto_bev_range, render_bev

PLY_LIST = [
    ("outputs/008/complete/latest/baseline.ply",                 "008_baseline_bev.png"),
    ("outputs/008/complete/latest/refined.ply",                  "008_refined_bev.png"),
    ("outputs/008/complete/latest/hybrid.ply",                   "008_hybrid_bev.png"),
    ("outputs/002/complete/latest/baseline.ply",                 "002_baseline_bev.png"),
    ("outputs/002/complete/latest/refined.ply",                  "002_refined_bev.png"),
    ("outputs/008/complete/ablation_skipground_on/hybrid.ply",   "ablation_skipground_on.png"),
    ("outputs/008/complete/ablation_skipground_off/hybrid.ply",  "ablation_skipground_off.png"),
    ("outputs/008/complete/ablation_labels_interp/hybrid.ply",   "ablation_labels_interp.png"),
    ("outputs/008/complete/ablation_labels_handlabel/hybrid.ply","ablation_labels_handlabel.png"),
]

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)
for ply, png in PLY_LIST:
    if not Path(ply).is_file():
        print(f"SKIP (missing): {ply}")
        continue
    xyz, rgb = read_ply_xyz_rgb(ply)
    xr, yr = auto_bev_range(xyz, pad_m=2.0)
    img = render_bev(xyz, rgb, x_range=xr, y_range=yr,
                     image_hw=(2048, 2048))
    Image.fromarray(img).save(out_dir / png)
    print(f"wrote {out_dir / png}  ({img.shape[1]}x{img.shape[0]})")
PY
```
