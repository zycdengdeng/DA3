# THICV-R2A — Dataset Guide (project copy)

> Verbatim copy of the user-maintained dataset guide, anchored here so any
> agent / contributor can read it without depending on the user's local
> machine. Keep this in sync with the canonical version on the user's
> desktop (`Literature_Review/THICV-R2A_dataset_guide.md`).
>
> Last sync: 2026-04-29

---

## 0. TL;DR (30 seconds to ground truth)

```
Dataset name        : THICV-R2A
Server data root    : /mnt/car_road_data_TianJin/        (verified)
Scenes              : 89 sessions of ONE intersection
Roadside frames     : 12,891    (≈ 0.49 hours continuous)
Roadside 3D objects : 1,083,109 across 20 classes
Car-side 3D objects :   661,515 across 22 classes
Car-side images     : 208,348   (29,764 frames × 7 cameras)
```

Our project (`lidar_anchored_depth`) operates on the **road-side stream
only** (cameras + LiDAR + V2X annotations). Car-side data is intentionally
unused — we are building a **pure roadside reconstruction**.

---

## 1. Server root & layout

### 1.1 Data root

```
/mnt/car_road_data_TianJin/
├── 001_car0325_road0327_t1/      ← 89 scene directories
├── 002_car0325_road0327_t2/
├── ...
├── 089_car0402_road0402_t71/
└── support_info/                 ← calib + carid + sensor configs
```

The legacy constant `/mnt/car_road_data_fix/` is **deprecated** — always
use `/mnt/car_road_data_TianJin/`.

### 1.2 One scene (example: scene 008)

```
008_car0325_road0327_t8/
├── car/                 ← UNUSED in this project
├── car_labels/          ← UNUSED in this project
├── road/
│   ├── cameras/
│   │   ├── pinhole0/   cam3_*.png    ← 4 pinhole, see §1.4
│   │   ├── pinhole1/   cam6_*.png
│   │   ├── pinhole2/   cam9_*.png
│   │   ├── pinhole3/   cam0_*.png
│   │   ├── fisheye0/  ... fisheye3/  ← UNUSED
│   └── lidar/
│       ├── lidar0/  *.pcd            ← 4 individual LiDARs
│       ├── lidar1/  *.pcd
│       ├── lidar2/  *.pcd
│       ├── lidar3/  *.pcd
│       └── merged_pcd/  *.pcd        ← merged in world frame (preferred)
├── road_labels/
│   ├── interpolation_labels/  *.json ← 3D bbox annotations (core)
│   └── merged_pcd_all/        *.pcd  ← merged PCD aligned to label timestamps
└── sync_info.txt
```

### 1.3 Naming conventions

| Source | Format | Unit | Example |
|---|---|---|---|
| Roadside images / pcds / labels | `cam{N}_{TS}.png`, `{TS}.pcd`, `{TS}.json` | ms (int) | `1742877031036` |
| Car-side files (UNUSED) | `addc_{date}_..._{ts}.{jpg,pcd,json}` | s (float) | `1742877030.849978` |

Conversion: `road_ts_ms = int(car_ts_sec * 1000)`.

Sampling rates: roadside sensors / labels at ~7–8 Hz (130–150 ms gap).

### 1.4 Pinhole folder ↔ camera-id mapping

The folder name (`pinhole0..3`) and the file prefix (`cam0/3/6/9_`) do
**not** agree. The mapping is fixed:

| Folder | File prefix | Calib `camera_id` |
|---|---|---|
| `pinhole0/` | `cam3_*` | 3 |
| `pinhole1/` | `cam6_*` | 6 |
| `pinhole2/` | `cam9_*` | 9 |
| `pinhole3/` | `cam0_*` | 0 |

The 4 pinhole cameras (cam0, cam3, cam6, cam9) are the **only camera
stream this project uses**. Fisheye cameras (cam2, 5, 8, 11) and all
car-side cameras are ignored.

---

## 2. Annotation format

### 2.1 Roadside annotation `road_labels/interpolation_labels/*.json`

```jsonc
{
  "timestamp": "1742877031036",       // string, ms
  "image_file": {},                    // usually empty in roadside-only mode
  "pcd_file":  "1742877031036.pcd",    // matching merged PCD filename
  "object": [
    {
      "id": 1,                                              // tracking id, RESETS per scene
      "label": "Car",
      "x": -31.197, "y": -9.21, "z": -1.71,                 // bbox CENTER in world frame, m
      "length": 5.41, "width": 2.06, "height": 1.49,        // m
      "roll": 0.0, "pitch": 0.0, "yaw": 3.083,              // ZYX Euler angles (rad)
      "occlusion": 0,                                        // {0, 1, 2, 3}
      "num_points": 697,
      "vx": 0.5, "vy": -0.1                                  // world-frame velocity, m/s
    }
  ]
}
```

Critical: per-frame `id` **resets between scenes**. To form a globally
unique key, use `(scene_id, instance_id)`.

### 2.2 Class table (roadside, 20 classes)

The full list with counts is in the canonical guide. The split that
matters here:

| Group | Classes | Note |
|---|---|---|
| Vehicle | Car, Suv, Truck, Bus, Huge_vehicle, Vehicle_else | most "dynamic" objects |
| Two-wheeler / pedestrian | Pedestrian, Pedestrian_else, Tricycle, Motorcycle, Motor_rider, Bicycle, Non_motor_rider, Other_rider | shorter, non-rigid |
| Static fixtures | **Bollards (14.9%)**, **Crash_bucket (2.3%)**, **Cone (0.01%)** | **Treat as scene-static**: solve once, reuse |
| Misc | Vehicle_door, Animal_small, Unknown | rare, low priority |

The "Static fixtures" group (~17.2% of all annotations) is treated as
**scene-static** in our pipeline: the first session in which a fixture
appears anchors its `(Z_min, Z_max)`; subsequent sessions reuse the
anchor without re-solving.

---

## 3. Calibration

### 3.1 `support_info/calib.json`

Single calibration applies to the entire intersection across all 89
sessions (rig is fixed). Schema:

```python
{
  "imgSize": {"fish": [1280, 1280], "notFish": [1280, 720]},
  "lidar":  { "0": {...}, "1": {...}, "2": {...}, "3": {...} },
  "camera": { "0": {...}, "2": {...}, "3": {...}, "5": {...},
              "6": {...}, "8": {...}, "9": {...}, "11": {...} }
}
```

**LiDAR record** (each `lidar.{i}`):
```python
{
  "name": "rad{i}_<ts>.pcd",
  "lidarToVirtualLidar": {
    "rotateMatrix": [9 floats, row-major 3x3],
    "trans":        [3 floats]
  }
}
```

**Camera record** (each `camera.{N}`):
```python
{
  "name":   "cam{N}_<ts>.png",
  "isFish": 0 | 1,
  "intri":  [9 floats, row-major K (3x3)],
  "distor": [5 floats for pinhole (plumb_bob) | 4 floats for fisheye],
  "virtualLidarToCam": {
    "rotate": [3 floats, Rodrigues vector],
    "trans":  [3 floats]
  }
}
```

The world frame is the **VirtualLidar** frame. Pipeline:

```
LiDAR_i  ─(R_L2V_i, t_L2V_i)→  VirtualLidar  ─(Rod(rot_V2C), t_V2C)→  Camera_c  → image
```

### 3.2 carid.json — ego id per session

`support_info/carid.json` maps each scene to the V2X bbox `id` that is
the actual collection vehicle (an ego car driving through the
intersection). For a roadside-only project this is mostly informational
— useful to mask out the ego from "dynamic objects" if needed.

```json
{
  "metadata": {
    "origin_gps": { "latitude": 39.71767659, "longitude": 117.2841005,
                    "R_earth": 6378137.0 }
  },
  "results": [
    { "clip_name": "001_car0325_road0327_t1", "nearest_carid": 45,
      "nearest_label": "Car", ... },
    ...  // 89 entries
  ]
}
```

`nearest_label` may be stale relative to the latest annotations. **Look
up the ego by `id` only, never combine with `label`.**

---

## 4. Coordinate frames

| Frame | Origin | Axes | What lives here |
|---|---|---|---|
| **World (= VirtualLidar)** | LiDAR rig zero | x east, y north, z up | road-label `(x,y,z,yaw)`, merged PCD |
| Per-LiDAR | each unit's own zero | LiDAR-local | individual `lidar{i}/*.pcd` |
| Per-camera | camera centre | OpenCV (x right, y down, z fwd) | image plane |

ZYX Euler convention for `(roll, pitch, yaw)`:

```python
def euler_zyx_to_R(roll, pitch, yaw):
    Rx = [[1,0,0],[0,cos r,-sin r],[0,sin r, cos r]]
    Ry = [[cos p,0,sin p],[0,1,0],[-sin p,0,cos p]]
    Rz = [[cos y,-sin y,0],[sin y,cos y,0],[0,0,1]]
    return Rz @ Ry @ Rx
```

**Do not** call `cv2.Rodrigues([roll, pitch, yaw])` — that interprets
the triple as an axis-angle and silently produces the wrong rotation.

---

## 5. Pitfalls (cross-checked against this guide)

1. **ZYX Euler vs Rodrigues**: see §4
2. **Pinhole folder ↔ camera id**: `pinhole0/` contains `cam3_*.png` (§1.4)
3. **`id` resets per scene**: use `(scene_id, instance_id)`
4. **Roadside ts in ms (int), car-side in s (float)**: see §1.3
5. **Static fixture classes** are scene-static, do not re-solve every frame (§2.2)
6. **`carid.json.nearest_label` may be stale**: look up by `id` only
7. **PCD-image timestamp tolerance**: 500 ms in upstream pipeline
8. **fisheye / car-side**: deliberately unused in this project

---

## 6. Sanity checks (the agent should run all 5 once data is mounted)

```bash
# 1. data root present, count
test -d /mnt/car_road_data_TianJin && \
  ls /mnt/car_road_data_TianJin | wc -l                        # expect 90

# 2. target scene structure
ls /mnt/car_road_data_TianJin/008_car0325_road0327_t8/         # expect car/ car_labels/ road/ road_labels/ sync_info.txt

# 3. roadside annotations readable
ls /mnt/car_road_data_TianJin/008_*/road_labels/interpolation_labels/ | wc -l  # expect 163

# 4. calibration files present
test -f /mnt/car_road_data_TianJin/support_info/calib.json
test -d /mnt/car_road_data_TianJin/support_info/NoEER705_v3/camera/

# 5. pinhole folder ↔ cam{3,6,9,0} mapping
for i in 0 1 2 3; do
  ls /mnt/car_road_data_TianJin/008_*/road/cameras/pinhole$i/ | head -1
done   # expect cam3_*, cam6_*, cam9_*, cam0_* in that order
```

A Python equivalent lives at
`projects/lidar_anchored_depth/scripts/sanity_check_dataset.py`.

---

## 7. Where this connects to the project

| Project component | Reads from this dataset |
|---|---|
| `data.calibration.SceneCalibration` | `support_info/calib.json` |
| `data.carid_lookup.CaridLookup` | `support_info/carid.json` |
| `data.roadside_v2x.RoadsideV2XLoader` | each scene's `road/` + `road_labels/` |
| `Frame.lidar_world` | `road_labels/merged_pcd_all/<ts>.pcd` |
| `Frame.image` | `road/cameras/pinhole{i}/cam{3,6,9,0}_<ts>.png` |
| `Frame.dynamic_objects` | `road_labels/interpolation_labels/<ts>.json` minus static fixtures |
| `Loader.scene_static_fixtures` | the same JSON, filtered to `{Bollards, Crash_bucket, Cone}` |
