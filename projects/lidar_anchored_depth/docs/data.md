# Data card & loader contract

## 1. Frame schema

The library is dataset-agnostic. All algorithms consume a single `Frame`
object, defined in `src.lidar_anchored_depth.data.base`. A Frame must
provide:

| Field | Type | Units / convention | Required |
|---|---|---|---|
| `frame_id` | `str` | unique per dataset | yes |
| `image` | `np.ndarray (H, W, 3)`, `uint8`, RGB | — | yes |
| `K` | `np.ndarray (3, 3)`, `float64` | OpenCV pinhole, in pixels | yes |
| `T_wc` | `np.ndarray (4, 4)`, `float64` | Camera-to-world `SE(3)`. World Z is **up** (positive Z = above ground) | yes |
| `lidar_world` | `np.ndarray (N, 3)`, `float32` | LiDAR points in world frame, meters | yes |
| `dynamic_objects` | `list[DynamicObject]` or `None` | V2X 3D-detection annotations; see §3 | no |
| `image_ts_ms` | `int` or `None` | image timestamp in milliseconds (UTC) — required if `lidar_ts_ms` differs and `dynamic_objects` carry velocity | no |
| `lidar_ts_ms` | `int` or `None` | LiDAR sweep timestamp in milliseconds (UTC) | no |
| `gt_depth` | `np.ndarray (H, W)`, `float32` or `None` | optional; meters; `0` or `nan` = invalid | no |
| `sam_masks` | `np.ndarray (K, H, W)`, `bool` or `None` | optional pre-computed instance masks | no |
| `meta` | `dict` | dataset-specific extras (scene-id, sensor IDs, ...) | no |

### Conventions in detail

- **World frame.** Right-handed, Z-up. Ground plane is approximately
  `Z = 0` (small slope OK; HAD's ground branch fits the actual plane).
- **Camera frame (OpenCV).** X right, Y down, Z forward. Internal to the
  loader; algorithms see `T_wc` only.
- **`T_wc` semantics.** `P_world = T_wc · [P_camera; 1]`. Therefore
  `T_cw = inv(T_wc)` projects world → camera. Be explicit when adapting
  a dataset whose calibration ships `T_cw` (e.g., KITTI's `Tr_cam_to_velo`).
- **Distortion.** Algorithms assume undistorted images. If your dataset
  ships distortion coefficients, undistort upstream and adjust `K`.
- **LiDAR units.** Always meters in world frame. Reflectance/intensity is
  ignored.

## 1.1 The `DynamicObject` schema

Each entry in `Frame.dynamic_objects` is a `DynamicObject` dataclass
(see `src.lidar_anchored_depth.data.base`):

| Field | Type | Units / convention |
|---|---|---|
| `id` | `int` | tracking ID, stable across frames within a scene |
| `label` | `str` | class name, e.g. `"Car"`, `"Truck"`, `"Non_motor_rider"` |
| `xyz` | `np.ndarray (3,)` `float64` | bbox **center** in world frame, meters |
| `lwh` | `np.ndarray (3,)` `float64` | length, width, height in meters |
| `yaw` | `float` | rotation around world Z, radians (right-handed) |
| `roll`, `pitch` | `float` | typically `0.0` for ground vehicles |
| `velocity_xy` | `np.ndarray (2,)` `float64` | `(vx, vy)` in world frame, m/s; `(0, 0)` for stationary |
| `occlusion` | `int` | `{0, 1, 2}` per dataset convention; AA-HAD filters `>= 2` |
| `num_points` | `int` | LiDAR returns inside the bbox at the LiDAR timestamp |

`DynamicObject.Z_min` and `DynamicObject.Z_max` are derived properties:
```
Z_min = xyz[2] - lwh[2] / 2
Z_max = xyz[2] + lwh[2] / 2
```
These are consumed directly by AA-HAD as the height anchors (see
`docs/method.md` §4.3).

## 3. The roadside-V2X annotation JSON

Adapters that target deployments producing standard V2X 3D-detection
logs should consume per-timestamp JSON entries in this canonical form:

```jsonc
{
  "timestamp": "1743645353317",                 // ms, UTC
  "image_file": {
    "cam0":  "<path>/cam0_<ts>.png",
    "cam2":  "<path>/cam2_<ts>.png",
    "cam3":  "<path>/cam3_<ts>.png",
    "cam5":  "...", "cam6": "...",
    "cam8":  "...", "cam9": "...", "cam11": "..."
  },
  "pcd_file": "<path>/merged_pcd_all/<ts>.pcd",  // already merged across LiDARs
  "object": [
    {
      "id": 1, "label": "Car",
      "x": -61.96, "y": 14.01, "z": -1.46,       // bbox CENTER, world frame, meters
      "length": 4.45, "width": 2.43, "height": 1.97,
      "roll": 0.0, "pitch": 0.0, "yaw": -1.69,
      "occlusion": 0,                             // {0, 1, 2}
      "num_points": 1472,
      "vx": 0.0, "vy": 0.0                        // world m/s
    },
    /* ... */
  ]
}
```

Conventions to verify on first ingest of any new V2X dataset:

1. `z` is the bbox **center** (we have empirically observed bottoms
   clustering around `Z ≈ -2.4 m`, consistent with a roadside LiDAR
   mounted ~2.4 m above ground when `z_center − height/2` is computed
   per object). If a deployment uses **bottom-z** convention, the
   adapter must add `height/2` upon ingest.
2. `vx, vy` are world-frame, not body-frame, m/s.
3. `merged_pcd_all` denotes a multi-LiDAR fusion already in world frame.
   Single-LiDAR deployments simply provide one cloud at the same path.
4. The `image_file` map's keys are camera IDs; the per-camera intrinsics
   and extrinsics live in a separate calibration JSON (one per scene),
   not the per-frame entry.

## 2. Adding a new dataset

Subclass `BaseDataset` and implement `__len__` and `get_frame(idx)`.
A minimal example:

```python
from lidar_anchored_depth.data.base import BaseDataset, Frame

class MyRoadside(BaseDataset):
    def __init__(self, root: str):
        self.root = root
        self.entries = list_my_entries(root)

    def __len__(self):
        return len(self.entries)

    def get_frame(self, idx: int) -> Frame:
        e = self.entries[idx]
        return Frame(
            frame_id=e.id,
            image=load_rgb(e.image_path),
            K=load_K(e.calib_path),
            T_wc=load_T_wc(e.calib_path),
            lidar_world=load_lidar_world(e.lidar_path, e.calib_path),
            gt_depth=None,           # fill if you have it
            sam_masks=None,          # auto-computed downstream if absent
            meta={"scene": e.scene, "ts": e.ts},
        )
```

Then register in `configs/data/<your_dataset>.yaml`:

```yaml
_target_: lidar_anchored_depth.data.<your_module>.MyRoadside
root: /path/to/data
```

## 3. Built-in adapters (planned)

| Adapter | Status | Source |
|---|---|---|
| `data.car_road.CarRoadLoader` | ported from prior branch | TianJin self-collected |
| `data.self_data.SelfDataLoader` | ported from `lidar-calibration` branch | 4-camera + LiDAR + `calib.json` snapshot |
| `data.generic.GenericRoadside` | new, schema-only, for the user's other dataset | user-provided format spec |
| `data.dair_v2x.DAIRV2XLoader` | future | DAIR-V2X public |
| `data.kitti.KITTILoader` | future, for sanity benchmark | KITTI / KITTI-360 |

## 4. Calibration sanity checks (in tests/)

For every adapter, the test suite verifies:

1. `T_wc` is a valid `SE(3)` (`R^T R = I`, `det R = +1`).
2. Projecting `lidar_world` into the camera with `T_cw, K` yields a
   non-empty subset inside the image (i.e., the calibration is sane).
3. The fraction of LiDAR returns with `Z < 0.1 m` is at least 30 % (sanity
   for ground hits) — relax if the rig is exotic.
4. `np.linalg.inv(T_wc) @ T_wc ≈ I` to numerical tolerance.

## 5. Metric depth pseudo-GT

For datasets without dense depth GT, we generate **pseudo-GT** by:

1. Accumulating LiDAR sweeps within a short temporal window (default ±2 s)
   in world frame.
2. Filtering dynamic points via segmentation or tracking (or by holding
   only static-class returns: road/building/pole).
3. Projecting the accumulated cloud into the target frame's image and
   keeping the minimum depth per pixel (front-most surface).

The pseudo-GT generator lives in `src/lidar_anchored_depth/data/pseudo_gt.py`
(planned). Quality of pseudo-GT is bounded by calibration; we therefore
report metrics also restricted to **single-sweep LiDAR pixels** as a
sanity check.

## 6. Privacy & licensing

The TianJin Car-Road dataset and the SELF_data calibration capture are
internal. They will *not* be redistributed under this project. The repo
ships only:

- The loader code
- Anonymized example frames if approved
- A clear data-card describing collection protocol and consent
- Pointers to public datasets (DAIR-V2X, KITTI) for external reproduction
