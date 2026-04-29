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
| `gt_depth` | `np.ndarray (H, W)`, `float32` or `None` | optional; meters; `0` or `nan` = invalid | no |
| `sam_masks` | `np.ndarray (K, H, W)`, `bool` or `None` | optional pre-computed instance masks | no |
| `meta` | `dict` | dataset-specific extras (timestamp, scene-id, etc.) | no |

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
