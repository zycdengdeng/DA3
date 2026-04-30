"""Tests for the shared SAM mask loader and binary dilation."""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.segmentation.sam_io import (
    _dilate_bool_mask,
    load_sam_dynamic_mask,
    load_sam_masks_dict,
)


def test_dilate_zero_is_identity():
    m = np.zeros((20, 20), dtype=bool)
    m[10, 10] = True
    out = _dilate_bool_mask(m, 0)
    assert np.array_equal(out, m)


def test_dilate_single_pixel_grows_to_3x3():
    m = np.zeros((20, 20), dtype=bool)
    m[10, 10] = True
    out = _dilate_bool_mask(m, 1)
    expected = np.zeros_like(m)
    expected[9:12, 9:12] = True
    assert np.array_equal(out, expected)


def test_dilate_5px_grows_to_11x11():
    m = np.zeros((40, 40), dtype=bool)
    m[20, 20] = True
    out = _dilate_bool_mask(m, 5)
    expected = np.zeros_like(m)
    expected[15:26, 15:26] = True
    assert np.array_equal(out, expected)


def test_dilate_empty_mask_stays_empty():
    m = np.zeros((20, 20), dtype=bool)
    out = _dilate_bool_mask(m, 5)
    assert not out.any()


def test_dilate_negative_or_zero_returns_input():
    m = np.zeros((20, 20), dtype=bool)
    m[5, 5] = True
    assert _dilate_bool_mask(m, -3) is m or np.array_equal(_dilate_bool_mask(m, -3), m)


def test_dilate_does_not_shrink_existing_pixels():
    m = np.zeros((40, 40), dtype=bool)
    m[10:20, 10:20] = True
    out = _dilate_bool_mask(m, 3)
    # Every original True must still be True
    assert (out & m).sum() == m.sum()
    # And the bounding box has grown by 3 on each side
    rows, cols = np.where(out)
    assert rows.min() == 7
    assert rows.max() == 22
    assert cols.min() == 7
    assert cols.max() == 22


def _write_fake_sam_npz(path, masks: list[np.ndarray], ids: list[int]) -> None:
    arr = np.stack(masks).astype(bool) if masks else np.zeros((0, 1, 1), dtype=bool)
    np.savez(
        path,
        masks=arr,
        mask_ids=np.array(ids, dtype=np.int64),
    )


def test_load_sam_masks_dict_round_trip(tmp_path):
    H, W = 30, 40
    m1 = np.zeros((H, W), dtype=bool); m1[5:10, 5:10] = True
    m2 = np.zeros((H, W), dtype=bool); m2[20:25, 20:30] = True
    p = tmp_path / "008_ts1234_cam0_sam.npz"
    _write_fake_sam_npz(p, [m1, m2], [7, 13])

    out = load_sam_masks_dict(tmp_path, "008", 1234, "0")
    assert set(out.keys()) == {7, 13}
    assert np.array_equal(out[7], m1)
    assert np.array_equal(out[13], m2)


def test_load_sam_masks_dict_missing_returns_empty(tmp_path):
    out = load_sam_masks_dict(tmp_path, "008", 9999, "0")
    assert out == {}


def test_load_sam_masks_dict_none_dir_returns_empty():
    assert load_sam_masks_dict(None, "008", 1234, "0") == {}


def test_load_sam_dynamic_mask_unions_with_dilation(tmp_path):
    H, W = 30, 40
    m1 = np.zeros((H, W), dtype=bool); m1[10, 10] = True
    m2 = np.zeros((H, W), dtype=bool); m2[20, 20] = True
    p = tmp_path / "008_ts1234_cam0_sam.npz"
    _write_fake_sam_npz(p, [m1, m2], [1, 2])

    raw = load_sam_dynamic_mask(tmp_path, "008", 1234, "0", (H, W))
    assert raw.sum() == 2

    dil = load_sam_dynamic_mask(tmp_path, "008", 1234, "0", (H, W), dilate_px=2)
    # Each single pixel becomes a 5x5 patch -> 25 each, no overlap
    assert dil.sum() == 50
    assert dil[10, 10] and dil[8, 8] and dil[12, 12]
    assert dil[20, 20] and dil[18, 18] and dil[22, 22]


def test_load_sam_dynamic_mask_missing_file_returns_zeros(tmp_path):
    out = load_sam_dynamic_mask(tmp_path, "008", 9999, "0", (10, 12), dilate_px=5)
    assert out.shape == (10, 12)
    assert not out.any()


def test_load_sam_dynamic_mask_empty_masks_array_returns_zeros(tmp_path):
    p = tmp_path / "008_ts1234_cam0_sam.npz"
    np.savez(
        p,
        masks=np.zeros((0, 10, 12), dtype=bool),
        mask_ids=np.array([], dtype=np.int64),
    )
    out = load_sam_dynamic_mask(tmp_path, "008", 1234, "0", (10, 12), dilate_px=3)
    assert not out.any()
