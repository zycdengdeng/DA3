"""Tests for ``data.carid_lookup``."""

from __future__ import annotations

import json

import pytest

from lidar_anchored_depth.data.carid_lookup import load_carid_lookup


@pytest.fixture
def fake_carid_path(tmp_path):
    payload = {
        "metadata": {
            "total_clips": 2,
            "origin_gps": {
                "latitude": 39.71767659,
                "longitude": 117.2841005,
                "R_earth": 6378137.0,
            },
        },
        "results": [
            {
                "clip_name": "001_car0325_road0327_t1",
                "nearest_carid": 45,
                "nearest_label": "Car",
                "visualize_roadtime": "1742877037921",
                "visualize_camera": "3",
            },
            {
                "clip_name": "008_car0325_road0327_t8",
                "nearest_carid": 82,
                "nearest_label": "Suv",
                "visualize_roadtime": "1742879642908",
                "visualize_camera": "0",
            },
        ],
    }
    p = tmp_path / "carid.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_origin(fake_carid_path):
    look = load_carid_lookup(fake_carid_path)
    assert look.origin_lat == pytest.approx(39.71767659)
    assert look.origin_lon == pytest.approx(117.2841005)
    assert look.earth_radius_m == pytest.approx(6378137.0)


def test_get_ego_id(fake_carid_path):
    look = load_carid_lookup(fake_carid_path)
    assert look.get_ego_id("008_car0325_road0327_t8") == 82
    assert look.get_ego_id("001_car0325_road0327_t1") == 45


def test_unknown_scene_returns_none(fake_carid_path):
    look = load_carid_lookup(fake_carid_path)
    assert look.get_ego_id("999_does_not_exist") is None


def test_dunder_contains_and_len(fake_carid_path):
    look = load_carid_lookup(fake_carid_path)
    assert len(look) == 2
    assert "008_car0325_road0327_t8" in look
    assert "999_nope" not in look


def test_visualize_metadata_preserved(fake_carid_path):
    look = load_carid_lookup(fake_carid_path)
    e = look.entries["008_car0325_road0327_t8"]
    assert e.visualize_roadtime == "1742879642908"
    assert e.visualize_camera == "0"
