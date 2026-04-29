"""Tests for ``data.pcd_io``.

The ASCII fallback is exercised here. open3d-handled binary PCDs are
implicitly covered when running ``preview_frame.py`` on real data.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from lidar_anchored_depth.data.pcd_io import _read_pcd_ascii, read_pcd_xyz


def _write_ascii_pcd(path, xyz: np.ndarray) -> None:
    n = len(xyz)
    header = textwrap.dedent(
        f"""\
        # generated for tests
        VERSION 0.7
        FIELDS x y z
        SIZE 4 4 4
        TYPE F F F
        COUNT 1 1 1
        WIDTH {n}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {n}
        DATA ascii
        """
    )
    body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in xyz)
    path.write_text(header + body + "\n")


def test_ascii_roundtrip(tmp_path):
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1.5, -2.5, 3.5], [-10.0, 20.0, -1.5]],
        dtype=np.float32,
    )
    p = tmp_path / "x.pcd"
    _write_ascii_pcd(p, xyz)
    out = _read_pcd_ascii(p)
    assert out.shape == xyz.shape
    np.testing.assert_allclose(out, xyz, atol=1e-5)


def test_read_pcd_xyz_uses_ascii_when_open3d_missing(monkeypatch, tmp_path):
    # Force ImportError on open3d
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "open3d":
            raise ImportError("forced")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    p = tmp_path / "y.pcd"
    _write_ascii_pcd(p, xyz)
    out = read_pcd_xyz(p)
    np.testing.assert_allclose(out, xyz, atol=1e-5)


def test_missing_xyz_fields_raises(tmp_path):
    p = tmp_path / "bad.pcd"
    p.write_text(
        textwrap.dedent(
            """\
            VERSION 0.7
            FIELDS rgb
            SIZE 4
            TYPE F
            COUNT 1
            WIDTH 1
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS 1
            DATA ascii
            255
            """
        )
    )
    with pytest.raises(ValueError, match="missing x/y/z"):
        _read_pcd_ascii(p)


def test_binary_pcd_without_open3d_raises_clearly(tmp_path):
    p = tmp_path / "bin.pcd"
    p.write_text(
        textwrap.dedent(
            """\
            VERSION 0.7
            FIELDS x y z
            SIZE 4 4 4
            TYPE F F F
            COUNT 1 1 1
            WIDTH 1
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS 1
            DATA binary
            """
        )
    )
    with pytest.raises(RuntimeError, match="install open3d"):
        _read_pcd_ascii(p)


def test_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pcd_xyz(tmp_path / "missing.pcd")
