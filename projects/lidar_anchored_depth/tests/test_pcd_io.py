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


def test_binary_pcd_uncompressed_xyz_f32(tmp_path):
    """Native parser handles DATA binary (uncompressed, FIELDS x y z F4)."""
    xyz = np.array(
        [[1.5, -2.5, 3.5], [-10.0, 20.0, -1.5], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    header = textwrap.dedent(
        f"""\
        VERSION 0.7
        FIELDS x y z
        SIZE 4 4 4
        TYPE F F F
        COUNT 1 1 1
        WIDTH {len(xyz)}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {len(xyz)}
        DATA binary
        """
    ).encode("ascii")
    body = xyz.tobytes()
    p = tmp_path / "bin.pcd"
    p.write_bytes(header + body)
    out = _read_pcd_ascii(p)
    np.testing.assert_allclose(out, xyz, atol=1e-6)


def test_binary_pcd_with_extra_intensity_field(tmp_path):
    """Native parser strips intensity (just keeps x/y/z)."""
    n = 4
    xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]],
                   dtype=np.float32)
    intensity = np.array([100, 200, 50, 25], dtype=np.float32)
    header = textwrap.dedent(
        f"""\
        VERSION 0.7
        FIELDS x y z intensity
        SIZE 4 4 4 4
        TYPE F F F F
        COUNT 1 1 1 1
        WIDTH {n}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {n}
        DATA binary
        """
    ).encode("ascii")
    # Interleave xyz + intensity per-point
    body = b"".join(
        np.array([xyz[i, 0], xyz[i, 1], xyz[i, 2], intensity[i]],
                 dtype=np.float32).tobytes()
        for i in range(n)
    )
    p = tmp_path / "bin.pcd"
    p.write_bytes(header + body)
    out = _read_pcd_ascii(p)
    np.testing.assert_allclose(out, xyz, atol=1e-6)


def test_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pcd_xyz(tmp_path / "missing.pcd")


# --------------------------------------------------------------------- #
# LZF decoder & binary_compressed
# --------------------------------------------------------------------- #
import struct  # noqa: E402

from lidar_anchored_depth.data.pcd_io import _lzf_decompress  # noqa: E402


def _lzf_compress_literal_only(data: bytes) -> bytes:
    """Encode a byte string as LZF using only literal runs (chunks of 32)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        chunk = min(32, n - i)
        out.append(chunk - 1)  # literal-run ctrl byte = run_len - 1
        out.extend(data[i : i + chunk])
        i += chunk
    return bytes(out)


def test_lzf_literal_only_roundtrip():
    msg = b"abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()" * 5
    enc = _lzf_compress_literal_only(msg)
    dec = _lzf_decompress(enc, len(msg))
    assert dec == msg


def test_lzf_simple_backref():
    """Hand-encoded 'abcabcabc': literal 'abc' + backref(len=6, dist=3).

    Encoding:
        ctrl 0x02 (literal run length = 3) + 'abc'                 = 4 bytes
        backref: run_len_enc = 6 - 2 = 4; ref_offset_enc = 3 - 1 = 2
            ctrl = (4 << 5) | (2 >> 8) = 0x80
            offset_low = 2
        ctrl 0x80 + 0x02                                            = 2 bytes
    """
    enc = bytes([0x02]) + b"abc" + bytes([0x80, 0x02])
    out = _lzf_decompress(enc, 9)
    assert out == b"abcabcabc"


def test_lzf_overlap_rle_pattern():
    """Backref with len > distance: RLE-like extension."""
    # Literal 'x' then backref(len=5, dist=1) → 'xxxxxx'
    # ctrl 0x00 (literal run 1) + 'x'                  = 2 bytes
    # backref: run_len_enc = 5 - 2 = 3; offset_enc = 0
    #   ctrl = (3 << 5) | 0 = 0x60; offset_low = 0
    enc = bytes([0x00]) + b"x" + bytes([0x60, 0x00])
    out = _lzf_decompress(enc, 6)
    assert out == b"xxxxxx"


def test_binary_compressed_xyz(tmp_path):
    """Build a binary_compressed PCD with literal-only LZF and round-trip."""
    xyz = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-7.0, -8.0, -9.0]],
        dtype=np.float32,
    )
    n = len(xyz)
    # SoA layout: all x's, then all y's, then all z's
    soa = (
        xyz[:, 0].tobytes() + xyz[:, 1].tobytes() + xyz[:, 2].tobytes()
    )
    compressed = _lzf_compress_literal_only(soa)

    header = textwrap.dedent(
        f"""\
        VERSION 0.7
        FIELDS x y z
        SIZE 4 4 4
        TYPE F F F
        COUNT 1 1 1
        WIDTH {n}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {n}
        DATA binary_compressed
        """
    ).encode("ascii")
    sz = struct.pack("<II", len(compressed), len(soa))
    p = tmp_path / "bc.pcd"
    p.write_bytes(header + sz + compressed)

    out = _read_pcd_ascii(p)
    np.testing.assert_allclose(out, xyz, atol=1e-6)


def test_binary_compressed_with_intensity(tmp_path):
    """SoA layout with extra intensity field; only x/y/z extracted."""
    n = 4
    xyz = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]], dtype=np.float32
    )
    intensity = np.array([10, 20, 30, 40], dtype=np.float32)
    soa = (
        xyz[:, 0].tobytes()
        + xyz[:, 1].tobytes()
        + xyz[:, 2].tobytes()
        + intensity.tobytes()
    )
    compressed = _lzf_compress_literal_only(soa)
    header = textwrap.dedent(
        f"""\
        VERSION 0.7
        FIELDS x y z intensity
        SIZE 4 4 4 4
        TYPE F F F F
        COUNT 1 1 1 1
        WIDTH {n}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {n}
        DATA binary_compressed
        """
    ).encode("ascii")
    sz = struct.pack("<II", len(compressed), len(soa))
    p = tmp_path / "bci.pcd"
    p.write_bytes(header + sz + compressed)

    out = _read_pcd_ascii(p)
    np.testing.assert_allclose(out, xyz, atol=1e-6)
