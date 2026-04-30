"""Tiny in-house PLY writer / reader (XYZ + optional RGB).

Used to dump per-object accumulated point clouds for visualization in
MeshLab / CloudCompare / Blender / open3d / etc. without forcing an
open3d dependency on the eval pipeline. Writes ASCII PLY by default and
binary little-endian on demand.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_ply_xyz(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    binary: bool = False,
) -> None:
    """Write ``(N, 3)`` XYZ points (and optional ``(N, 3)`` uint8 RGB) as PLY.

    Parameters
    ----------
    path : output path. ``.ply`` extension is conventional.
    points : ``(N, 3)`` float, world-frame XYZ.
    colors : optional ``(N, 3)`` uint8 RGB. ``None`` writes geometry only.
    binary : if True, writes binary little-endian (faster + smaller for
        large clouds). Default ASCII for human-readability.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]

    has_rgb = colors is not None
    if has_rgb:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != n:
            raise ValueError(
                f"colors {colors.shape} must align with points {points.shape}"
            )

    fmt = "binary_little_endian" if binary else "ascii"
    header = [
        "ply",
        f"format {fmt} 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_rgb:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")

    if binary:
        with p.open("wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            if has_rgb:
                rec_dtype = np.dtype(
                    [("xyz", "<f4", 3), ("rgb", "u1", 3)]
                )
                rec = np.empty(n, dtype=rec_dtype)
                rec["xyz"] = points
                rec["rgb"] = colors
                f.write(rec.tobytes())
            else:
                f.write(points.tobytes())
    else:
        with p.open("w", encoding="ascii") as f:
            f.write("\n".join(header) + "\n")
            if has_rgb:
                for (x, y, z), (r, g, b) in zip(points, colors):
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                for x, y, z in points:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read XYZ from an ASCII or binary little-endian PLY.

    Returns ``(N, 3) float32``. Color channels are skipped. Designed for
    PLYs written by :func:`write_ply_xyz` plus other simple variants.
    """
    p = Path(path)
    with p.open("rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: malformed PLY header")
            header_bytes += line
            if line.strip() == b"end_header":
                break

        header = header_bytes.decode("ascii", errors="replace").splitlines()
        fmt_line = next(line for line in header if line.startswith("format"))
        n_vertex_line = next(
            line for line in header if line.startswith("element vertex")
        )
        n = int(n_vertex_line.split()[-1])
        props = [
            line for line in header if line.startswith("property")
        ]
        prop_names = [p.split()[-1] for p in props]
        prop_types = [p.split()[1] for p in props]

        ix = prop_names.index("x")
        iy = prop_names.index("y")
        iz = prop_names.index("z")

        if "binary_little_endian" in fmt_line:
            type_to_struct = {
                "float": ("f", 4), "float32": ("f", 4),
                "double": ("d", 8), "float64": ("d", 8),
                "uchar": ("B", 1), "uint8": ("B", 1),
                "char": ("b", 1), "int8": ("b", 1),
                "ushort": ("H", 2), "uint16": ("H", 2),
                "short": ("h", 2), "int16": ("h", 2),
                "uint": ("I", 4), "uint32": ("I", 4),
                "int": ("i", 4), "int32": ("i", 4),
            }
            chars = "".join(type_to_struct[t][0] for t in prop_types)
            sizes = sum(type_to_struct[t][1] for t in prop_types)
            rec_size = struct.calcsize("<" + chars)
            assert rec_size == sizes
            data = f.read(n * rec_size)
            unpacked = struct.iter_unpack("<" + chars, data)
            out = np.empty((n, 3), dtype=np.float32)
            for k, vals in enumerate(unpacked):
                out[k, 0] = vals[ix]
                out[k, 1] = vals[iy]
                out[k, 2] = vals[iz]
            return out
        else:  # ASCII
            body = f.read().decode("ascii", errors="replace")
            out = np.empty((n, 3), dtype=np.float32)
            for k, line in enumerate(body.splitlines()[:n]):
                toks = line.split()
                out[k, 0] = float(toks[ix])
                out[k, 1] = float(toks[iy])
                out[k, 2] = float(toks[iz])
            return out
