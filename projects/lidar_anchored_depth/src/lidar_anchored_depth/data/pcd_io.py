"""Read PCD point-cloud files.

Strategy:
- Primary: ``open3d.io.read_point_cloud`` (handles ascii / binary /
  binary_compressed, all major fields).
- Fallback: a minimal in-house parser for ASCII PCDs only. This is just
  for environments where ``open3d`` is unavailable; if the file is binary
  the fallback raises a clear error directing the user to install
  ``open3d``.

Only XYZ is exposed. The downstream pipeline does not need intensity or
RGB at this stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    """Read a PCD file and return its XYZ points as ``(N, 3) float32``.

    Tries ``open3d`` first; if not installed, falls back to a tiny ASCII
    parser. Raises if the file is binary and ``open3d`` is missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pcd = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    except ImportError:
        return _read_pcd_ascii(p)


def _read_pcd_ascii(path: Path) -> np.ndarray:
    """Minimal ASCII PCD reader (XYZ fields only).

    Supports the standard PCD v0.7 header. Raises if ``DATA`` is not
    ``ascii``.
    """
    with path.open("rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: malformed PCD header (no DATA line)")
            header_bytes += line
            if line.lstrip().startswith(b"DATA"):
                break
        rest = f.read()

    header = header_bytes.decode("utf-8", errors="replace")
    fields: list[str] = []
    n_points = 0
    data_kind = "ascii"
    for raw_line in header.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("POINTS"):
            n_points = int(line.split()[1])
        elif line.startswith("DATA"):
            data_kind = line.split()[1].strip().lower()

    if data_kind != "ascii":
        raise RuntimeError(
            f"{path}: DATA={data_kind!r}; install open3d to read non-ASCII "
            "PCDs (`pip install open3d`)"
        )

    if not all(c in fields for c in ("x", "y", "z")):
        raise ValueError(f"{path}: PCD missing x/y/z fields (FIELDS={fields})")

    ix = fields.index("x")
    iy = fields.index("y")
    iz = fields.index("z")

    body = rest.decode("utf-8", errors="replace")
    out = np.empty((n_points, 3), dtype=np.float32)
    written = 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        out[written, 0] = float(toks[ix])
        out[written, 1] = float(toks[iy])
        out[written, 2] = float(toks[iz])
        written += 1
        if written == n_points:
            break

    return out[:written]
