"""Read PCD point-cloud files.

Self-contained: handles ``DATA ascii``, ``DATA binary``, and
``DATA binary_compressed`` without any compiled extensions. ``open3d``
is preferred when available (faster on huge files); the in-house path
covers environments where open3d cannot be installed (e.g. Python 3.13).

Only XYZ is exposed; downstream code does not need intensity or RGB.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import IO

import numpy as np


_NUMPY_DTYPE_FOR = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
}


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    """Read a PCD file and return its XYZ points as ``(N, 3) float32``.

    Tries ``open3d`` first; if not installed, uses the in-house parser
    that supports ``DATA ascii`` and ``DATA binary``. Raises a clear
    error for ``DATA binary_compressed`` (LZF) when open3d is missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pcd = o3d.io.read_point_cloud(str(p))
        return np.asarray(pcd.points, dtype=np.float32)
    except ImportError:
        return _read_pcd_native(p)


def _read_pcd_native(path: Path) -> np.ndarray:
    """In-house parser for ``DATA ascii`` and ``DATA binary`` PCDs."""
    with path.open("rb") as f:
        header = _read_header(f, path)
        if header["data"] == "ascii":
            body = f.read().decode("utf-8", errors="replace")
            return _parse_ascii_body(body, header, path)
        if header["data"] == "binary":
            return _parse_binary_body(f, header, path)
        if header["data"] == "binary_compressed":
            return _parse_binary_compressed_body(f, header, path)
        raise ValueError(f"{path}: unknown DATA kind {header['data']!r}")


# Internal name kept for backward-compat with existing tests
def _read_pcd_ascii(path: Path) -> np.ndarray:
    return _read_pcd_native(path)


def _read_header(f: IO[bytes], path: Path) -> dict:
    header_bytes = b""
    while True:
        line = f.readline()
        if not line:
            raise ValueError(f"{path}: malformed PCD header (no DATA line)")
        header_bytes += line
        if line.lstrip().startswith(b"DATA"):
            break

    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    n_points = 0
    data_kind = "ascii"

    for raw_line in header_bytes.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        if key == "FIELDS":
            fields = vals
        elif key == "SIZE":
            sizes = [int(v) for v in vals]
        elif key == "TYPE":
            types = [v.upper() for v in vals]
        elif key == "COUNT":
            counts = [int(v) for v in vals]
        elif key == "POINTS":
            n_points = int(vals[0])
        elif key == "DATA":
            data_kind = vals[0].strip().lower()

    if not fields:
        raise ValueError(f"{path}: PCD header has no FIELDS line")
    if not sizes:
        sizes = [4] * len(fields)
    if not types:
        types = ["F"] * len(fields)
    if not counts:
        counts = [1] * len(fields)

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(
            f"{path}: FIELDS/SIZE/TYPE/COUNT length mismatch: "
            f"{fields} {sizes} {types} {counts}"
        )

    return {
        "fields": fields,
        "sizes": sizes,
        "types": types,
        "counts": counts,
        "n_points": n_points,
        "data": data_kind,
    }


def _xyz_indices(fields: list[str], path: Path) -> tuple[int, int, int]:
    if not all(c in fields for c in ("x", "y", "z")):
        raise ValueError(f"{path}: PCD missing x/y/z fields (FIELDS={fields})")
    return fields.index("x"), fields.index("y"), fields.index("z")


def _parse_ascii_body(body: str, header: dict, path: Path) -> np.ndarray:
    fields = header["fields"]
    n_points = header["n_points"]
    ix, iy, iz = _xyz_indices(fields, path)

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


def _parse_binary_body(
    f: IO[bytes], header: dict, path: Path
) -> np.ndarray:
    """Parse a binary PCD body (uncompressed, struct-of-arrays per point).

    For each point the bytes are: ``concat over fields of (SIZE * COUNT)
    bytes interpreted as the field's TYPE``.
    """
    fields = header["fields"]
    sizes = header["sizes"]
    types = header["types"]
    counts = header["counts"]
    n_points = header["n_points"]

    field_bytes = [s * c for s, c in zip(sizes, counts)]
    point_bytes = sum(field_bytes)
    rest = f.read(n_points * point_bytes)
    if len(rest) < n_points * point_bytes:
        raise ValueError(
            f"{path}: truncated binary PCD body "
            f"(expected {n_points * point_bytes} B, got {len(rest)} B)"
        )

    ix, iy, iz = _xyz_indices(fields, path)
    # Build offsets within a single point
    offsets = [0]
    for nb in field_bytes[:-1]:
        offsets.append(offsets[-1] + nb)

    out = np.empty((n_points, 3), dtype=np.float32)

    # Fast path for the very common case (FIELDS x y z, F4, COUNT 1, in order)
    fast_path = (
        fields[:3] == ["x", "y", "z"]
        and types[0] == "F" and sizes[0] == 4 and counts[0] == 1
        and types[1] == "F" and sizes[1] == 4 and counts[1] == 1
        and types[2] == "F" and sizes[2] == 4 and counts[2] == 1
    )
    if fast_path and point_bytes == 12:
        return np.frombuffer(rest, dtype=np.float32).reshape(n_points, 3).astype(
            np.float32, copy=True
        )

    # General path
    for axis_idx, fi in zip(range(3), (ix, iy, iz)):
        dtype = _NUMPY_DTYPE_FOR.get((types[fi], sizes[fi]))
        if dtype is None:
            raise ValueError(
                f"{path}: unsupported field {fields[fi]!r} TYPE={types[fi]} "
                f"SIZE={sizes[fi]}"
            )
        n_elem = counts[fi]
        if n_elem != 1:
            raise ValueError(
                f"{path}: COUNT>1 unsupported for x/y/z (got {n_elem})"
            )
        # Strided read of one column from the row-major byte block
        start = offsets[fi]
        col = np.frombuffer(rest, dtype=np.uint8)
        col = col.reshape(n_points, point_bytes)[:, start : start + sizes[fi]]
        col = np.ascontiguousarray(col).view(dtype).reshape(n_points)
        out[:, axis_idx] = col.astype(np.float32, copy=False)

    return out


# --------------------------------------------------------------------- #
# binary_compressed support (LZF)
# --------------------------------------------------------------------- #
def _parse_binary_compressed_body(
    f: IO[bytes], header: dict, path: Path
) -> np.ndarray:
    """Parse a ``DATA binary_compressed`` body using the in-house LZF decoder.

    Layout after the ``DATA binary_compressed\\n`` line:

        uint32 little-endian: compressed_size
        uint32 little-endian: uncompressed_size
        compressed_size bytes: LZF-compressed payload

    The decompressed payload is in **struct-of-arrays** form: all x's
    concatenated, then all y's, etc. (Different from ``DATA binary``,
    which is array-of-structs.)
    """
    fields = header["fields"]
    sizes = header["sizes"]
    types = header["types"]
    counts = header["counts"]
    n_points = header["n_points"]

    sz_hdr = f.read(8)
    if len(sz_hdr) < 8:
        raise ValueError(f"{path}: truncated LZF size header")
    compressed_size = struct.unpack("<I", sz_hdr[:4])[0]
    uncompressed_size = struct.unpack("<I", sz_hdr[4:])[0]

    payload = f.read(compressed_size)
    if len(payload) < compressed_size:
        raise ValueError(
            f"{path}: truncated LZF payload "
            f"(expected {compressed_size} B, got {len(payload)} B)"
        )

    raw = _lzf_decompress(payload, uncompressed_size)

    # SoA: field block sizes
    field_block_sizes = [s * c * n_points for s, c in zip(sizes, counts)]
    field_offsets = [0]
    for sz in field_block_sizes[:-1]:
        field_offsets.append(field_offsets[-1] + sz)

    ix, iy, iz = _xyz_indices(fields, path)

    out = np.empty((n_points, 3), dtype=np.float32)
    for axis_idx, fi in zip(range(3), (ix, iy, iz)):
        dtype = _NUMPY_DTYPE_FOR.get((types[fi], sizes[fi]))
        if dtype is None:
            raise ValueError(
                f"{path}: unsupported field {fields[fi]!r} TYPE={types[fi]} "
                f"SIZE={sizes[fi]}"
            )
        if counts[fi] != 1:
            raise ValueError(
                f"{path}: COUNT>1 unsupported for x/y/z (got {counts[fi]})"
            )
        offset = field_offsets[fi]
        block = np.frombuffer(
            raw, dtype=dtype, count=n_points, offset=offset
        )
        out[:, axis_idx] = block.astype(np.float32, copy=False)

    return out


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress an LZF stream.

    Pure-Python implementation of the liblzf stream format:

    - ``ctrl`` byte ``< 0x20``: literal run of ``ctrl + 1`` bytes follows
    - ``ctrl`` byte ``>= 0x20``: back-reference; high 3 bits of ctrl are
      the run length (with ``7`` meaning "extra length byte follows"),
      low 5 bits + the next byte are the back-distance minus 1.

    Back-references with ``run > distance`` overlap their own output and
    must be copied byte-by-byte (this is RLE-like extension).
    """
    out = bytearray(expected_size)
    op = 0
    ip = 0
    n = len(data)

    while ip < n:
        ctrl = data[ip]
        ip += 1

        if ctrl < 0x20:  # literal run
            run_len = ctrl + 1
            if ip + run_len > n:
                raise ValueError("LZF: truncated literal run")
            if op + run_len > expected_size:
                raise ValueError("LZF: literal run overflows output")
            out[op : op + run_len] = data[ip : ip + run_len]
            op += run_len
            ip += run_len
        else:  # back-reference
            run_len = ctrl >> 5
            if run_len == 7:
                if ip >= n:
                    raise ValueError("LZF: truncated extended length")
                run_len += data[ip]
                ip += 1
            run_len += 2

            if ip >= n:
                raise ValueError("LZF: truncated back-reference offset")
            ref_offset = ((ctrl & 0x1F) << 8) | data[ip]
            ip += 1
            ref_offset += 1

            ref_pos = op - ref_offset
            if ref_pos < 0:
                raise ValueError("LZF: back-reference before output start")
            if op + run_len > expected_size:
                raise ValueError("LZF: back-reference overflows output")

            # Byte-by-byte to handle overlap (run > offset: RLE-like)
            for i in range(run_len):
                out[op + i] = out[ref_pos + i]
            op += run_len

    if op != expected_size:
        raise ValueError(
            f"LZF: decoded {op} bytes, expected {expected_size}"
        )
    return bytes(out)
