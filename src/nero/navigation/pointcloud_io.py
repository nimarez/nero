"""Point-cloud readers shared by map loading and conversion commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_pointcloud(path: str | Path) -> np.ndarray:
    """Load XYZ points, including binary Gaussian-splat PLY files."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".ply":
        return _load_ply(path)
    if path.suffix.lower() == ".pcd":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("PCD support requires: uv add open3d") from exc
        return np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if path.suffix.lower() in (".las", ".laz"):
        try:
            import pylas
        except ImportError as exc:
            raise ImportError("LAS/LAZ support requires: uv add pylas") from exc
        cloud = pylas.read(str(path))
        return np.column_stack([cloud.x, cloud.y, cloud.z])
    raise ValueError(f"Unsupported point cloud format: {path.suffix}")


def _load_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        first = stream.readline()
        if first.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(
                f"{path} is a Git LFS pointer; run 'git lfs install' and 'git lfs pull'"
            )
        if first.strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        encoding = ""
        vertex_count = 0
        vertex_properties: list[tuple[str, str]] = []
        current_element = ""
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii").strip()
            fields = line.split()
            if fields[:1] == ["format"]:
                encoding = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                current_element = "vertex"
                vertex_count = int(fields[2])
            elif fields[:1] == ["element"]:
                current_element = fields[1]
            elif fields[:1] == ["property"] and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List-valued PLY vertex properties are unsupported")
                vertex_properties.append((fields[2], fields[1]))
            elif line == "end_header":
                data_offset = stream.tell()
                break

    names = {name for name, _ in vertex_properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY has no XYZ vertex properties: {path}")
    if encoding == "ascii":
        property_names = [name for name, _ in vertex_properties]
        indices = [property_names.index(axis) for axis in ("x", "y", "z")]
        values = np.loadtxt(path, skiprows=_header_line_count(path), max_rows=vertex_count)
        return np.asarray(np.atleast_2d(values)[:, indices], dtype=float)
    if encoding not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Unsupported PLY encoding {encoding!r}")
    endian = "<" if encoding == "binary_little_endian" else ">"
    try:
        dtype = np.dtype([(name, endian + _PLY_DTYPES[kind]) for name, kind in vertex_properties])
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY property type: {exc.args[0]}") from exc
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,))
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]])


def _header_line_count(path: Path) -> int:
    with path.open("rb") as stream:
        for index, line in enumerate(stream, start=1):
            if line.strip() == b"end_header":
                return index
    raise ValueError(f"Incomplete PLY header: {path}")
