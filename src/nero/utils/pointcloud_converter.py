"""CLI tool to convert point clouds to occupancy grid maps.

Supports:
- PLY files
- PCD files
- NumPy .npy files
- LAS/LAZ files (with pylas)

Usage:
    nero-pc2map scan.ply -o maps/office --resolution 0.05
    nero-pc2map scan.pcd -o maps/office --height-thresh 0.3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from nero.navigation import (
    pointcloud_to_grid,
    save_grid_as_png,
    save_grid_yaml,
)

logger = logging.getLogger(__name__)


def load_pointcloud(path: str | Path) -> np.ndarray:
    """Load point cloud from file.

    Returns:
        (N, 3) array of (x, y, z) points
    """
    path = Path(path)

    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".ply":
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points)
        except ImportError:
            # Fallback: simple PLY parser
            return _parse_ply(path)

    if path.suffix == ".pcd":
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points)
        except ImportError:
            raise ImportError("Install open3d for PCD support: pip install open3d")

    if path.suffix in (".las", ".laz"):
        try:
            import pylas
            las = pylas.read(str(path))
            return np.column_stack([las.x, las.y, las.z])
        except ImportError:
            raise ImportError("Install pylas for LAS/LAZ support: pip install pylas")

    raise ValueError(f"Unsupported point cloud format: {path.suffix}")


def _parse_ply(path: Path) -> np.ndarray:
    """Simple PLY parser for vertex data."""
    points = []
    in_header = True
    vertex_count = 0
    props = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                in_header = False
                continue
            if in_header:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property"):
                    props.append(line.split()[-1])
            else:
                values = line.split()
                if len(values) >= 3:
                    points.append([float(values[0]), float(values[1]), float(values[2])])

    if not points:
        raise ValueError(f"No points found in {path}")

    return np.array(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid")
    parser.add_argument("input", help="Input point cloud file (.ply, .pcd, .npy, .las)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for map files")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution (m/px)")
    parser.add_argument("--grid-size", type=float, default=20.0, help="Grid size in meters")
    parser.add_argument(
        "--height-thresh",
        type=float,
        default=0.5,
        help="Height threshold to filter ground points (m)",
    )
    parser.add_argument("--name", default="map", help="Output filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load point cloud
    logger.info(f"Loading point cloud: {args.input}")
    points = load_pointcloud(args.input)
    logger.info(f"Loaded {len(points)} points")

    # Convert to occupancy grid
    logger.info("Converting to occupancy grid...")
    grid = pointcloud_to_grid(
        points,
        resolution=args.resolution,
        grid_size=args.grid_size,
        height_threshold=args.height_thresh,
    )
    logger.info(
        f"Grid: {grid.width}x{grid.height} at {grid.resolution}m/px"
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{args.name}.png"
    yaml_path = output_dir / f"{args.name}.yaml"

    save_grid_as_png(grid, png_path)
    save_grid_yaml(grid, yaml_path)

    logger.info(f"Saved map to {png_path} and {yaml_path}")


if __name__ == "__main__":
    main()