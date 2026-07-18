"""CLI tool to convert point clouds to occupancy grid maps.

Supports:
- PLY files
- PCD files
- NumPy .npy files
- LAS/LAZ files (with pylas)

Usage:
    uv run nero-pc2map scan.ply -o maps/office --resolution 0.05
    uv run nero-pc2map scan.pcd -o maps/office --height-thresh 0.3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nero.navigation import (
    pointcloud_to_grid,
    save_grid_as_png,
    save_grid_yaml,
)
from nero.navigation.pointcloud_io import load_pointcloud

logger = logging.getLogger(__name__)

_BUNDLED_MAIN_ROOM = (
    Path(__file__).resolve().parents[1] / "simulation/scenes/main_room/assets/main_room.ply"
)


def resolve_pointcloud_path(value: str | Path) -> Path:
    """Resolve the documented main-room shorthand without copying its LFS asset."""
    path = Path(value)
    if path.exists():
        return path
    if str(path) in ("main_room", "assets/main_room.ply"):
        return _BUNDLED_MAIN_ROOM
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid")
    parser.add_argument("input", help="Input point cloud file (.ply, .pcd, .npy, .las)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for map files")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution (m/px)")
    parser.add_argument("--grid-size", type=float, default=20.0, help="Grid size in meters")
    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z", "auto"),
        default="auto",
        help="Vertical axis; main_room.ply is Y-up",
    )
    parser.add_argument(
        "--height-thresh",
        type=float,
        default=0.5,
        help="Height threshold to filter ground points (m)",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Ignore points above this height (useful for ceilings)",
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
    input_path = resolve_pointcloud_path(args.input)
    logger.info(f"Loading point cloud: {input_path}")
    points = load_pointcloud(input_path)
    logger.info(f"Loaded {len(points)} points")

    # Convert to occupancy grid
    logger.info("Converting to occupancy grid...")
    grid = pointcloud_to_grid(
        points,
        resolution=args.resolution,
        grid_size=args.grid_size,
        height_threshold=args.height_thresh,
        max_height=args.max_height,
        up_axis=args.up_axis,
    )
    logger.info(f"Grid: {grid.width}x{grid.height} at {grid.resolution}m/px")

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
