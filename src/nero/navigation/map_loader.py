"""Map loader for static occupancy grid maps.

Supports loading maps from:
- PNG/YAML (ROS-style occupancy grid)
- NumPy .npy files
- Point cloud (.ply, .pcd) converted to 2D grid
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OccupancyGrid:
    """2D occupancy grid map."""

    data: np.ndarray  # 2D array: 0=free, 100=occupied, -1=unknown
    resolution: float  # meters per pixel
    origin: tuple[float, float]  # (x, y) of pixel (0, 0) in world coords
    width: int  # pixels
    height: int  # pixels

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        # Flip y axis (image coords)
        py = self.height - 1 - py
        return (px, py)

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        # Flip y axis
        py = self.height - 1 - py
        x = px * self.resolution + self.origin[0]
        y = py * self.resolution + self.origin[1]
        return (x, y)

    def is_occupied(self, x: float, y: float, radius: float = 0.0) -> bool:
        """Check if a world position is unsafe to traverse.

        Unknown cells are deliberately treated as occupied. A fixed-map robot
        must fail closed instead of planning through space the map has not
        observed.
        """
        px, py = self.world_to_pixel(x, y)
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True  # Out of bounds = occupied

        if radius > 0:
            # Check area around point
            # Round outward so floating-point truncation can never make the
            # realized clearance smaller than the configured robot radius.
            r_px = int(np.ceil(radius / self.resolution))
            y_min = max(0, py - r_px)
            y_max = min(self.height, py + r_px + 1)
            x_min = max(0, px - r_px)
            x_max = min(self.width, px + r_px + 1)
            region = self.data[y_min:y_max, x_min:x_max]
            return np.any(region != 0)

        return self.data[py, px] != 0

    def get_cost(self, x: float, y: float) -> float:
        """Get traversal cost at a world position."""
        px, py = self.world_to_pixel(x, y)
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return float("inf")
        val = self.data[py, px]
        if val == -1:
            return 50  # Unknown = medium cost
        return float(val)


def load_occupancy_grid(
    map_path: str | Path,
    yaml_path: Optional[str | Path] = None,
    resolution: float = 0.05,
    origin: tuple[float, float] = (0.0, 0.0),
    threshold: int = 65,
) -> OccupancyGrid:
    """Load an occupancy grid from file.

    Args:
        map_path: Path to PNG image or .npy file
        yaml_path: Optional path to YAML metadata (for ROS-style maps)
        resolution: Meters per pixel (used if no YAML)
        origin: World origin of pixel (0,0) (used if no YAML)
        threshold: Grayscale threshold for occupied (0-255)

    Returns:
        OccupancyGrid object
    """
    map_path = Path(map_path)
    ros_thresholds = False

    # Load YAML metadata if provided
    if yaml_path is not None:
        import yaml

        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        resolution = meta.get("resolution", resolution)
        origin_values = meta.get("origin", list(origin))
        origin = (float(origin_values[0]), float(origin_values[1]))
        occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        free_thresh = meta.get("free_thresh", 0.25)
        negate = meta.get("negate", 0)
        ros_thresholds = True

    if map_path.suffix == ".npy":
        data = np.load(map_path)
        return OccupancyGrid(
            data=data,
            resolution=resolution,
            origin=origin,
            width=data.shape[1],
            height=data.shape[0],
        )

    # Load PNG image
    img = Image.open(map_path).convert("L")  # Grayscale
    gray = np.array(img)

    # Convert to occupancy values
    data = np.full(gray.shape, -1, dtype=np.int8)  # Default unknown

    if ros_thresholds:
        occupancy = gray.astype(np.float32) / 255.0
        if not negate:
            occupancy = 1.0 - occupancy
        free_mask = occupancy < free_thresh
        occ_mask = occupancy > occupied_thresh
    elif negate:
        # Dark = free, light = occupied
        free_mask = gray < (threshold - free_thresh * 255)
        occ_mask = gray > threshold
    else:
        # Light = free, dark = occupied
        free_mask = gray > (255 - threshold + free_thresh * 255)
        occ_mask = gray < (255 - threshold)

    data[free_mask] = 0  # Free
    data[occ_mask] = 100  # Occupied

    return OccupancyGrid(
        data=data,
        resolution=resolution,
        origin=origin,
        width=data.shape[1],
        height=data.shape[0],
    )


def pointcloud_to_grid(
    points: np.ndarray,  # (N, 3) array
    resolution: float = 0.05,
    grid_size: float = 20.0,  # meters
    origin: Optional[tuple[float, float]] = None,
    height_threshold: float = 0.5,  # meters above ground
    max_height: float | None = None,
    up_axis: str = "z",
) -> OccupancyGrid:
    """Convert a 3D point cloud to a 2D occupancy grid.

    Args:
        points: (N, 3) array of (x, y, z) points
        resolution: Meters per pixel
        grid_size: Size of grid in meters (square)
        origin: World origin, defaults to min of point cloud
        height_threshold: Consider points above this as obstacles
        max_height: Ignore points above this height (for example ceilings)
        up_axis: Vertical point-cloud axis, or ``auto`` to infer the shortest span

    Returns:
        OccupancyGrid object
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("points must be a non-empty (N, 3) array")
    if resolution <= 0 or grid_size <= 0:
        raise ValueError("resolution and grid_size must be positive")
    if up_axis == "auto":
        robust_spans = np.quantile(points[:, :3], 0.99, axis=0) - np.quantile(
            points[:, :3], 0.01, axis=0
        )
        up_index = int(np.argmin(robust_spans))
    else:
        try:
            up_index = {"x": 0, "y": 1, "z": 2}[up_axis]
        except KeyError as exc:
            raise ValueError("up_axis must be x, y, z, or auto") from exc
    plane_indices = [index for index in range(3) if index != up_index]
    plane = points[:, plane_indices]
    height = points[:, up_index]

    if origin is None:
        origin = (float(plane[:, 0].min()), float(plane[:, 1].min()))

    grid_pixels = int(grid_size / resolution)
    data = np.zeros((grid_pixels, grid_pixels), dtype=np.int8)

    selected = height >= height_threshold
    if max_height is not None:
        selected &= height <= max_height
    pixels = np.floor(
        (plane[selected] - np.asarray(origin, dtype=float)) / resolution
    ).astype(np.int64)
    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < grid_pixels)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < grid_pixels)
    )
    pixels = pixels[in_bounds]
    data[grid_pixels - 1 - pixels[:, 1], pixels[:, 0]] = 100

    return OccupancyGrid(
        data=data,
        resolution=resolution,
        origin=origin,
        width=grid_pixels,
        height=grid_pixels,
    )


def save_grid_as_png(grid: OccupancyGrid, path: str | Path) -> None:
    """Save an occupancy grid as a grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to image: 0=white (free), 100=black (occupied), -1=gray (unknown)
    img = np.full((grid.height, grid.width), 205, dtype=np.uint8)  # Gray for unknown
    img[grid.data == 0] = 254  # White for free
    img[grid.data == 100] = 0  # Black for occupied

    Image.fromarray(img).save(path)
    logger.info(f"Saved occupancy grid to {path}")


def save_grid_yaml(grid: OccupancyGrid, path: str | Path) -> None:
    """Save grid metadata as YAML (ROS-style)."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "image": path.with_suffix(".png").name,
        "resolution": grid.resolution,
        "origin": [grid.origin[0], grid.origin[1], 0.0],
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "negate": 0,
    }
    with open(path, "w") as f:
        yaml.dump(meta, f)
    logger.info(f"Saved grid metadata to {path}")
