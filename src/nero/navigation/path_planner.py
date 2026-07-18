"""A* path planner for occupancy grid navigation."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nero.navigation.map_loader import OccupancyGrid

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """Node in the A* search tree."""
    x: int
    y: int
    g: float  # Cost from start
    h: float  # Heuristic to goal
    parent: Optional['PathNode'] = None

    @property
    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


@dataclass
class Path:
    """Planned path from start to goal."""
    waypoints: list[tuple[float, float]]  # World coordinates
    pixels: list[tuple[int, int]]  # Pixel coordinates
    cost: float

    def __len__(self) -> int:
        return len(self.waypoints)


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_neighbors(
    px: int, py: int, grid: OccupancyGrid, allow_diagonal: bool = True
) -> list[tuple[int, int, float]]:
    """Get valid neighboring pixels with movement costs."""
    neighbors = []
    moves = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),  # Cardinal
    ]
    if allow_diagonal:
        moves.extend([
            (1, 1, 1.414), (-1, 1, 1.414),
            (1, -1, 1.414), (-1, -1, 1.414),
        ])

    for dx, dy, cost in moves:
        nx, ny = px + dx, py + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            # Flip y for grid access
            ny_grid = grid.height - 1 - ny
            if grid.data[ny_grid, nx] != 100:  # Not occupied
                neighbors.append((nx, ny, cost))

    return neighbors


def astar(
    grid: OccupancyGrid,
    start_world: tuple[float, float],
    goal_world: tuple[float, float],
    allow_diagonal: bool = True,
    inflation_radius: float = 0.0,
) -> Optional[Path]:
    """Plan a path using A* algorithm.

    Args:
        grid: Occupancy grid map
        start_world: Start position in world coordinates
        goal_world: Goal position in world coordinates
        allow_diagonal: Allow diagonal movement
        inflation_radius: Inflate obstacles by this radius (meters)

    Returns:
        Path object or None if no path found
    """
    # Convert to pixel coordinates
    start_px, start_py = grid.world_to_pixel(*start_world)
    goal_px, goal_py = grid.world_to_pixel(*goal_world)

    # Validate start/goal
    if grid.is_occupied(*start_world, radius=inflation_radius):
        logger.warning(f"Start position {start_world} is occupied")
        return None
    if grid.is_occupied(*goal_world, radius=inflation_radius):
        logger.warning(f"Goal position {goal_world} is occupied")
        return None

    # A* search
    start_node = PathNode(start_px, start_py, 0.0, heuristic((start_px, start_py), (goal_px, goal_py)))
    open_set: list[PathNode] = [start_node]
    closed_set: set[tuple[int, int]] = set()
    g_scores: dict[tuple[int, int], float] = {(start_px, start_py): 0.0}

    while open_set:
        current = heapq.heappop(open_set)

        # Check if goal reached
        if (current.x, current.y) == (goal_px, goal_py):
            # Reconstruct path
            path_pixels = []
            node = current
            while node is not None:
                path_pixels.append((node.x, node.y))
                node = node.parent
            path_pixels.reverse()

            # Convert to world coordinates
            waypoints = [grid.pixel_to_world(px, py) for px, py in path_pixels]

            return Path(
                waypoints=waypoints,
                pixels=path_pixels,
                cost=current.g,
            )

        if (current.x, current.y) in closed_set:
            continue
        closed_set.add((current.x, current.y))

        # Explore neighbors
        for nx, ny, move_cost in get_neighbors(current.x, current.y, grid, allow_diagonal):
            if (nx, ny) in closed_set:
                continue

            # Check inflation radius
            ny_world = grid.pixel_to_world(nx, ny)[1]
            nx_world = grid.pixel_to_world(nx, ny)[0]
            if inflation_radius > 0 and grid.is_occupied(nx_world, ny_world, radius=inflation_radius):
                continue

            new_g = current.g + move_cost * grid.resolution

            if new_g < g_scores.get((nx, ny), float('inf')):
                g_scores[(nx, ny)] = new_g
                h = heuristic((nx, ny), (goal_px, goal_py))
                neighbor = PathNode(nx, ny, new_g, h, parent=current)
                heapq.heappush(open_set, neighbor)

    logger.warning(f"No path found from {start_world} to {goal_world}")
    return None


def smooth_path(
    path: Path,
    grid: OccupancyGrid,
    max_iterations: int = 100,
) -> Path:
    """Smooth a path by removing unnecessary waypoints.

    Uses line-of-sight checks to skip intermediate waypoints.
    """
    if len(path.waypoints) <= 2:
        return path

    smoothed = [path.waypoints[0]]
    current_idx = 0

    for _ in range(max_iterations):
        if current_idx >= len(path.waypoints) - 1:
            break

        # Try to skip ahead as far as possible
        for next_idx in range(len(path.waypoints) - 1, current_idx, -1):
            if line_of_sight(
                grid,
                path.waypoints[current_idx],
                path.waypoints[next_idx],
            ):
                smoothed.append(path.waypoints[next_idx])
                current_idx = next_idx
                break

    return Path(
        waypoints=smoothed,
        pixels=[grid.world_to_pixel(x, y) for x, y in smoothed],
        cost=path.cost,
    )


def line_of_sight(
    grid: OccupancyGrid,
    start: tuple[float, float],
    end: tuple[float, float],
    step_size: float = 0.05,
) -> bool:
    """Check if there's a clear line of sight between two points."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist < 0.01:
        return True

    steps = int(dist / step_size)
    for i in range(steps + 1):
        t = i / max(steps, 1)
        x = start[0] + dx * t
        y = start[1] + dy * t
        if grid.is_occupied(x, y):
            return False

    return True


def follow_path(
    path: Path,
    current_world: tuple[float, float],
    lookahead_distance: float = 0.5,
) -> tuple[float, float]:
    """Get the next waypoint to follow from a path.

    Returns the lookahead point on the path.
    """
    if not path.waypoints:
        return current_world

    # Find closest point on path
    min_dist = float('inf')
    closest_idx = 0

    for i, wp in enumerate(path.waypoints):
        dist = np.sqrt(
            (wp[0] - current_world[0]) ** 2 + (wp[1] - current_world[1]) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Find lookahead point
    lookahead_idx = closest_idx
    for i in range(closest_idx, len(path.waypoints)):
        dist = np.sqrt(
            (path.waypoints[i][0] - current_world[0]) ** 2 +
            (path.waypoints[i][1] - current_world[1]) ** 2
        )
        if dist >= lookahead_distance:
            lookahead_idx = i
            break

    return path.waypoints[lookahead_idx]
