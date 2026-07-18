"""Map-based navigation policy using static occupancy grid.

This policy navigates to a goal using:
- Pre-built occupancy grid map (no SLAM)
- Visual odometry for localization
- A* path planning
- Velocity control for execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from nero.navigation.map_loader import OccupancyGrid
from nero.navigation.path_planner import Path, astar, follow_path, smooth_path
from nero.navigation.visual_odometry import Pose2D, VisualOdometry

logger = logging.getLogger(__name__)


class MapNavState(Enum):
    """States for map-based navigation."""
    IDLE = "idle"
    LOCALIZING = "localizing"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    LOST = "lost"


@dataclass
class MapNavConfig:
    """Configuration for map-based navigation."""
    # Map settings
    map_path: str = ""
    yaml_path: Optional[str] = None
    resolution: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)

    # Odometry settings
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    use_depth: bool = True

    # Planning settings
    inflation_radius: float = 0.3  # meters
    path_smoothing: bool = True
    lookahead_distance: float = 0.5  # meters

    # Control settings
    max_linear_vel: float = 0.3  # m/s
    max_angular_vel: float = 0.5  # rad/s
    goal_threshold: float = 0.3  # meters
    obstacle_threshold: float = 0.5  # meters


class MapNavigationPolicy:
    """Navigation policy using static map + visual odometry."""

    def __init__(self, config: Optional[MapNavConfig] = None):
        self._config = config or MapNavConfig()
        self._state = MapNavState.IDLE

        # Components
        self._grid: Optional[OccupancyGrid] = None
        self._vo: Optional[VisualOdometry] = None
        self._current_path: Optional[Path] = None

        # State
        self._goal_world: Optional[tuple[float, float]] = None
        self._current_pose = Pose2D()
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None

    def load_map(self) -> bool:
        """Load the occupancy grid map."""
        from nero.navigation.map_loader import load_occupancy_grid

        try:
            self._grid = load_occupancy_grid(
                map_path=self._config.map_path,
                yaml_path=self._config.yaml_path,
                resolution=self._config.resolution,
                origin=self._config.origin,
            )
            logger.info(
                f"Loaded map: {self._grid.width}x{self._grid.height} "
                f"at {self._grid.resolution}m/px"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load map: {e}")
            return False

    def init_odometry(self, frame: np.ndarray) -> None:
        """Initialize visual odometry with first frame."""
        self._vo = VisualOdometry(
            fx=self._config.fx,
            fy=self._config.fy,
            cx=self._config.cx,
            cy=self._config.cy,
        )
        self._vo.set_use_depth(self._config.use_depth)
        self._vo.initialize(frame)
        self._state = MapNavState.LOCALIZING
        logger.info("Visual odometry initialized")

    def set_goal(self, x: float, y: float) -> bool:
        """Set navigation goal in world coordinates."""
        if self._grid is None:
            logger.error("No map loaded")
            return False

        if self._grid.is_occupied(x, y, radius=self._config.inflation_radius):
            logger.warning(f"Goal ({x}, {y}) is occupied")
            return False

        self._goal_world = (x, y)
        self._state = MapNavState.PLANNING
        logger.info(f"Goal set to ({x}, {y})")
        return True

    def set_goal_from_pixel(self, px: int, py: int) -> bool:
        """Set navigation goal from pixel coordinates on the map."""
        if self._grid is None:
            logger.error("No map loaded")
            return False

        x, y = self._grid.pixel_to_world(px, py)
        return self.set_goal(x, y)

    def update(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float]:
        """Run policy loop, return velocity command.

        Returns:
            (vx, vy, vyaw) velocity command
        """
        if self._state == MapNavState.IDLE:
            return (0.0, 0.0, 0.0)

        if self._state == MapNavState.ARRIVED:
            return (0.0, 0.0, 0.0)

        # Update odometry
        if self._vo is not None:
            pose = self._vo.update(frame, depth)
            if pose is not None:
                self._current_pose = pose
            else:
                logger.warning("Odometry update failed")
                if self._state == MapNavState.LOCALIZING:
                    self._state = MapNavState.LOST
                    return (0.0, 0.0, 0.0)

        if self._state == MapNavState.LOCALIZING:
            self._state = MapNavState.PLANNING

        # Plan path if needed
        if self._state == MapNavState.PLANNING:
            if self._goal_world is None or self._grid is None:
                self._state = MapNavState.IDLE
                return (0.0, 0.0, 0.0)

            current_world = (self._current_pose.x, self._current_pose.y)
            path = astar(
                self._grid,
                current_world,
                self._goal_world,
                inflation_radius=self._config.inflation_radius,
            )

            if path is None:
                logger.error("No path found to goal")
                self._state = MapNavState.LOST
                return (0.0, 0.0, 0.0)

            if self._config.path_smoothing:
                path = smooth_path(path, self._grid)

            self._current_path = path
            self._state = MapNavState.NAVIGATING
            logger.info(f"Path planned: {len(path.waypoints)} waypoints")

        # Follow path
        if self._state == MapNavState.NAVIGATING and self._current_path is not None:
            current_world = (self._current_pose.x, self._current_pose.y)

            # Check if arrived
            dist_to_goal = np.sqrt(
                (current_world[0] - self._goal_world[0]) ** 2 +
                (current_world[1] - self._goal_world[1]) ** 2
            )
            if dist_to_goal < self._config.goal_threshold:
                self._state = MapNavState.ARRIVED
                logger.info("Arrived at goal")
                return (0.0, 0.0, 0.0)

            # Get next waypoint
            target = follow_path(
                self._current_path,
                current_world,
                self._config.lookahead_distance,
            )

            # Compute velocity command
            return self._compute_velocity(current_world, target)

        return (0.0, 0.0, 0.0)

    def _compute_velocity(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float, float]:
        """Compute velocity command to reach target."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.01:
            return (0.0, 0.0, 0.0)

        # Desired heading
        desired_theta = np.arctan2(dy, dx)

        # Heading error
        theta_error = desired_theta - self._current_pose.theta
        # Normalize to [-pi, pi]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        # Angular velocity (proportional)
        vyaw = np.clip(
            theta_error * 1.0,
            -self._config.max_angular_vel,
            self._config.max_angular_vel,
        )

        # Linear velocity (proportional to distance, reduced when turning)
        turn_factor = max(0, 1 - abs(theta_error) / (np.pi / 2))
        vx = np.clip(
            dist * 0.5 * turn_factor,
            0.0,
            self._config.max_linear_vel,
        )

        return (vx, 0.0, vyaw)

    def reset(self) -> None:
        """Reset policy to idle state."""
        self._state = MapNavState.IDLE
        self._goal_world = None
        self._current_path = None
        if self._vo is not None:
            self._vo.reset()
        self._current_pose = Pose2D()

    @property
    def state(self) -> MapNavState:
        return self._state

    @property
    def current_pose(self) -> Pose2D:
        return self._current_pose

    @property
    def grid(self) -> Optional[OccupancyGrid]:
        return self._grid

    def render_map(
        self,
        show_path: bool = True,
        show_pose: bool = True,
        show_goal: bool = True,
    ) -> np.ndarray:
        """Render the map with current state overlay."""
        if self._grid is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Create color image from grid
        img = np.zeros((self._grid.height, self._grid.width, 3), dtype=np.uint8)
        img[self._grid.data == 0] = [254, 254, 254]  # Free = white
        img[self._grid.data == 100] = [0, 0, 0]  # Occupied = black
        img[self._grid.data == -1] = [205, 205, 205]  # Unknown = gray

        # Draw path
        if show_path and self._current_path is not None:
            for i in range(len(self._current_path.pixels) - 1):
                px1, py1 = self._current_path.pixels[i]
                px2, py2 = self._current_path.pixels[i + 1]
                # Flip y for image coords
                py1_img = self._grid.height - 1 - py1
                py2_img = self._grid.height - 1 - py2
                cv2.line(img, (px1, py1_img), (px2, py2_img), (0, 255, 0), 2)

        # Draw current pose
        if show_pose:
            px, py = self._grid.world_to_pixel(
                self._current_pose.x, self._current_pose.y
            )
            py_img = self._grid.height - 1 - py
            cv2.circle(img, (px, py_img), 5, (0, 0, 255), -1)

            # Draw heading
            heading_len = 20
            hx = px + int(heading_len * np.cos(self._current_pose.theta))
            hy = py_img - int(heading_len * np.sin(self._current_pose.theta))
            cv2.line(img, (px, py_img), (hx, hy), (0, 0, 255), 2)

        # Draw goal
        if show_goal and self._goal_world is not None:
            gx, gy = self._grid.world_to_pixel(*self._goal_world)
            gy_img = self._grid.height - 1 - gy
            cv2.circle(img, (gx, gy_img), 8, (255, 0, 0), -1)

        return img