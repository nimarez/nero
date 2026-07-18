"""Simulated camera for testing without physical hardware.

Generates synthetic camera frames showing a 2D top-down view
of the simulated environment with the robot and objects.
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    """Camera rendering mode."""

    TOP_DOWN = "top_down"  # Top-down 2D view
    FORWARD = "forward"  # Forward-facing perspective
    DEPTH = "depth"  # Depth visualization


class SimCamera:
    """Simulated camera that generates synthetic frames.

    Creates visual frames showing the robot's environment,
    including objects, obstacles, and the robot itself.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mode: CameraMode = CameraMode.TOP_DOWN,
        fov: float = 90.0,  # Field of view in degrees
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode
        self.fov = fov
        self._running = False
        self._last_frame_time = 0.0

        # Environment elements
        self._objects: dict[str, tuple[float, float]] = {}  # name -> (x, y)
        self._obstacles: list[tuple[float, float]] = []
        self._grid_size = 10.0  # meters

    def start(self) -> bool:
        """Start the simulated camera."""
        self._running = True
        self._last_frame_time = time.time()
        logger.info(
            f"Simulated camera started ({self.width}x{self.height} @ {self.fps}fps)"
        )
        return True

    def stop(self) -> None:
        """Stop the simulated camera."""
        self._running = False
        logger.info("Simulated camera stopped")

    def get_frame(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Generate a synthetic camera frame.

        Args:
            robot_x: Robot x position in world frame
            robot_y: Robot y position in world frame
            robot_yaw: Robot yaw in radians

        Returns:
            BGR image frame, or None if not running
        """
        if not self._running:
            return None

        if self.mode == CameraMode.TOP_DOWN:
            return self._render_top_down(robot_x, robot_y, robot_yaw)
        elif self.mode == CameraMode.FORWARD:
            return self._render_forward(robot_x, robot_y, robot_yaw)
        elif self.mode == CameraMode.DEPTH:
            return self._render_depth(robot_x, robot_y, robot_yaw)
        else:
            return self._render_top_down(robot_x, robot_y, robot_yaw)

    def get_depth_frame(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Generate a synthetic depth frame.

        Args:
            robot_x: Robot x position in world frame
            robot_y: Robot y position in world frame
            robot_yaw: Robot yaw in radians

        Returns:
            Depth image (H, W) float32 in meters, or None if not running
        """
        if not self._running:
            return None

        # Create depth array
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        max_depth = 10.0

        center_x = self.width // 2
        center_y = self.height // 2
        scale = self.width / self._grid_size

        # Fill with base depth (distance from center)
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                d = dist / scale
                depth[y, x] = min(d, max_depth)

        # Add objects as closer depth values
        cos_yaw = math.cos(-robot_yaw)
        sin_yaw = math.sin(-robot_yaw)

        for name, (obj_x, obj_y) in self._objects.items():
            # Transform to robot frame
            dx = obj_x - robot_x
            dy = obj_y - robot_y
            local_x = dx * cos_yaw - dy * sin_yaw
            local_y = dx * sin_yaw + dy * cos_yaw

            # Distance to object
            obj_dist = math.sqrt(local_x**2 + local_y**2)

            # Project to pixel coordinates
            px = int(center_x + local_x * scale)
            py = int(center_y - local_y * scale)

            # Draw object as a region with the object's depth
            obj_size = 15
            for dy in range(max(0, py - obj_size), min(self.height, py + obj_size)):
                for dx in range(max(0, px - obj_size), min(self.width, px + obj_size)):
                    depth[dy, dx] = obj_dist

        return depth

    def get_fps(self) -> float:
        """Get camera FPS."""
        return float(self.fps)

    def add_object(self, name: str, x: float, y: float) -> None:
        """Add an object to the simulated environment.

        Args:
            name: Object name (for detection)
            x: Object x position in world frame
            y: Object y position in world frame
        """
        self._objects[name] = (x, y)

    def add_obstacle(self, x: float, y: float) -> None:
        """Add an obstacle to the environment.

        Args:
            x: Obstacle x position
            y: Obstacle y position
        """
        self._obstacles.append((x, y))

    def get_objects(self) -> dict[str, tuple[float, float]]:
        """Get a copy of the objects in the synthetic scene."""
        return self._objects.copy()

    def clear_objects(self) -> None:
        """Clear all objects."""
        self._objects.clear()

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self._obstacles.clear()

    def _render_top_down(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """Render top-down view of the environment."""
        # Create blank frame
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Scale: map world meters to pixels
        scale = self.width / self._grid_size
        center_x = self.width // 2
        center_y = self.height // 2

        def world_to_pixel(wx: float, wy: float) -> tuple[int, int]:
            # Transform to robot-centric frame
            cos_yaw = math.cos(-robot_yaw - math.pi / 2)
            sin_yaw = math.sin(-robot_yaw - math.pi / 2)
            dx = wx - robot_x
            dy = wy - robot_y
            px = dx * cos_yaw - dy * sin_yaw
            py = dx * sin_yaw + dy * cos_yaw
            return (
                int(center_x + px * scale),
                int(center_y - py * scale),  # Flip y for image coords
            )

        # Draw grid
        grid_step = 1.0  # 1 meter grid
        for i in range(-int(self._grid_size), int(self._grid_size) + 1):
            if i % grid_step == 0:
                # Vertical lines
                p1 = world_to_pixel(i, -self._grid_size)
                p2 = world_to_pixel(i, self._grid_size)
                cv2.line(frame, p1, p2, (200, 200, 200), 1)
                # Horizontal lines
                p1 = world_to_pixel(-self._grid_size, i)
                p2 = world_to_pixel(self._grid_size, i)
                cv2.line(frame, p1, p2, (200, 200, 200), 1)

        # Draw obstacles (red circles)
        for obs_x, obs_y in self._obstacles:
            px, py = world_to_pixel(obs_x, obs_y)
            if 0 <= px < self.width and 0 <= py < self.height:
                cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)

        # Draw objects (colored rectangles with labels)
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        for idx, (name, (obj_x, obj_y)) in enumerate(self._objects.items()):
            px, py = world_to_pixel(obj_x, obj_y)
            if 0 <= px < self.width and 0 <= py < self.height:
                color = colors[idx % len(colors)]
                cv2.rectangle(frame, (px - 15, py - 15), (px + 15, py + 15), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (px - 20, py - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Draw robot (triangle at center)
        robot_size = 15
        cos = math.cos(robot_yaw)
        sin = math.sin(robot_yaw)
        # Forward direction
        tip = (center_x, center_y - robot_size)
        # Left corner
        left = (
            int(center_x - robot_size * 0.7 * cos - robot_size * 0.7 * sin),
            int(center_y + robot_size * 0.7 * sin - robot_size * 0.7 * cos),
        )
        # Right corner
        right = (
            int(center_x + robot_size * 0.7 * cos - robot_size * 0.7 * sin),
            int(center_y - robot_size * 0.7 * sin - robot_size * 0.7 * cos),
        )
        cv2.drawContours(frame, [np.array([tip, left, right])], 0, (0, 0, 0), 2)

        # Add info text
        cv2.putText(
            frame,
            f"Robot: ({robot_x:.2f}, {robot_y:.2f}, {math.degrees(robot_yaw):.1f}°)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        return frame

    def _render_forward(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """Render forward-facing perspective view."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128

        # Simple perspective projection
        focal_length = self.width / (2 * math.tan(math.radians(self.fov / 2)))

        # Draw horizon
        horizon_y = self.height // 2
        cv2.line(frame, (0, horizon_y), (self.width, horizon_y), (100, 150, 100), 2)

        # Draw ground plane grid
        for depth in range(1, 10):
            y = int(horizon_y + (self.height / 2) * (1 / depth))
            if y < self.height:
                cv2.line(frame, (0, y), (self.width, y), (80, 120, 80), 1)

        # Project objects onto the frame
        for name, (obj_x, obj_y) in self._objects.items():
            # Transform to robot frame
            dx = obj_x - robot_x
            dy = obj_y - robot_y
            cos = math.cos(-robot_yaw)
            sin = math.sin(-robot_yaw)
            local_x = dx * cos - dy * sin
            local_y = dx * sin + dy * cos

            # Only render objects in front of robot
            if local_y > 0:
                # Project to image coordinates
                img_x = int(self.width / 2 + focal_length * local_x / local_y)
                size = int(100 / local_y)  # Size inversely proportional to distance
                size = max(10, min(size, 100))

                if 0 <= img_x < self.width:
                    cv2.rectangle(
                        frame,
                        (img_x - size // 2, horizon_y - size),
                        (img_x + size // 2, horizon_y),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        name,
                        (img_x - 30, horizon_y - size - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        # Add info
        cv2.putText(
            frame,
            "Forward View (Simulated)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame

    def _render_depth(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """Render depth visualization."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        # Simple depth: distance from robot
        center_x = self.width // 2
        center_y = self.height // 2
        max_depth = 10.0

        for y in range(self.height):
            for x in range(self.width):
                # Convert pixel to world coordinates (simplified)
                depth = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                depth = depth / (self.width / 2) * max_depth
                depth = min(depth, max_depth)
                # Map to grayscale (closer = brighter)
                frame[y, x] = int(255 * (1 - depth / max_depth))

        # Convert to BGR for display
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
