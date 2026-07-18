"""Mock robot for simulation testing.

Provides a drop-in replacement for RobotInterface that simulates
robot movement in a 2D plane without requiring physical hardware.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MockRobotState:
    """State of the simulated robot."""

    x: float = 0.0  # meters
    y: float = 0.0  # meters
    yaw: float = 0.0  # radians
    vx: float = 0.0  # m/s
    vy: float = 0.0  # m/s
    vyaw: float = 0.0  # rad/s
    mode: str = "kPrepare"
    battery: float = 100.0
    timestamp: float = 0.0


class MockRobot:
    """Simulated K1 robot for testing without physical hardware.

    This class provides the same interface as RobotInterface but
    simulates robot movement in a 2D plane. It tracks position,
    velocity, and can be used with the navigation policy.
    """

    def __init__(
        self,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_yaw: float = 0.0,
        update_rate: float = 30.0,
    ):
        self.state = MockRobotState(
            x=initial_x,
            y=initial_y,
            yaw=initial_yaw,
            timestamp=time.time(),
        )
        self.update_rate = update_rate
        self._running = False
        self._last_update = time.time()
        self._obstacles: list[tuple[float, float]] = []

    def initialize(self) -> None:
        """Initialize the mock robot."""
        self._running = True
        self.state.mode = "kWalking"
        logger.info("Mock robot initialized in walk mode")

    def stop(self) -> None:
        """Stop the mock robot."""
        self._running = False
        self.state.vx = 0.0
        self.state.vy = 0.0
        self.state.vyaw = 0.0
        logger.info("Mock robot stopped")

    def set_velocity(
        self,
        vx: float,
        vy: float = 0.0,
        vyaw: float = 0.0,
    ) -> None:
        """Set robot velocity.

        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            vyaw: Yaw velocity (rad/s)
        """
        # Clamp velocities to K1 limits
        max_vx = 0.3
        max_vy = 0.2
        max_vyaw = 1.0

        self.state.vx = np.clip(vx, -max_vx, max_vx)
        self.state.vy = np.clip(vy, -max_vy, max_vy)
        self.state.vyaw = np.clip(vyaw, -max_vyaw, max_vyaw)

    def get_pose(self) -> np.ndarray:
        """Get current robot pose.

        Returns:
            numpy array [x, y, yaw]
        """
        self._update_state()
        return np.array([self.state.x, self.state.y, self.state.yaw])

    def get_state(self) -> MockRobotState:
        """Get full robot state."""
        self._update_state()
        return self.state

    def set_mode(self, mode: str) -> None:
        """Set robot mode."""
        self.state.mode = mode

    def add_obstacle(self, x: float, y: float) -> None:
        """Add an obstacle to the simulation.

        Args:
            x: Obstacle x position
            y: Obstacle y position
        """
        self._obstacles.append((x, y))

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self._obstacles.clear()

    def get_obstacles(self) -> list[tuple[float, float]]:
        """Get list of obstacles."""
        return self._obstacles.copy()

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Reset robot to initial position."""
        self.state.x = x
        self.state.y = y
        self.state.yaw = yaw
        self.state.vx = 0.0
        self.state.vy = 0.0
        self.state.vyaw = 0.0

    def _update_state(self) -> None:
        """Update robot state based on current velocity."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        if not self._running:
            return

        # Update position based on velocity
        # Transform velocity from robot frame to world frame
        cos_yaw = math.cos(self.state.yaw)
        sin_yaw = math.sin(self.state.yaw)

        world_vx = self.state.vx * cos_yaw - self.state.vy * sin_yaw
        world_vy = self.state.vx * sin_yaw + self.state.vy * cos_yaw

        self.state.x += world_vx * dt
        self.state.y += world_vy * dt
        self.state.yaw += self.state.vyaw * dt

        # Normalize yaw
        self.state.yaw = self._normalize_angle(self.state.yaw)
        self.state.timestamp = now

        # Check for collisions with obstacles
        self._check_collisions()

    def _check_collisions(self) -> None:
        """Check if robot has collided with any obstacles."""
        for obs_x, obs_y in self._obstacles:
            dist = math.sqrt((self.state.x - obs_x) ** 2 + (self.state.y - obs_y) ** 2)
            if dist < 0.3:  # Robot radius + obstacle radius
                # Push robot away from obstacle
                angle = math.atan2(self.state.y - obs_y, self.state.x - obs_x)
                self.state.x = obs_x + 0.3 * math.cos(angle)
                self.state.y = obs_y + 0.3 * math.sin(angle)
                self.state.vx *= 0.5  # Slow down on collision
                self.state.vy *= 0.5

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
