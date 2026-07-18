"""Velocity controller for K1 robot navigation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VelocityCommand:
    """Velocity command to send to robot.

    The K1 robot requires 3 velocity components:
    - vx: forward/backward (m/s)
    - vy: lateral/sideways (m/s)
    - vyaw: yaw rotation (rad/s)
    """

    linear_x: float = 0.0  # m/s (forward)
    linear_y: float = 0.0  # m/s (lateral)
    angular_z: float = 0.0  # rad/s (yaw)
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None


class VelocityController:
    """Converts navigation goals into velocity commands.

    Implements a simple proportional controller for:
    - Goal following (move to target position)
    - Obstacle avoidance
    - Object tracking (move toward detected object)
    """

    def __init__(
        self,
        max_linear_velocity: float = 0.3,  # K1 recommended max for safe operation
        max_angular_velocity: float = 1.0,
        goal_threshold: float = 0.3,
        kp_linear: float = 1.0,
        kp_angular: float = 2.0,
        min_velocity: float = 0.05,
    ):
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.goal_threshold = goal_threshold
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.min_velocity = min_velocity

    def compute_goal_velocity(
        self,
        current_pose: np.ndarray,  # [x, y, yaw]
        goal_pose: np.ndarray,  # [x, y, yaw]
        obstacle_info: Optional[dict] = None,
    ) -> VelocityCommand:
        """Compute velocity to reach goal.

        Args:
            current_pose: Current robot pose [x, y, yaw]
            goal_pose: Goal pose [x, y, yaw]
            obstacle_info: Optional obstacle detection info

        Returns:
            VelocityCommand
        """
        # Compute error
        dx = goal_pose[0] - current_pose[0]
        dy = goal_pose[1] - current_pose[1]
        distance = math.sqrt(dx**2 + dy**2)

        # Check if we've reached the goal
        if distance < self.goal_threshold:
            return VelocityCommand()

        # Compute desired heading
        desired_yaw = math.atan2(dy, dx)
        yaw_error = self._normalize_angle(desired_yaw - current_pose[2])

        # Compute velocities
        linear = self.kp_linear * distance
        angular = self.kp_angular * yaw_error

        # Clamp velocities
        linear = np.clip(linear, self.min_velocity, self.max_linear_velocity)
        angular = np.clip(
            angular, -self.max_angular_velocity, self.max_angular_velocity
        )

        # Apply obstacle avoidance
        if obstacle_info and obstacle_info.get("has_obstacle"):
            linear, angular = self._avoid_obstacles(
                linear, angular, obstacle_info, distance
            )

        return VelocityCommand(
            linear_x=float(linear),
            angular_z=float(angular),
        )

    def compute_object_tracking_velocity(
        self,
        object_position: np.ndarray,  # [x, y, z] in camera frame
        obstacle_info: Optional[dict] = None,
        target_distance: float = 1.0,
    ) -> VelocityCommand:
        """Compute velocity to track an object.

        Args:
            object_position: Object position in camera frame
            obstacle_info: Optional obstacle detection info
            target_distance: Desired distance to maintain from object

        Returns:
            VelocityCommand
        """
        # Object direction in camera frame
        obj_x = object_position[0]  # lateral
        obj_z = object_position[2]  # forward

        # Distance error
        distance = math.sqrt(obj_x**2 + obj_z**2)
        distance_error = distance - target_distance

        # Angular error (turn toward object)
        angular_error = math.atan2(obj_x, obj_z)

        # Compute velocities
        linear = self.kp_linear * distance_error
        angular = self.kp_angular * angular_error

        # Clamp
        linear = np.clip(linear, -self.max_linear_velocity, self.max_linear_velocity)
        angular = np.clip(
            angular, -self.max_angular_velocity, self.max_angular_velocity
        )

        # Apply minimum velocity
        if abs(linear) > 0 and abs(linear) < self.min_velocity:
            linear = self.min_velocity * np.sign(linear)
        if abs(angular) > 0 and abs(angular) < self.min_velocity:
            angular = self.min_velocity * np.sign(angular)

        # Apply obstacle avoidance
        if obstacle_info and obstacle_info.get("has_obstacle"):
            linear, angular = self._avoid_obstacles(
                linear, angular, obstacle_info, distance
            )

        return VelocityCommand(
            linear_x=float(linear),
            angular_z=float(angular),
        )

    def compute_avoidance_velocity(
        self,
        obstacle_info: dict,
        current_velocity: Optional[VelocityCommand] = None,
    ) -> VelocityCommand:
        """Compute velocity to avoid obstacles.

        Args:
            obstacle_info: Obstacle detection info
            current_velocity: Current velocity command (to modify)

        Returns:
            VelocityCommand for avoidance
        """
        if current_velocity is None:
            current_velocity = VelocityCommand()

        linear = current_velocity.linear_x
        angular = current_velocity.angular_z

        # If obstacle is very close, stop or reverse
        min_dist = obstacle_info.get("min_distance", 1.0)
        if min_dist < 0.3:
            return VelocityCommand(linear_x=-0.1)  # Back up slowly

        # Determine which direction to turn
        if not obstacle_info.get("left_clear", True):
            angular = self.max_angular_velocity * 0.5  # Turn right
        elif not obstacle_info.get("right_clear", True):
            angular = -self.max_angular_velocity * 0.5  # Turn left
        elif not obstacle_info.get("center_clear", True):
            # Center blocked, choose direction with more space
            depths = obstacle_info.get("depths", {})
            left_depth = depths.get("left", 1.0)
            right_depth = depths.get("right", 1.0)
            if left_depth > right_depth:
                angular = self.max_angular_velocity * 0.3
            else:
                angular = -self.max_angular_velocity * 0.3

        # Reduce forward speed when obstacles are near
        if min_dist < 0.8:
            linear *= min_dist / 0.8

        return VelocityCommand(
            linear_x=float(np.clip(linear, -0.2, self.max_linear_velocity)),
            angular_z=float(
                np.clip(angular, -self.max_angular_velocity, self.max_angular_velocity)
            ),
        )

    def _avoid_obstacles(
        self,
        linear: float,
        angular: float,
        obstacle_info: dict,
        distance_to_goal: float,
    ) -> tuple[float, float]:
        """Modify velocity to avoid obstacles."""
        min_dist = obstacle_info.get("min_distance", 1.0)

        # Emergency stop if too close
        if min_dist < 0.25:
            return -0.1, angular  # Back up

        # Reduce speed near obstacles
        if min_dist < 0.6:
            linear *= min_dist / 0.6

        # Steer away from obstacles
        if not obstacle_info.get("left_clear", True):
            angular = max(angular, 0.3)
        elif not obstacle_info.get("right_clear", True):
            angular = min(angular, -0.3)

        return linear, angular

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
