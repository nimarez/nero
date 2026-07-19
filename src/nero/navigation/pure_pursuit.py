"""Camera-frame pure pursuit for direct object following."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nero.navigation.controller import VelocityCommand


@dataclass(frozen=True)
class PurePursuitConfig:
    """Tuning values for direct RGB-D target pursuit."""

    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 0.7
    min_linear_velocity: float = 0.05
    position_tolerance: float = 0.12
    bearing_tolerance: float = 0.12
    slowdown_distance: float = 0.8


class PurePursuitController:
    """Drive toward a target measured in the robot camera frame.

    The target point is ``[x, y, z]`` with ``x`` to camera-right and ``z``
    forward. Internally, camera-right is converted to negative body lateral,
    matching the rest of Nero's forward-left-yaw convention. No global pose,
    map, path planner, or SLAM state is required.
    """

    def __init__(self, config: PurePursuitConfig | None = None) -> None:
        self.config = config or PurePursuitConfig()
        values = (
            self.config.max_linear_velocity,
            self.config.max_angular_velocity,
            self.config.min_linear_velocity,
            self.config.position_tolerance,
            self.config.bearing_tolerance,
            self.config.slowdown_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pure-pursuit configuration must be finite")
        if self.config.max_linear_velocity <= 0:
            raise ValueError("max_linear_velocity must be positive")
        if self.config.max_angular_velocity <= 0:
            raise ValueError("max_angular_velocity must be positive")
        if not 0 <= self.config.min_linear_velocity <= self.config.max_linear_velocity:
            raise ValueError(
                "min_linear_velocity must be non-negative and no greater than "
                "max_linear_velocity"
            )
        if self.config.position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative")
        if self.config.bearing_tolerance < 0:
            raise ValueError("bearing_tolerance must be non-negative")
        if self.config.slowdown_distance <= 0:
            raise ValueError("slowdown_distance must be positive")

    def has_arrived(self, target_camera: np.ndarray, stand_off: float) -> bool:
        """Return whether range and bearing are both inside their tolerances."""
        lateral, forward = self._planar_target(target_camera)
        distance = math.hypot(lateral, forward)
        bearing = math.atan2(lateral, forward)
        return (
            distance <= stand_off + self.config.position_tolerance
            and abs(bearing) <= self.config.bearing_tolerance
        )

    def compute_command(
        self,
        target_camera: np.ndarray,
        stand_off: float,
    ) -> VelocityCommand:
        """Compute a curvature command to a stand-off point before the target."""
        if stand_off <= 0:
            raise ValueError("stand_off must be positive")
        lateral, forward = self._planar_target(target_camera)
        distance = math.hypot(lateral, forward)
        bearing = math.atan2(lateral, forward)
        if distance <= stand_off + self.config.position_tolerance:
            angular = float(
                np.clip(
                    2.0 * bearing,
                    -self.config.max_angular_velocity,
                    self.config.max_angular_velocity,
                )
            )
            return VelocityCommand(angular_z=angular)

        travel = distance - stand_off
        scale = travel / distance
        goal_x = lateral * scale
        goal_z = forward * scale
        lookahead_sq = goal_x * goal_x + goal_z * goal_z
        if lookahead_sq <= 1e-9:
            return VelocityCommand()

        bearing = math.atan2(goal_x, goal_z)
        curvature = 2.0 * goal_x / lookahead_sq
        heading_scale = max(0.0, math.cos(bearing))
        speed_scale = min(1.0, travel / self.config.slowdown_distance)
        linear = self.config.max_linear_velocity * heading_scale * speed_scale
        if abs(bearing) > math.pi / 3:
            linear = 0.0
        elif 0 < linear < self.config.min_linear_velocity:
            linear = self.config.min_linear_velocity

        angular = float(
            np.clip(
                linear * curvature if linear else 2.0 * bearing,
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )
        )
        return VelocityCommand(linear_x=float(linear), angular_z=angular)

    @staticmethod
    def _planar_target(target_camera: np.ndarray) -> tuple[float, float]:
        target = np.asarray(target_camera, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_camera must be a finite [x, y, z] vector")
        lateral, forward = -float(target[0]), float(target[2])
        if forward <= 0:
            raise ValueError("target must be in front of the camera")
        return lateral, forward
