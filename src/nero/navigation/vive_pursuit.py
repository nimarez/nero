"""Map-frame point pursuit using only an externally tracked robot pose."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nero.navigation.controller import VelocityCommand


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class VivePursuitConfig:
    """Conservative limits for navigation without onboard obstacle sensing."""

    max_linear_velocity: float = 0.10
    max_angular_velocity: float = 0.35
    min_linear_velocity: float = 0.03
    position_tolerance: float = 0.10
    bearing_tolerance: float = 0.12
    slowdown_distance: float = 0.75
    angular_gain: float = 1.5


class VivePursuitController:
    """Pursue a fixed world-frame point from a tracked ``[x, y, yaw]`` pose."""

    def __init__(self, config: VivePursuitConfig | None = None) -> None:
        self.config = config or VivePursuitConfig()
        values = tuple(vars(self.config).values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vive pursuit configuration must be finite")
        if self.config.max_linear_velocity <= 0 or self.config.max_angular_velocity <= 0:
            raise ValueError("maximum velocities must be positive")
        if not 0 <= self.config.min_linear_velocity <= self.config.max_linear_velocity:
            raise ValueError("minimum velocity must be within the linear velocity limit")
        if self.config.position_tolerance < 0 or self.config.bearing_tolerance < 0:
            raise ValueError("pursuit tolerances must be non-negative")
        if self.config.slowdown_distance <= 0 or self.config.angular_gain <= 0:
            raise ValueError("slowdown distance and angular gain must be positive")

    @staticmethod
    def _errors(robot_pose: np.ndarray, target_xy: np.ndarray) -> tuple[float, float]:
        pose = np.asarray(robot_pose, dtype=float)
        target = np.asarray(target_xy, dtype=float)
        if pose.shape != (3,) or target.shape != (2,):
            raise ValueError("robot_pose and target_xy must have shapes (3,) and (2,)")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(target)):
            raise ValueError("Vive pursuit inputs must be finite")
        delta = target - pose[:2]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        return distance, _normalize_angle(bearing - float(pose[2]))

    def has_arrived(self, robot_pose: np.ndarray, target_xy: np.ndarray, stand_off: float) -> bool:
        distance, bearing_error = self._errors(robot_pose, target_xy)
        return (
            distance <= stand_off + self.config.position_tolerance
            and abs(bearing_error) <= self.config.bearing_tolerance
        )

    def compute_command(
        self, robot_pose: np.ndarray, target_xy: np.ndarray, stand_off: float
    ) -> VelocityCommand:
        if not math.isfinite(stand_off) or stand_off <= 0:
            raise ValueError("stand_off must be positive and finite")
        distance, bearing_error = self._errors(robot_pose, target_xy)
        if distance <= stand_off + self.config.position_tolerance:
            if abs(bearing_error) <= self.config.bearing_tolerance:
                return VelocityCommand()
            return VelocityCommand(
                angular_z=float(
                    np.clip(
                        self.config.angular_gain * bearing_error,
                        -self.config.max_angular_velocity,
                        self.config.max_angular_velocity,
                    )
                )
            )

        travel = distance - stand_off
        heading_scale = max(0.0, math.cos(bearing_error))
        speed_scale = min(1.0, travel / self.config.slowdown_distance)
        linear = self.config.max_linear_velocity * heading_scale * speed_scale
        if abs(bearing_error) > math.pi / 3:
            linear = 0.0
        elif 0 < linear < self.config.min_linear_velocity:
            linear = self.config.min_linear_velocity
        angular = float(
            np.clip(
                self.config.angular_gain * bearing_error,
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )
        )
        return VelocityCommand(linear_x=float(linear), angular_z=angular)
