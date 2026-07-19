"""Smooth object-approach planning using only an externally tracked robot pose."""

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


@dataclass(frozen=True)
class ViveApproachPath:
    """A smooth route ending in front of, and facing, an object pose."""

    object_pose: np.ndarray
    goal_pose: np.ndarray
    points: np.ndarray


def object_approach_pose(object_pose: np.ndarray, stand_off: float) -> np.ndarray:
    """Return the robot pose in front of an object and facing back toward it."""
    target = np.asarray(object_pose, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("object_pose must be a finite [x, y, yaw] vector")
    if not math.isfinite(stand_off) or stand_off <= 0:
        raise ValueError("stand_off must be positive and finite")
    heading = np.array([math.cos(target[2]), math.sin(target[2])])
    return np.array(
        [
            target[0] + stand_off * heading[0],
            target[1] + stand_off * heading[1],
            _normalize_angle(float(target[2]) + math.pi),
        ]
    )


def _cubic_controls(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    chord = float(np.linalg.norm(goal[:2] - start[:2]))
    tangent_length = min(1.0, max(0.20, 0.4 * chord))
    start_heading = np.array([math.cos(start[2]), math.sin(start[2])])
    goal_heading = np.array([math.cos(goal[2]), math.sin(goal[2])])
    return np.array(
        [
            start[:2],
            start[:2] + tangent_length * start_heading,
            goal[:2] - tangent_length * goal_heading,
            goal[:2],
        ]
    )


def _sample_cubic(controls: np.ndarray, spacing: float) -> np.ndarray:
    control_length = float(np.linalg.norm(np.diff(controls, axis=0), axis=1).sum())
    count = max(2, min(2000, int(math.ceil(control_length / spacing)) + 1))
    t = np.linspace(0.0, 1.0, count)[:, None]
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * controls[0]
        + 3.0 * one_minus_t**2 * t * controls[1]
        + 3.0 * one_minus_t * t**2 * controls[2]
        + t**3 * controls[3]
    )


def _is_regular_cubic(controls: np.ndarray) -> bool:
    """Return whether a cubic has no zero-tangent direction reversal."""
    t = np.linspace(0.0, 1.0, 501)[:, None]
    derivative = 3.0 * (
        (1.0 - t) ** 2 * (controls[1] - controls[0])
        + 2.0 * (1.0 - t) * t * (controls[2] - controls[1])
        + t**2 * (controls[3] - controls[2])
    )
    speed = np.linalg.norm(derivative, axis=1)
    scale = max(1.0, float(np.linalg.norm(np.diff(controls, axis=0), axis=1).sum()))
    # Near-zero tangents are numerically regular but behave like cusps for a
    # sampled pure-pursuit tracker, so require useful forward progress too.
    if float(speed.min()) <= 1e-2 * scale:
        return False
    directions = derivative / speed[:, None]
    return bool(np.all(np.sum(directions[1:] * directions[:-1], axis=1) > 0.0))


def plan_object_approach(
    start_pose: np.ndarray,
    object_pose: np.ndarray,
    stand_off: float,
    *,
    spacing: float = 0.05,
) -> ViveApproachPath:
    """Plan a regular Bezier route with start and terminal heading constraints."""
    start = np.asarray(start_pose, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("start_pose must be a finite [x, y, yaw] vector")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")
    target = np.asarray(object_pose, dtype=float)
    goal = object_approach_pose(target, stand_off)
    controls = _cubic_controls(start, goal)
    if _is_regular_cubic(controls):
        points = _sample_cubic(controls, spacing)
    else:
        delta = goal[:2] - start[:2]
        chord = float(np.linalg.norm(delta))
        if chord <= 1e-6:
            raise ValueError("cannot plan a regular path between coincident positions")
        direction = delta / chord
        normal = np.array([-direction[1], direction[0]])
        detour = min(2.0, max(0.5, 0.6 * chord))
        candidates: list[tuple[float, np.ndarray]] = []
        for side in (-1.0, 1.0):
            midpoint = np.array(
                [
                    *(0.5 * (start[:2] + goal[:2]) + side * detour * normal),
                    math.atan2(direction[1], direction[0]),
                ]
            )
            first = _cubic_controls(start, midpoint)
            second = _cubic_controls(midpoint, goal)
            if not (_is_regular_cubic(first) and _is_regular_cubic(second)):
                continue
            first_points = _sample_cubic(first, spacing)
            second_points = _sample_cubic(second, spacing)
            candidate = np.vstack((first_points, second_points[1:]))
            length = float(np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum())
            candidates.append((length, candidate))
        if not candidates:
            raise ValueError("cannot plan a regular forward path for these headings")
        points = min(candidates, key=lambda value: value[0])[1]
    return ViveApproachPath(object_pose=target.copy(), goal_pose=goal, points=points)


class VivePathTracker:
    """Select forward-only lookahead points along one fixed sampled path."""

    def __init__(self, points: np.ndarray) -> None:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError("path points must have shape (N, 2) with N >= 2")
        if not np.all(np.isfinite(values)):
            raise ValueError("path points must be finite")
        self.points = values
        self.index = 0

    def lookahead(self, position: np.ndarray, distance: float) -> np.ndarray:
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("position must be a finite [x, y] vector")
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("lookahead distance must be positive and finite")
        remaining = self.points[self.index :]
        self.index += int(np.argmin(np.linalg.norm(remaining - current, axis=1)))
        traveled = 0.0
        for index in range(self.index + 1, len(self.points)):
            traveled += float(np.linalg.norm(self.points[index] - self.points[index - 1]))
            if traveled >= distance:
                return self.points[index].copy()
        return self.points[-1].copy()

    def remaining_distance(self, position: np.ndarray) -> float:
        """Return distance from the robot through the remaining sampled path."""
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("position must be a finite [x, y] vector")
        connection = float(np.linalg.norm(current - self.points[self.index]))
        path = float(np.linalg.norm(np.diff(self.points[self.index :], axis=0), axis=1).sum())
        return connection + path


class VivePursuitController:
    """Follow a world-frame lookahead point and settle at a terminal pose."""

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

    def has_reached_pose(self, robot_pose: np.ndarray, goal_pose: np.ndarray) -> bool:
        goal = np.asarray(goal_pose, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_pose must be a finite [x, y, yaw] vector")
        distance, _ = self._errors(robot_pose, goal[:2])
        yaw_error = _normalize_angle(float(goal[2] - robot_pose[2]))
        return (
            distance <= self.config.position_tolerance
            and abs(yaw_error) <= self.config.bearing_tolerance
        )

    def compute_path_command(
        self,
        robot_pose: np.ndarray,
        lookahead_xy: np.ndarray,
        goal_pose: np.ndarray,
        remaining_distance: float | None = None,
    ) -> VelocityCommand:
        goal = np.asarray(goal_pose, dtype=float)
        goal_distance, _ = self._errors(robot_pose, goal[:2])
        if goal_distance <= self.config.position_tolerance:
            yaw_error = _normalize_angle(float(goal[2] - robot_pose[2]))
            if abs(yaw_error) <= self.config.bearing_tolerance:
                return VelocityCommand()
            return VelocityCommand(
                angular_z=float(
                    np.clip(
                        self.config.angular_gain * yaw_error,
                        -self.config.max_angular_velocity,
                        self.config.max_angular_velocity,
                    )
                )
            )
        if remaining_distance is None:
            remaining_distance = goal_distance
        if not math.isfinite(remaining_distance) or remaining_distance < 0:
            raise ValueError("remaining_distance must be non-negative and finite")
        _, bearing_error = self._errors(robot_pose, lookahead_xy)
        heading_scale = max(0.0, math.cos(bearing_error))
        speed_scale = min(1.0, remaining_distance / self.config.slowdown_distance)
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

    def compute_command(
        self, robot_pose: np.ndarray, target_xy: np.ndarray, stand_off: float
    ) -> VelocityCommand:
        if not math.isfinite(stand_off) or stand_off < 0:
            raise ValueError("stand_off must be non-negative and finite")
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
