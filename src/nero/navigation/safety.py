"""Safety monitor for K1 robot navigation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyStatus:
    """Current safety status."""
    is_safe: bool = True
    reason: str = ""
    emergency_stop: bool = False
    warnings: list[str] = field(default_factory=list)


class SafetyMonitor:
    """Monitors robot state for safety violations.

    Checks:
    - Tilt angle (don't tip over)
    - Velocity limits
    - Obstacle proximity
    - SLAM tracking status
    - Battery level (if available)
    """

    def __init__(
        self,
        max_tilt_angle: float = 0.2,  # radians (~11 degrees)
        max_velocity: float = 0.8,  # m/s
        min_obstacle_distance: float = 0.2,  # meters
        max_tracking_lost_time: float = 3.0,  # seconds
    ):
        self.max_tilt_angle = max_tilt_angle
        self.max_velocity = max_velocity
        self.min_obstacle_distance = min_obstacle_distance
        self.max_tracking_lost_time = max_tracking_lost_time

        self._tracking_lost_since: Optional[float] = None
        self._last_safe_status = SafetyStatus()

    def check_safety(
        self,
        imu_rpy: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        obstacle_distance: float = 10.0,
        slam_tracking: bool = True,
        battery_level: Optional[float] = None,
    ) -> SafetyStatus:
        """Check all safety conditions.

        Args:
            imu_rpy: [roll, pitch, yaw] from IMU
            velocity: [linear_x, angular_z] current velocity
            obstacle_distance: Distance to nearest obstacle (m)
            slam_tracking: Whether SLAM is tracking
            battery_level: Battery percentage (0-100)

        Returns:
            SafetyStatus
        """
        warnings = []
        emergency_stop = False
        is_safe = True
        reason = ""

        # Check tilt angle
        if imu_rpy is not None:
            roll = abs(imu_rpy[0])
            pitch = abs(imu_rpy[1])
            if roll > self.max_tilt_angle or pitch > self.max_tilt_angle:
                is_safe = False
                reason = f"Excessive tilt: roll={roll:.2f}, pitch={pitch:.2f}"
                emergency_stop = True
                warnings.append(reason)

        # Check velocity
        if velocity is not None:
            linear = abs(velocity[0])
            angular = abs(velocity[1])
            if linear > self.max_velocity:
                warnings.append(f"Linear velocity exceeded: {linear:.2f} > {self.max_velocity}")
            if angular > self.max_velocity:
                warnings.append(f"Angular velocity exceeded: {angular:.2f} > {self.max_velocity}")

        # Check obstacle distance
        if obstacle_distance < self.min_obstacle_distance:
            is_safe = False
            reason = f"Obstacle too close: {obstacle_distance:.2f}m"
            emergency_stop = True
            warnings.append(reason)

        # Check SLAM tracking
        if not slam_tracking:
            if self._tracking_lost_since is None:
                self._tracking_lost_since = time.time()
            elif time.time() - self._tracking_lost_since > self.max_tracking_lost_time:
                is_safe = False
                reason = f"SLAM tracking lost for {time.time() - self._tracking_lost_since:.1f}s"
                emergency_stop = True
                warnings.append(reason)
        else:
            self._tracking_lost_since = None

        # Check battery
        if battery_level is not None and battery_level < 10:
            warnings.append(f"Low battery: {battery_level:.0f}%")
            if battery_level < 5:
                is_safe = False
                reason = "Critical battery level"
                emergency_stop = True

        status = SafetyStatus(
            is_safe=is_safe,
            reason=reason,
            emergency_stop=emergency_stop,
            warnings=warnings,
        )

        # Log warnings
        for w in warnings:
            logger.warning(w)
        if emergency_stop:
            logger.error(f"EMERGENCY STOP: {reason}")

        self._last_safe_status = status
        return status

    def reset(self) -> None:
        """Reset safety monitor state."""
        self._tracking_lost_since = None
        self._last_safe_status = SafetyStatus()
        logger.info("Safety monitor reset")

    @property
    def last_status(self) -> SafetyStatus:
        return self._last_safe_status