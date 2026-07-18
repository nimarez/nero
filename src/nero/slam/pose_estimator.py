"""Pose estimator: fuses SLAM, odometry, and IMU data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _blend_angles(first: float, second: float, second_weight: float) -> float:
    """Interpolate yaw on the unit circle without a +/-pi discontinuity."""
    delta = np.arctan2(np.sin(second - first), np.cos(second - first))
    return float(np.arctan2(np.sin(first + second_weight * delta), np.cos(first + second_weight * delta)))


@dataclass
class FusedPose:
    """Fused pose estimate from multiple sources."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z]
    yaw: float = 0.0
    timestamp: float = 0.0
    confidence: float = 1.0  # 0-1, how confident we are in this estimate
    source: str = "odom"  # "slam", "odom", "fused"

    @property
    def position_2d(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1], self.yaw])


class PoseEstimator:
    """Fuses SLAM pose, odometry, and IMU into a single estimate.

    Uses a complementary filter:
    - SLAM provides long-term accurate position (low frequency)
    - Odometry provides high-frequency updates (drifts over time)
    - IMU provides orientation and short-term motion smoothing
    """

    def __init__(
        self,
        slam_weight: float = 0.7,
        odom_weight: float = 0.3,
        imu_orientation_weight: float = 0.5,
    ):
        if slam_weight < 0 or odom_weight < 0 or slam_weight + odom_weight <= 0:
            raise ValueError("pose weights must be non-negative with a positive sum")
        if not 0 <= imu_orientation_weight <= 1:
            raise ValueError("imu_orientation_weight must be between zero and one")
        self.slam_weight = slam_weight
        self.odom_weight = odom_weight
        self.imu_orientation_weight = imu_orientation_weight

        self._last_slam_pose = None
        self._last_odom_pose = None
        self._last_imu_rpy = None
        self._odom_to_slam: np.ndarray | None = None
        self._fused_pose: Optional[FusedPose] = None
        self._last_update_time = 0.0

    def update(
        self,
        slam_pose=None,
        odom_pose=None,
        imu_rpy=None,
        timestamp: Optional[float] = None,
    ) -> FusedPose:
        """Update fused pose estimate.

        Args:
            slam_pose: SLAMPose from SLAM tracker (may be None if tracking lost)
            odom_pose: [x, y, yaw] from robot odometry
            imu_rpy: [roll, pitch, yaw] from IMU
            timestamp: Current timestamp (defaults to time.time())

        Returns:
            FusedPose estimate
        """
        ts = time.time() if timestamp is None else timestamp
        self._last_update_time = ts

        # Store latest readings
        if slam_pose is not None:
            self._last_slam_pose = slam_pose
        if odom_pose is not None:
            self._last_odom_pose = np.array(odom_pose)
        if imu_rpy is not None:
            self._last_imu_rpy = np.array(imu_rpy)

        # Determine which sources are available
        has_slam = (
            self._last_slam_pose is not None
            and self._last_slam_pose.tracking_status == "OK"
        )
        has_odom = self._last_odom_pose is not None
        has_imu = self._last_imu_rpy is not None

        if has_slam and has_odom and self._odom_to_slam is None:
            yaw_offset = float(
                np.arctan2(
                    np.sin(self._last_slam_pose.yaw - self._last_odom_pose[2]),
                    np.cos(self._last_slam_pose.yaw - self._last_odom_pose[2]),
                )
            )
            rotation = np.array(
                [
                    [np.cos(yaw_offset), -np.sin(yaw_offset)],
                    [np.sin(yaw_offset), np.cos(yaw_offset)],
                ]
            )
            self._odom_to_slam = np.eye(3)
            self._odom_to_slam[:2, :2] = rotation
            self._odom_to_slam[:2, 2] = (
                self._last_slam_pose.position[:2]
                - rotation @ self._last_odom_pose[:2]
            )

        aligned_odom = self._last_odom_pose
        if has_odom and self._odom_to_slam is not None:
            xy = self._odom_to_slam @ [
                self._last_odom_pose[0],
                self._last_odom_pose[1],
                1.0,
            ]
            yaw_offset = np.arctan2(
                self._odom_to_slam[1, 0], self._odom_to_slam[0, 0]
            )
            aligned_odom = np.array(
                [
                    xy[0],
                    xy[1],
                    _blend_angles(
                        self._last_odom_pose[2],
                        self._last_odom_pose[2] + yaw_offset,
                        1.0,
                    ),
                ]
            )

        if not has_slam and not has_odom:
            # No data available, return last estimate or zeros
            if self._fused_pose is None:
                return FusedPose(timestamp=ts, confidence=0.0)
            return self._fused_pose

        # Compute fused position
        if has_slam and has_odom:
            slam_pos = self._last_slam_pose.position[:2]
            odom_pos = aligned_odom[:2]
            position_2d = self.slam_weight * slam_pos + self.odom_weight * odom_pos
            # Normalize weights
            position_2d /= self.slam_weight + self.odom_weight
            confidence = 0.9
        elif has_slam:
            position_2d = self._last_slam_pose.position[:2]
            confidence = 0.8
        else:
            position_2d = aligned_odom[:2]
            confidence = 0.5

        # Compute fused yaw
        if has_slam:
            yaw = self._last_slam_pose.yaw
        elif has_imu and has_odom:
            yaw = _blend_angles(
                aligned_odom[2],
                self._last_imu_rpy[2],
                self.imu_orientation_weight,
            )
        elif has_imu:
            yaw = self._last_imu_rpy[2]
        elif has_odom:
            yaw = aligned_odom[2]
        else:
            yaw = 0.0

        # Z position (height) - use SLAM if available, otherwise 0
        if has_slam:
            z = self._last_slam_pose.position[2]
        else:
            z = 0.0

        position = np.array([position_2d[0], position_2d[1], z])

        self._fused_pose = FusedPose(
            position=position,
            yaw=yaw,
            timestamp=ts,
            confidence=confidence,
            source=(
                "fused" if has_slam and has_odom else ("slam" if has_slam else "odom")
            ),
        )

        return self._fused_pose

    def get_pose(self) -> Optional[FusedPose]:
        """Get current fused pose."""
        return self._fused_pose

    def reset(self) -> None:
        """Reset estimator state."""
        self._last_slam_pose = None
        self._last_odom_pose = None
        self._last_imu_rpy = None
        self._odom_to_slam = None
        self._fused_pose = None
        logger.info("Pose estimator reset")
