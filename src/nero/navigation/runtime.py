"""Shared sensor, localization, safety, and command runtime for navigation policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from nero.navigation.controller import VelocityCommand
from nero.slam.orb_slam3_node import SLAMPose
from nero.slam.pose_estimator import FusedPose


@dataclass(frozen=True)
class SensorFrame:
    """One synchronized K1 RGB-D/IMU/odometry observation."""

    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    camera_info: Any
    imu_rpy: np.ndarray
    imu_samples: Any
    odometry: np.ndarray
    raw_state: Any = None


@dataclass(frozen=True)
class LocalizedFrame:
    """Shared result consumed by semantic and map navigation policies."""

    sensor: SensorFrame
    slam_pose: SLAMPose
    fused_pose: FusedPose
    safety_status: Any
    obstacle_info: dict


def read_sensor_frame(robot: Any) -> SensorFrame:
    """Read and normalize one synchronized observation from a robot adapter."""
    state = robot.get_state(include_images=True)
    return SensorFrame(
        rgb=robot.image_to_array(state.rgb),
        depth=robot.image_to_array(state.depth),
        timestamp=robot.image_timestamp(state.rgb),
        camera_info=state.camera_info,
        imu_rpy=np.asarray(state.orientation_rpy, dtype=float),
        imu_samples=getattr(state, "imu_samples", None),
        odometry=np.asarray(state.position_2d, dtype=float),
        raw_state=state,
    )


def initialize_sensor_navigation(
    robot: Any,
    slam: Any,
    pose_estimator: Any,
    safety: Any,
) -> None:
    """Initialize the common real/sim-adapter navigation components."""
    robot.initialize()
    slam.initialize(robot.get_camera_info())
    pose_estimator.reset()
    safety.reset()


def localize_sensor_frame(
    sensor: SensorFrame,
    *,
    slam: Any,
    pose_estimator: Any,
    depth_processor: Any,
    safety: Any,
) -> LocalizedFrame:
    """Run IMU-RGBD SLAM, pose fusion, safety, and obstacle processing once."""
    slam_pose = slam.track_frame(
        sensor.rgb,
        sensor.depth,
        imu_data=sensor.imu_samples,
        timestamp=sensor.timestamp,
    )
    fused_pose = pose_estimator.update(
        slam_pose=slam.body_pose(slam_pose),
        odom_pose=sensor.odometry,
        imu_rpy=sensor.imu_rpy,
        timestamp=sensor.timestamp,
    )
    depth_m = depth_processor.preprocess(sensor.depth)
    obstacle_info = depth_processor.detect_obstacles(depth_m)
    # Tracking readiness is handled as a recoverable navigation state by the
    # policy. Physical hazards remain emergency-stop conditions here.
    safety_status = safety.check_safety(
        imu_rpy=sensor.imu_rpy,
        obstacle_distance=float(obstacle_info.get("min_distance", float("inf"))),
        battery_level=getattr(sensor.raw_state, "battery_level", None),
    )
    return LocalizedFrame(
        sensor=sensor,
        slam_pose=slam_pose,
        fused_pose=fused_pose,
        safety_status=safety_status,
        obstacle_info=obstacle_info,
    )


def send_velocity(robot: Any, command: Optional[VelocityCommand] = None) -> None:
    """Send a typed command, defaulting to a safe stop."""
    command = command or VelocityCommand()
    robot.set_velocity(
        command.linear_x,
        command.linear_y,
        command.angular_z,
    )
