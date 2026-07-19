"""Stable ROS 2 topic contract consumed by visualization and recording tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityTopics:
    rgb: str = "/nero/sensors/rgb"
    depth: str = "/nero/sensors/depth"
    camera_info: str = "/nero/sensors/camera_info"
    imu: str = "/nero/sensors/imu"
    odometry: str = "/nero/sensors/odometry"
    joint_states: str = "/nero/sensors/joint_states"
    pose: str = "/nero/slam/pose"
    path: str = "/nero/slam/path"
    map_points: str = "/nero/slam/map_points"
    tracking: str = "/nero/slam/tracking"
    detections: str = "/nero/navigation/detections"
    status: str = "/nero/navigation/status"
    command: str = "/nero/navigation/cmd_vel"
    plan: str = "/nero/navigation/plan"
    goal_pose: str = "/nero/navigation/goal_pose"
    object_position: str = "/nero/navigation/object_position"
    reference_pose: str = "/nero/reference/pose"
    reference_path: str = "/nero/reference/path"
    reference_map: str = "/nero/reference/map_points"
