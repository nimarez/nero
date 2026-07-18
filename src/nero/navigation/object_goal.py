"""Geometry for turning live object detections into world-frame approach poses."""

from __future__ import annotations

import math

import numpy as np


def camera_point_to_world(
    position_camera: np.ndarray, camera_pose_world: np.ndarray
) -> np.ndarray:
    """Transform an OpenCV camera point (right, down, forward) into the map frame."""
    point = np.asarray(position_camera, dtype=float)
    transform = np.asarray(camera_pose_world, dtype=float)
    if point.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("camera point and pose must have shapes (3,) and (4, 4)")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(transform)):
        raise ValueError("camera point and pose must be finite")
    return (transform @ np.append(point, 1.0))[:3]


def planar_detection_to_world(
    position_camera: np.ndarray, robot_pose: np.ndarray
) -> np.ndarray:
    """Project a fixed forward-facing optical camera detection into a 2D world.

    This is used only by the lightweight synthetic simulator, which has no
    calibrated camera transform or SLAM camera pose.
    """
    point = np.asarray(position_camera, dtype=float)
    pose = np.asarray(robot_pose, dtype=float)
    if point.shape != (3,) or pose.shape != (3,):
        raise ValueError("camera point and robot pose must both be 3-vectors")
    # OpenCV optical x points right; a planar robot's positive y points left.
    relative_body = np.array([point[2], -point[0]])
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    body_to_world = np.array([[cosine, -sine], [sine, cosine]])
    xy = pose[:2] + body_to_world @ relative_body
    return np.array([xy[0], xy[1], 0.0])


def body_point_to_world(
    position_body: np.ndarray, robot_pose: np.ndarray
) -> np.ndarray:
    """Transform a body-frame point (forward, left, up) into the map frame."""
    point = np.asarray(position_body, dtype=float)
    pose = np.asarray(robot_pose, dtype=float)
    if point.shape != (3,) or pose.shape != (3,):
        raise ValueError("body point and robot pose must both be 3-vectors")
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    body_to_world = np.array([[cosine, -sine], [sine, cosine]])
    xy = pose[:2] + body_to_world @ point[:2]
    return np.array([xy[0], xy[1], point[2]])


def approach_pose(
    robot_pose: np.ndarray,
    object_position_world: np.ndarray,
    stand_off_distance: float,
) -> np.ndarray:
    """Return an ``[x, y, yaw]`` pose that faces an object from a safe radius."""
    robot = np.asarray(robot_pose, dtype=float)
    object_position = np.asarray(object_position_world, dtype=float)
    if robot.shape != (3,) or object_position.shape not in {(2,), (3,)}:
        raise ValueError("robot pose and object position have invalid shapes")
    if stand_off_distance <= 0 or not np.isfinite(stand_off_distance):
        raise ValueError("stand_off_distance must be finite and positive")
    delta = object_position[:2] - robot[:2]
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return robot.copy()
    direction = delta / distance
    goal_xy = (
        robot[:2]
        if distance <= stand_off_distance
        else object_position[:2] - stand_off_distance * direction
    )
    goal_yaw = math.atan2(delta[1], delta[0])
    return np.array([goal_xy[0], goal_xy[1], goal_yaw])


def blend_world_position(
    previous: np.ndarray | None, observed: np.ndarray, observation_weight: float
) -> np.ndarray:
    """Low-pass filter object position without hiding invalid observations."""
    if not 0 < observation_weight <= 1:
        raise ValueError("observation_weight must be in (0, 1]")
    value = np.asarray(observed, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("observed object position must be a finite 3-vector")
    if previous is None:
        return value.copy()
    old = np.asarray(previous, dtype=float)
    if old.shape != (3,) or not np.all(np.isfinite(old)):
        raise ValueError("previous object position must be a finite 3-vector")
    return (1.0 - observation_weight) * old + observation_weight * value
