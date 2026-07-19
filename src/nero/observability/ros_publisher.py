"""Publish Nero's in-process state as standard ROS 2 messages.

ROS is an optional observability boundary: failures here must never take control
away from the navigation safety loop.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .topics import ObservabilityTopics

logger = logging.getLogger(__name__)


def _seconds_to_stamp(stamp: Any, timestamp: float) -> None:
    seconds = max(0.0, float(timestamp))
    stamp.sec = int(seconds)
    stamp.nanosec = int(round((seconds - stamp.sec) * 1_000_000_000))
    if stamp.nanosec == 1_000_000_000:
        stamp.sec += 1
        stamp.nanosec = 0


def _point_cloud_xyz(message: Any, points: np.ndarray) -> None:
    """Fill a PointCloud2-compatible object without requiring sensor_msgs_py."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    message.height = 1
    message.width = len(values)
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12 * len(values)
    message.is_dense = bool(np.all(np.isfinite(values)))
    message.data = np.ascontiguousarray(values).tobytes()


def navigation_geometry_payload(status: Any) -> dict[str, Any] | None:
    """Normalize SLAM-world or direct-camera stand-off geometry for Rerun."""
    state = getattr(getattr(status, "state", None), "value", "unknown")
    goal = getattr(status, "current_goal", None)
    object_world = getattr(goal, "object_position_world", None)
    radius = getattr(goal, "stand_off_distance", None)
    if object_world is not None and radius is not None:
        center = np.asarray(object_world, dtype=float).reshape(-1)
        approach = getattr(goal, "approach_pose", None)
        pose = getattr(status, "current_pose", None)
        robot_position = getattr(pose, "position", None)
        if center.size >= 3 and np.all(np.isfinite(center[:3])):
            payload: dict[str, Any] = {
                "frame": "map",
                "center": center[:3].tolist(),
                "radius": float(radius),
                "tolerance": 0.12,
                "state": state,
                "target": getattr(goal, "object_name", None),
            }
            waypoint = getattr(status, "active_waypoint", None)
            if waypoint is not None:
                values = np.asarray(waypoint, dtype=float).reshape(-1)
                if values.size >= 2 and np.all(np.isfinite(values[:2])):
                    payload["waypoint"] = [float(values[0]), float(values[1]), 0.03]
            if approach is not None:
                values = np.asarray(approach, dtype=float).reshape(-1)
                if values.size >= 2 and np.all(np.isfinite(values[:2])):
                    payload["approach"] = [
                        float(values[0]),
                        float(values[1]),
                        float(center[2]),
                    ]
            if robot_position is not None:
                values = np.asarray(robot_position, dtype=float).reshape(-1)
                if values.size >= 2 and np.all(np.isfinite(values[:2])):
                    payload["robot"] = [
                        float(values[0]),
                        float(values[1]),
                        float(values[2]) if values.size >= 3 else 0.0,
                    ]
            return payload

    center_camera = getattr(status, "target_position_camera", None)
    radius = getattr(status, "stand_off_distance", None)
    if center_camera is None or radius is None:
        return None
    center = np.asarray(center_camera, dtype=float).reshape(-1)
    if center.size < 3 or not np.all(np.isfinite(center[:3])):
        return None
    return {
        "frame": "camera",
        "center": center[:3].tolist(),
        "radius": float(radius),
        "tolerance": float(getattr(status, "stand_off_tolerance", 0.12)),
        "state": state,
        "target": getattr(status, "target", None),
        "robot": [0.0, 0.0, 0.0],
    }


def obstacle_mask_and_points(
    depth: np.ndarray | None,
    camera_matrix: np.ndarray | None,
    obstacle_info: dict[str, Any] | None,
    *,
    stride: int = 4,
    max_points: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand the controller mask and back-project its valid depth pixels."""
    if depth is None:
        return np.empty((0, 0), dtype=np.uint8), np.empty((0, 3), dtype=np.float32)
    values = np.asarray(depth)
    full_mask = np.zeros(values.shape[:2], dtype=np.uint8)
    region = None if not obstacle_info else obstacle_info.get("obstacle_mask")
    if region is not None:
        region = np.asarray(region, dtype=bool)
        height = min(full_mask.shape[0], region.shape[0])
        width = min(full_mask.shape[1], region.shape[1])
        full_mask[-height:, :width] = region[-height:, :width]
    if camera_matrix is None or not np.any(full_mask):
        return full_mask, np.empty((0, 3), dtype=np.float32)
    depth_m = values.astype(np.float32)
    if values.dtype == np.uint16:
        depth_m /= 1000.0
    sampled = full_mask.astype(bool)
    lattice = np.zeros_like(sampled)
    lattice[:: max(1, stride), :: max(1, stride)] = True
    valid = sampled & lattice & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.where(valid)
    if len(xs) > max_points:
        indices = np.linspace(0, len(xs) - 1, max_points, dtype=int)
        ys, xs = ys[indices], xs[indices]
    z = depth_m[ys, xs]
    intrinsics = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    points = np.column_stack(
        (
            (xs - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    return full_mask, points.astype(np.float32)


def inflated_occupancy_data(grid: Any, inflation_radius: float) -> np.ndarray:
    """Return ROS-order occupancy values with inflated known-free cells as 75."""
    data = np.asarray(grid.data, dtype=np.int8)
    radius_pixels = max(0, int(np.ceil(float(inflation_radius) / grid.resolution)))
    inflated = data.copy()
    if radius_pixels:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_pixels + 1, 2 * radius_pixels + 1)
        )
        blocked = (data != 0).astype(np.uint8)
        halo = cv2.dilate(blocked, kernel).astype(bool) & (data == 0)
        inflated[halo] = 75
    return np.flipud(inflated).reshape(-1)


def safety_payload(status: Any) -> dict[str, Any] | None:
    """Serialize the live safety decision and the measurements behind it."""
    safety = getattr(status, "safety_status", None)
    if safety is None:
        return None
    obstacles = getattr(status, "obstacle_info", None) or {}

    def finite(value: Any) -> float | None:
        if value is None:
            return None
        number = float(value)
        return number if np.isfinite(number) else None

    obstacle_distance = finite(getattr(safety, "obstacle_distance", obstacles.get("min_distance")))
    return {
        "is_safe": bool(getattr(safety, "is_safe", True)),
        "emergency_stop": bool(getattr(safety, "emergency_stop", False)),
        "reason": str(getattr(safety, "reason", "")),
        "warnings": [str(value) for value in getattr(safety, "warnings", [])],
        "roll_rad": finite(getattr(safety, "roll_rad", None)),
        "pitch_rad": finite(getattr(safety, "pitch_rad", None)),
        "max_tilt_rad": finite(getattr(safety, "max_tilt_angle", None)),
        "obstacle_distance": obstacle_distance,
        "min_obstacle_distance": finite(getattr(safety, "min_obstacle_distance", None)),
        "battery_percent": finite(getattr(safety, "battery_level", None)),
        "depth_sensor_blind": bool(
            getattr(
                safety,
                "depth_sensor_blind",
                obstacles.get("sensor_blind", False),
            )
        ),
        "has_obstacle": bool(obstacles.get("has_obstacle", False)),
        "left_clear": bool(obstacles.get("left_clear", True)),
        "center_clear": bool(obstacles.get("center_clear", True)),
        "right_clear": bool(obstacles.get("right_clear", True)),
    }


class RosObservabilityPublisher:
    """Best-effort publisher for live sensors, policy state, and references."""

    def __init__(
        self,
        *,
        node: Any | None = None,
        topics: ObservabilityTopics | None = None,
        map_frame: str = "map",
        camera_frame: str = "nero_camera",
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PointStamped, PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid as RosOccupancyGrid, Odometry, Path
        from sensor_msgs.msg import (
            CameraInfo,
            Image,
            Imu,
            JointState,
            PointCloud2,
            PointField,
        )
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = node or rclpy.create_node("nero_observability")
        self._owns_node = node is None
        self._topics = topics or ObservabilityTopics()
        self._map_frame = map_frame
        self._camera_frame = camera_frame
        self._types = {
            "PoseStamped": PoseStamped,
            "PointStamped": PointStamped,
            "Twist": Twist,
            "Path": Path,
            "CameraInfo": CameraInfo,
            "Image": Image,
            "Imu": Imu,
            "Odometry": Odometry,
            "OccupancyGrid": RosOccupancyGrid,
            "JointState": JointState,
            "PointCloud2": PointCloud2,
            "PointField": PointField,
            "String": String,
        }
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        self._publishers = {
            "rgb": self._node.create_publisher(Image, self._topics.rgb, sensor_qos),
            "depth": self._node.create_publisher(Image, self._topics.depth, sensor_qos),
            "camera_info": self._node.create_publisher(CameraInfo, self._topics.camera_info, 1),
            "imu": self._node.create_publisher(Imu, self._topics.imu, sensor_qos),
            "odometry": self._node.create_publisher(Odometry, self._topics.odometry, sensor_qos),
            "joint_states": self._node.create_publisher(
                JointState, self._topics.joint_states, sensor_qos
            ),
            "pose": self._node.create_publisher(PoseStamped, self._topics.pose, 10),
            "path": self._node.create_publisher(Path, self._topics.path, 1),
            "map_points": self._node.create_publisher(
                PointCloud2, self._topics.map_points, sensor_qos
            ),
            "tracking": self._node.create_publisher(String, self._topics.tracking, 10),
            "status": self._node.create_publisher(String, self._topics.status, 10),
            "command": self._node.create_publisher(Twist, self._topics.command, 10),
            "plan": self._node.create_publisher(Path, self._topics.plan, 1),
            "goal_pose": self._node.create_publisher(PoseStamped, self._topics.goal_pose, 10),
            "object_position": self._node.create_publisher(
                PointStamped, self._topics.object_position, 10
            ),
            "reference_pose": self._node.create_publisher(
                PoseStamped, self._topics.reference_pose, 10
            ),
            "reference_path": self._node.create_publisher(Path, self._topics.reference_path, 1),
            "reference_map": self._node.create_publisher(
                PointCloud2, self._topics.reference_map, sensor_qos
            ),
            "detections": self._node.create_publisher(String, self._topics.detections, sensor_qos),
            "obstacle_mask": self._node.create_publisher(
                Image, self._topics.obstacle_mask, sensor_qos
            ),
            "obstacle_points": self._node.create_publisher(
                PointCloud2, self._topics.obstacle_points, sensor_qos
            ),
            "occupancy_grid": self._node.create_publisher(
                RosOccupancyGrid, self._topics.occupancy_grid, 1
            ),
        }
        self._path = Path()
        self._path.header.frame_id = map_frame
        self._plan = Path()
        self._plan.header.frame_id = map_frame
        self._reference_path = Path()
        self._reference_path.header.frame_id = map_frame
        self._last_sensor_timestamp: float | None = None
        self._last_depth: np.ndarray | None = None
        self._last_camera_matrix: np.ndarray | None = None
        self._last_grid_publish_monotonic = 0.0

    @classmethod
    def try_create(
        cls, *, enabled: bool = True, **kwargs: Any
    ) -> "RosObservabilityPublisher | None":
        if not enabled:
            return None
        try:
            return cls(**kwargs)
        except (ImportError, RuntimeError) as exc:
            logger.warning("ROS observability is unavailable: %s", exc)
            return None

    def _header(self, message: Any, timestamp: float, frame_id: str) -> None:
        _seconds_to_stamp(message.header.stamp, timestamp)
        message.header.frame_id = frame_id

    def _pose_message(self, matrix: np.ndarray, timestamp: float) -> Any:
        message = self._types["PoseStamped"]()
        self._header(message, timestamp, self._map_frame)
        transform = np.asarray(matrix, dtype=float)
        message.pose.position.x, message.pose.position.y, message.pose.position.z = transform[:3, 3]
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = quaternion
        return message

    def publish_robot_state(self, state: Any, robot: Any) -> None:
        if state.rgb is None or state.depth is None:
            return
        timestamp = float(robot.image_timestamp(state.rgb))
        if timestamp == self._last_sensor_timestamp:
            return
        self._last_sensor_timestamp = timestamp
        rgb = np.ascontiguousarray(robot.image_to_array(state.rgb))
        depth = np.ascontiguousarray(robot.image_to_array(state.depth))
        self._last_depth = depth

        rgb_msg = self._types["Image"]()
        self._header(rgb_msg, timestamp, self._camera_frame)
        rgb_msg.height, rgb_msg.width = rgb.shape[:2]
        rgb_msg.encoding = "bgr8" if rgb.ndim == 3 else "mono8"
        rgb_msg.is_bigendian = False
        rgb_msg.step = int(rgb.strides[0])
        rgb_msg.data = rgb.tobytes()
        self._publishers["rgb"].publish(rgb_msg)

        depth_msg = self._types["Image"]()
        self._header(depth_msg, timestamp, self._camera_frame)
        depth_msg.height, depth_msg.width = depth.shape[:2]
        depth_msg.encoding = "16UC1" if depth.dtype == np.uint16 else "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = int(depth.strides[0])
        depth_msg.data = depth.tobytes()
        self._publishers["depth"].publish(depth_msg)

        info = state.camera_info
        if info is not None:
            self._last_camera_matrix = np.asarray(info.k, dtype=float).reshape(3, 3)
            info_msg = self._types["CameraInfo"]()
            self._header(info_msg, timestamp, self._camera_frame)
            info_msg.width, info_msg.height = int(info.width), int(info.height)
            info_msg.k = np.asarray(info.k, dtype=float).reshape(-1).tolist()
            info_msg.d = list(getattr(info, "d", []))
            info_msg.r = np.eye(3).reshape(-1).tolist()
            info_msg.p = (
                np.hstack((np.asarray(info.k).reshape(3, 3), np.zeros((3, 1)))).reshape(-1).tolist()
            )
            self._publishers["camera_info"].publish(info_msg)

        if state.imu is not None:
            imu_msg = self._types["Imu"]()
            self._header(imu_msg, timestamp, "imu_link")
            quaternion = Rotation.from_euler("xyz", state.orientation_rpy).as_quat()
            (
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w,
            ) = quaternion
            (
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ) = state.angular_velocity
            (
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ) = state.linear_acceleration
            self._publishers["imu"].publish(imu_msg)

        if state.odom is not None:
            odometry = self._types["Odometry"]()
            self._header(odometry, timestamp, "odom")
            odometry.child_frame_id = "base_link"
            x, y, yaw = np.asarray(state.position_2d, dtype=float)
            odometry.pose.pose.position.x = float(x)
            odometry.pose.pose.position.y = float(y)
            quaternion = Rotation.from_euler("z", float(yaw)).as_quat()
            (
                odometry.pose.pose.orientation.x,
                odometry.pose.pose.orientation.y,
                odometry.pose.pose.orientation.z,
                odometry.pose.pose.orientation.w,
            ) = quaternion
            self._publishers["odometry"].publish(odometry)

        if state.joints is not None:
            joints = self._types["JointState"]()
            self._header(joints, timestamp, "base_link")
            joints.name = list(getattr(state.joints, "name", []))
            joints.position = list(getattr(state.joints, "position", []))
            joints.velocity = list(getattr(state.joints, "velocity", []))
            joints.effort = list(getattr(state.joints, "effort", []))
            self._publishers["joint_states"].publish(joints)

    def publish_policy(self, status: Any, timestamp: float) -> None:
        state_name = getattr(getattr(status, "state", None), "value", "unknown")
        status_message = self._types["String"]()
        goal_name = getattr(getattr(status, "current_goal", None), "object_name", None)
        if goal_name is None:
            goal_name = getattr(status, "target", None)
        status_message.data = json.dumps(
            {
                "state": state_name,
                "message": getattr(status, "message", ""),
                "goal": goal_name,
                "tracking_confidence": getattr(
                    getattr(status, "current_pose", None), "confidence", None
                ),
                "navigation_geometry": navigation_geometry_payload(status),
                "safety": safety_payload(status),
                "detector": getattr(status, "detector_metrics", None),
            },
            separators=(",", ":"),
        )
        self._publishers["status"].publish(status_message)

        command = getattr(status, "velocity_command", None)
        twist = self._types["Twist"]()
        if command is not None:
            twist.linear.x = float(command.linear_x)
            twist.linear.y = float(command.linear_y)
            twist.angular.z = float(command.angular_z)
        self._publishers["command"].publish(twist)

        self.publish_plan(getattr(status, "planned_path", None), timestamp)

        pose = getattr(status, "current_pose", None)
        if pose is not None:
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler("z", float(pose.yaw)).as_matrix()
            matrix[:3, 3] = np.asarray(pose.position, dtype=float)
            self.publish_pose(matrix, timestamp)
        elif getattr(status, "pose", None) is not None:
            planar_pose = np.asarray(status.pose, dtype=float)
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler("z", float(planar_pose[2])).as_matrix()
            matrix[:2, 3] = planar_pose[:2]
            self.publish_pose(matrix, timestamp)
        goal = getattr(status, "current_goal", None)
        approach = getattr(goal, "approach_pose", None)
        if approach is None:
            approach = getattr(status, "goal_pose", None)
        if approach is not None:
            goal_matrix = np.eye(4)
            goal_matrix[:3, :3] = Rotation.from_euler("z", float(approach[2])).as_matrix()
            goal_matrix[:2, 3] = np.asarray(approach[:2], dtype=float)
            self._publishers["goal_pose"].publish(self._pose_message(goal_matrix, timestamp))
        object_position = getattr(goal, "object_position_world", None)
        if object_position is not None:
            point = self._types["PointStamped"]()
            self._header(point, timestamp, self._map_frame)
            point.point.x, point.point.y, point.point.z = map(
                float, np.asarray(object_position, dtype=float)
            )
            self._publishers["object_position"].publish(point)
        self.publish_detections(getattr(status, "detections", []), timestamp)
        self.publish_obstacles(getattr(status, "obstacle_info", None), timestamp)
        grid = getattr(status, "occupancy_grid", None)
        if grid is not None and time.monotonic() - self._last_grid_publish_monotonic >= 1.0:
            self.publish_occupancy_grid(
                grid, timestamp, float(getattr(status, "map_inflation_radius", 0.0) or 0.0)
            )
            self._last_grid_publish_monotonic = time.monotonic()

    def publish_obstacles(self, obstacle_info: dict | None, timestamp: float) -> None:
        mask, points = obstacle_mask_and_points(
            getattr(self, "_last_depth", None),
            getattr(self, "_last_camera_matrix", None),
            obstacle_info,
        )
        if not mask.size:
            return
        image = self._types["Image"]()
        self._header(image, timestamp, self._camera_frame)
        image.height, image.width = mask.shape
        image.encoding = "mono8"
        image.is_bigendian = False
        image.step = int(mask.strides[0])
        image.data = np.ascontiguousarray(mask).tobytes()
        self._publishers["obstacle_mask"].publish(image)

        cloud = self._types["PointCloud2"]()
        self._header(cloud, timestamp, self._camera_frame)
        field_type = self._types["PointField"]
        cloud.fields = [
            field_type(name=name, offset=offset, datatype=field_type.FLOAT32, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        ]
        _point_cloud_xyz(cloud, points)
        self._publishers["obstacle_points"].publish(cloud)

    def publish_occupancy_grid(self, grid: Any, timestamp: float, inflation_radius: float) -> None:
        message = self._types["OccupancyGrid"]()
        self._header(message, timestamp, self._map_frame)
        message.info.resolution = float(grid.resolution)
        message.info.width = int(grid.width)
        message.info.height = int(grid.height)
        message.info.origin.position.x = float(grid.origin[0])
        message.info.origin.position.y = float(grid.origin[1])
        message.info.origin.orientation.w = 1.0
        message.data = inflated_occupancy_data(grid, inflation_radius).tolist()
        self._publishers["occupancy_grid"].publish(message)

    def publish_plan(self, points: Any, timestamp: float) -> None:
        """Publish the current controller route, including an empty path to clear it."""
        path = self._types["Path"]()
        self._header(path, timestamp, self._map_frame)
        if points is not None:
            values = np.asarray(points, dtype=float)
            if values.size:
                values = values.reshape(-1, 3)
                for value in values:
                    pose = self._types["PoseStamped"]()
                    self._header(pose, timestamp, self._map_frame)
                    pose.pose.position.x = float(value[0])
                    pose.pose.position.y = float(value[1])
                    pose.pose.position.z = float(value[2])
                    pose.pose.orientation.w = 1.0
                    path.poses.append(pose)
        self._plan = path
        self._publishers["plan"].publish(path)

    def publish_pose(
        self, matrix: np.ndarray, timestamp: float, *, reference: bool = False
    ) -> None:
        message = self._pose_message(matrix, timestamp)
        path = self._reference_path if reference else self._path
        publisher = self._publishers["reference_pose" if reference else "pose"]
        path_publisher = self._publishers["reference_path" if reference else "path"]
        publisher.publish(message)
        path.header = message.header
        path.poses.append(message)
        if len(path.poses) > 10_000:
            del path.poses[:5_000]
        path_publisher.publish(path)

    def publish_tracking(self, tracking_status: str, map_point_count: int) -> None:
        message = self._types["String"]()
        message.data = json.dumps(
            {"status": str(tracking_status), "map_points": int(map_point_count)},
            separators=(",", ":"),
        )
        self._publishers["tracking"].publish(message)

    def publish_detections(self, detections: list[Any], timestamp: float) -> None:
        message = self._types["String"]()
        items = []
        for detection in detections:
            position = getattr(detection, "position_3d", None)
            items.append(
                {
                    "label": str(detection.label),
                    "confidence": float(detection.confidence),
                    "bbox": [int(value) for value in detection.bbox],
                    "position_3d": (
                        np.asarray(position, dtype=float).tolist() if position is not None else None
                    ),
                    "distance": float(getattr(detection, "distance", 0.0)),
                    "coordinate_frame": str(getattr(detection, "coordinate_frame", "camera")),
                }
            )
        message.data = json.dumps(
            {"timestamp": float(timestamp), "detections": items},
            separators=(",", ":"),
        )
        self._publishers["detections"].publish(message)

    def publish_point_cloud(
        self, points: np.ndarray, timestamp: float, *, reference: bool = False
    ) -> None:
        message = self._types["PointCloud2"]()
        self._header(message, timestamp, self._map_frame)
        field_type = self._types["PointField"]
        message.fields = [
            field_type(name=name, offset=offset, datatype=field_type.FLOAT32, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        ]
        finite = np.asarray(points, dtype=float).reshape(-1, 3)
        finite = finite[np.all(np.isfinite(finite), axis=1)]
        _point_cloud_xyz(message, finite)
        self._publishers["reference_map" if reference else "map_points"].publish(message)

    def close(self) -> None:
        if self._owns_node:
            self._node.destroy_node()
