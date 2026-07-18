"""Publish Nero's in-process state as standard ROS 2 messages.

ROS is an optional observability boundary: failures here must never take control
away from the navigation safety loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Path
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
        from std_msgs.msg import String
        from vision_msgs.msg import Detection2DArray

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
            "Twist": Twist,
            "Path": Path,
            "CameraInfo": CameraInfo,
            "Image": Image,
            "Imu": Imu,
            "PointCloud2": PointCloud2,
            "PointField": PointField,
            "String": String,
            "Detection2DArray": Detection2DArray,
        }
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        self._publishers = {
            "rgb": self._node.create_publisher(Image, self._topics.rgb, sensor_qos),
            "depth": self._node.create_publisher(Image, self._topics.depth, sensor_qos),
            "camera_info": self._node.create_publisher(CameraInfo, self._topics.camera_info, 1),
            "imu": self._node.create_publisher(Imu, self._topics.imu, sensor_qos),
            "pose": self._node.create_publisher(PoseStamped, self._topics.pose, 10),
            "path": self._node.create_publisher(Path, self._topics.path, 1),
            "map_points": self._node.create_publisher(PointCloud2, self._topics.map_points, sensor_qos),
            "tracking": self._node.create_publisher(String, self._topics.tracking, 10),
            "detections": self._node.create_publisher(Detection2DArray, self._topics.detections, sensor_qos),
            "status": self._node.create_publisher(String, self._topics.status, 10),
            "command": self._node.create_publisher(Twist, self._topics.command, 10),
            "reference_pose": self._node.create_publisher(PoseStamped, self._topics.reference_pose, 10),
            "reference_path": self._node.create_publisher(Path, self._topics.reference_path, 1),
            "reference_map": self._node.create_publisher(PointCloud2, self._topics.reference_map, sensor_qos),
        }
        self._path = Path()
        self._path.header.frame_id = map_frame
        self._reference_path = Path()
        self._reference_path.header.frame_id = map_frame
        self._last_sensor_timestamp: float | None = None

    @classmethod
    def try_create(cls, *, enabled: bool = True, **kwargs: Any) -> "RosObservabilityPublisher | None":
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
            info_msg = self._types["CameraInfo"]()
            self._header(info_msg, timestamp, self._camera_frame)
            info_msg.width, info_msg.height = int(info.width), int(info.height)
            info_msg.k = np.asarray(info.k, dtype=float).reshape(-1).tolist()
            info_msg.d = list(getattr(info, "d", []))
            info_msg.r = np.eye(3).reshape(-1).tolist()
            info_msg.p = np.hstack(
                (np.asarray(info.k).reshape(3, 3), np.zeros((3, 1)))
            ).reshape(-1).tolist()
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
            imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = state.angular_velocity
            imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = state.linear_acceleration
            self._publishers["imu"].publish(imu_msg)

    def publish_policy(self, status: Any, timestamp: float) -> None:
        state_name = getattr(getattr(status, "state", None), "value", "unknown")
        status_message = self._types["String"]()
        status_message.data = json.dumps(
            {
                "state": state_name,
                "message": getattr(status, "message", ""),
                "goal": getattr(getattr(status, "current_goal", None), "object_name", None),
                "tracking_confidence": getattr(getattr(status, "current_pose", None), "confidence", None),
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

        pose = getattr(status, "current_pose", None)
        if pose is not None:
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler("z", float(pose.yaw)).as_matrix()
            matrix[:3, 3] = np.asarray(pose.position, dtype=float)
            self.publish_pose(matrix, timestamp)
        self.publish_detections(getattr(status, "detections", []), timestamp)

    def publish_pose(self, matrix: np.ndarray, timestamp: float, *, reference: bool = False) -> None:
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
        message = self._types["Detection2DArray"]()
        self._header(message, timestamp, self._camera_frame)
        try:
            from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
        except ImportError:
            return
        for detection in detections:
            item = Detection2D()
            self._header(item, timestamp, self._camera_frame)
            x0, y0, x1, y1 = detection.bbox
            center = item.bbox.center
            center_position = getattr(center, "position", center)
            center_position.x = (x0 + x1) / 2.0
            center_position.y = (y0 + y1) / 2.0
            item.bbox.size_x = float(x1 - x0)
            item.bbox.size_y = float(y1 - y0)
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = str(detection.label)
            result.hypothesis.score = float(detection.confidence)
            if detection.position_3d is not None:
                position = np.asarray(detection.position_3d, dtype=float)
                result.pose.pose.position.x, result.pose.pose.position.y, result.pose.pose.position.z = position
            item.results.append(result)
            message.detections.append(item)
        self._publishers["detections"].publish(message)

    def publish_point_cloud(self, points: np.ndarray, timestamp: float, *, reference: bool = False) -> None:
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
