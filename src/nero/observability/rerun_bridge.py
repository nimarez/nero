"""ROS 2 to Rerun bridge for Nero's normalized observability topics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from nero.slam.k1_calibration import K1Calibration

from .topics import ObservabilityTopics

logger = logging.getLogger(__name__)


def image_message_to_array(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding in {"bgr8", "rgb8"}:
        array = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, 3)
        return array[..., ::-1].copy() if encoding == "bgr8" else array.copy()
    if encoding in {"mono8", "8uc1"}:
        return np.frombuffer(message.data, np.uint8).reshape(message.height, message.width).copy()
    if encoding in {"mono16", "16uc1"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        return np.frombuffer(message.data, dtype).reshape(message.height, message.width).astype(np.uint16)
    if encoding == "32fc1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        return np.frombuffer(message.data, dtype).reshape(message.height, message.width).astype(np.float32)
    raise ValueError(f"unsupported ROS image encoding: {message.encoding}")


def pointcloud2_to_xyz(message: Any) -> np.ndarray:
    offsets = {field.name: int(field.offset) for field in message.fields}
    if not {"x", "y", "z"}.issubset(offsets):
        return np.empty((0, 3), dtype=np.float32)
    byteorder = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [byteorder + "f4"] * 3,
            "offsets": [offsets[axis] for axis in ("x", "y", "z")],
            "itemsize": int(message.point_step),
        }
    )
    count = int(message.width) * int(message.height)
    values = np.frombuffer(message.data, dtype=dtype, count=count)
    points = np.column_stack((values["x"], values["y"], values["z"])).astype(np.float32)
    return points[np.all(np.isfinite(points), axis=1)]


def _timestamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class RerunRosBridge:
    def __init__(
        self,
        recording: Any,
        *,
        topics: ObservabilityTopics | None = None,
        body_to_camera: np.ndarray | None = None,
    ) -> None:
        import rerun as rr
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Path as RosPath
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
        from std_msgs.msg import String
        from vision_msgs.msg import Detection2DArray

        self._rr = rr
        self._recording = recording
        self._topics = topics or ObservabilityTopics()
        self._node = rclpy.create_node("nero_rerun_bridge")
        self._subscriptions = []
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        subscribe = self._node.create_subscription
        self._subscriptions.extend(
            [
                subscribe(Image, self._topics.rgb, self._on_rgb, sensor_qos),
                subscribe(Image, self._topics.depth, self._on_depth, sensor_qos),
                subscribe(CameraInfo, self._topics.camera_info, self._on_camera_info, 1),
                subscribe(Imu, self._topics.imu, self._on_imu, sensor_qos),
                subscribe(PoseStamped, self._topics.pose, self._on_pose, 10),
                subscribe(RosPath, self._topics.path, self._on_path, 1),
                subscribe(PointCloud2, self._topics.map_points, self._on_map, sensor_qos),
                subscribe(String, self._topics.tracking, self._on_tracking, 10),
                subscribe(Detection2DArray, self._topics.detections, self._on_detections, sensor_qos),
                subscribe(String, self._topics.status, self._on_status, 10),
                subscribe(Twist, self._topics.command, self._on_command, 10),
                subscribe(PoseStamped, self._topics.reference_pose, self._on_reference_pose, 10),
                subscribe(RosPath, self._topics.reference_path, self._on_reference_path, 1),
                subscribe(PointCloud2, self._topics.reference_map, self._on_reference_map, sensor_qos),
            ]
        )
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        transform = np.eye(4) if body_to_camera is None else np.asarray(body_to_camera)
        if transform.shape != (4, 4):
            raise ValueError("body_to_camera must be a 4x4 transform")
        mount_rotation = Rotation.from_matrix(transform[:3, :3]).as_quat()
        recording.log(
            "world/robot/camera",
            rr.Transform3D(
                translation=transform[:3, 3], quaternion=mount_rotation
            ),
            static=True,
        )

    @property
    def node(self) -> Any:
        return self._node

    def _time(self, message: Any) -> None:
        self._recording.set_time_nanos("ros_time", _timestamp_ns(message))

    def _receipt_time(self) -> None:
        """Timestamp headerless ROS messages on receipt."""
        self._recording.set_time_nanos("wall_time", time.time_ns())

    def _on_rgb(self, message: Any) -> None:
        self._time(message)
        self._recording.log("world/robot/camera/rgb", self._rr.Image(image_message_to_array(message)))

    def _on_depth(self, message: Any) -> None:
        self._time(message)
        depth = image_message_to_array(message)
        meter = 1000.0 if depth.dtype == np.uint16 else 1.0
        self._recording.log("world/robot/camera/depth", self._rr.DepthImage(depth, meter=meter))

    def _on_camera_info(self, message: Any) -> None:
        self._time(message)
        self._recording.log(
            "world/robot/camera",
            self._rr.Pinhole(
                resolution=[message.width, message.height],
                image_from_camera=np.asarray(message.k, dtype=float).reshape(3, 3),
                camera_xyz=self._rr.ViewCoordinates.RDF,
                image_plane_distance=0.25,
            ),
        )

    def _on_imu(self, message: Any) -> None:
        self._time(message)
        for axis in ("x", "y", "z"):
            self._recording.log(
                f"metrics/imu/angular_velocity/{axis}",
                self._rr.Scalar(float(getattr(message.angular_velocity, axis))),
            )
            self._recording.log(
                f"metrics/imu/linear_acceleration/{axis}",
                self._rr.Scalar(float(getattr(message.linear_acceleration, axis))),
            )

    def _log_pose(self, message: Any, entity: str) -> None:
        self._time(message)
        position = message.pose.position
        orientation = message.pose.orientation
        self._recording.log(
            entity,
            self._rr.Transform3D(
                translation=[position.x, position.y, position.z],
                quaternion=[orientation.x, orientation.y, orientation.z, orientation.w],
            ),
        )

    def _on_pose(self, message: Any) -> None:
        self._log_pose(message, "world/robot")

    def _on_reference_pose(self, message: Any) -> None:
        self._log_pose(message, "world/reference_robot")

    def _log_path(self, message: Any, entity: str, color: list[int]) -> None:
        self._time(message)
        points = [
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in message.poses
        ]
        if len(points) >= 2:
            self._recording.log(entity, self._rr.LineStrips3D([points], colors=color, radii=0.01))

    def _on_path(self, message: Any) -> None:
        self._log_path(message, "world/slam/trajectory", [0, 200, 255])

    def _on_reference_path(self, message: Any) -> None:
        self._log_path(message, "world/reference/trajectory", [255, 180, 0])

    def _log_map(self, message: Any, entity: str, color: list[int]) -> None:
        self._time(message)
        points = pointcloud2_to_xyz(message)
        self._recording.log(entity, self._rr.Points3D(points, colors=color, radii=0.01))

    def _on_map(self, message: Any) -> None:
        self._log_map(message, "world/slam/map_points", [0, 200, 255])

    def _on_reference_map(self, message: Any) -> None:
        self._log_map(message, "world/reference/map_points", [255, 180, 0])

    def _on_detections(self, message: Any) -> None:
        self._time(message)
        centers, sizes, labels = [], [], []
        for detection in message.detections:
            center = detection.bbox.center
            position = getattr(center, "position", center)
            centers.append([position.x, position.y])
            sizes.append([detection.bbox.size_x, detection.bbox.size_y])
            result = detection.results[0] if detection.results else None
            labels.append(result.hypothesis.class_id if result is not None else "object")
        self._recording.log(
            "world/robot/camera/rgb/detections",
            self._rr.Boxes2D(centers=centers, sizes=sizes, labels=labels),
        )

    def _on_status(self, message: Any) -> None:
        self._receipt_time()
        try:
            data = json.loads(message.data)
            text = f"{data.get('state', 'unknown')}: {data.get('message', '')}"
        except json.JSONDecodeError:
            text = message.data
        self._recording.log("status/navigation", self._rr.TextLog(text))

    def _on_tracking(self, message: Any) -> None:
        self._receipt_time()
        try:
            data = json.loads(message.data)
        except json.JSONDecodeError:
            data = {"status": message.data}
        self._recording.log("status/tracking", self._rr.TextLog(str(data.get("status", "unknown"))))
        if "map_points" in data:
            self._recording.log("metrics/slam/map_points", self._rr.Scalar(float(data["map_points"])))

    def _on_command(self, message: Any) -> None:
        self._receipt_time()
        values = {
            "linear_x": message.linear.x,
            "linear_y": message.linear.y,
            "angular_z": message.angular.z,
        }
        for name, value in values.items():
            self._recording.log(f"metrics/command/{name}", self._rr.Scalar(float(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Nero ROS 2 topics in Rerun")
    sink = parser.add_mutually_exclusive_group()
    sink.add_argument(
        "--connect",
        default=os.environ.get("NERO_RERUN_URL"),
        help="Rerun viewer address, e.g. host.docker.internal:9876",
    )
    sink.add_argument("--save", type=Path, help="Write a .rrd recording instead of streaming")
    sink.add_argument("--spawn", action="store_true", help="Spawn a local native viewer")
    parser.add_argument(
        "--sensor-calibration",
        type=Path,
        default=Path("config/k1_calibration.json"),
        help="K1 calibration used for the body-to-camera transform",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        import rerun as rr
        import rclpy
        from rclpy.executors import ExternalShutdownException
    except ImportError as exc:
        raise SystemExit("Install the visualization extra with: uv sync --extra viz") from exc

    rclpy.init(args=None)
    recording = rr.new_recording("nero_k1", make_default=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(args.save))
    elif args.spawn:
        recording.spawn()
    else:
        recording.connect(args.connect or "127.0.0.1:9876")
    calibration_path = args.sensor_calibration
    if not calibration_path.is_file():
        calibration_path = Path("config/k1_geek_nominal_calibration.json")
    calibration = K1Calibration.load(calibration_path)
    bridge = RerunRosBridge(recording, body_to_camera=np.asarray(calibration.tbc))
    logger.info("Rerun bridge subscribed to /nero/*")
    try:
        rclpy.spin(bridge.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        bridge.node.destroy_node()
        recording.flush()
        recording.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
