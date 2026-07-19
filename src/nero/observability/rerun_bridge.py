"""ROS 2 to Rerun bridge for Nero's normalized observability topics."""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import numpy as np
from scipy.spatial.transform import Rotation

from nero.slam.k1_calibration import K1Calibration

from .topics import ObservabilityTopics

logger = logging.getLogger(__name__)


def _host_for_url(host_header: str) -> str:
    """Return the request hostname in URL-safe form without trusting its port."""
    hostname = urlsplit(f"//{host_header}").hostname or "127.0.0.1"
    return f"[{hostname}]" if ":" in hostname else hostname


def start_web_gateway(
    port: int,
    *,
    viewer_port: int,
    websocket_port: int,
    path: str = "/rerun",
) -> http.server.ThreadingHTTPServer:
    """Expose Rerun's root-only HTTP server below a stable robot URL path."""
    normalized_path = "/" + path.strip("/")

    class GatewayHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            raw_path = self.path.partition("?")[0]
            request_path = raw_path.rstrip("/") or "/"
            if request_path == "/healthz":
                payload = b'{"status":"ok"}\n'
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if request_path == "/":
                self.send_response(http.HTTPStatus.TEMPORARY_REDIRECT)
                self.send_header("Location", normalized_path)
                self.end_headers()
                return
            if request_path != normalized_path and not raw_path.startswith(f"{normalized_path}/"):
                self.send_error(http.HTTPStatus.NOT_FOUND)
                return

            host = _host_for_url(self.headers.get("Host", "127.0.0.1"))
            if raw_path == normalized_path:
                websocket_url = quote(f"ws://{host}:{websocket_port}", safe="")
                viewer_url = f"{normalized_path}/?url={websocket_url}"
                self.send_response(http.HTTPStatus.TEMPORARY_REDIRECT)
                self.send_header("Location", viewer_url)
                self.end_headers()
                return

            upstream_path = self.path[len(normalized_path) :] or "/"
            connection = http.client.HTTPConnection("127.0.0.1", viewer_port, timeout=5)
            try:
                connection.request("GET", upstream_path)
                response = connection.getresponse()
                payload = response.read()
            except OSError as exc:
                logger.error("Rerun web viewer proxy failed: %s", exc)
                self.send_error(http.HTTPStatus.BAD_GATEWAY)
                return
            finally:
                connection.close()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in {"connection", "transfer-encoding"}:
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("Rerun web gateway: " + format, *args)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, name="nero-rerun-web", daemon=True)
    thread.start()
    return server


def create_default_blueprint(rrb: Any) -> Any:
    """Create a deterministic dashboard with the live camera visible immediately."""
    camera = "world/robot/camera"
    return rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(
                origin=camera,
                contents=[
                    f"{camera}/rgb",
                    f"{camera}/rgb/detections",
                    f"{camera}/rgb/obstacles",
                ],
                name="K1 RGB",
            ),
            rrb.Spatial2DView(
                origin=camera,
                contents=[f"{camera}/depth"],
                name="K1 Depth",
            ),
            rrb.Spatial3DView(origin="world", name="Robot, SLAM, and Goals"),
            rrb.TimeSeriesView(origin="metrics", name="Sensors and Commands"),
            rrb.TextLogView(origin="status", name="Safety and State"),
            grid_columns=2,
        ),
        auto_views=False,
        collapse_panels=False,
    )


def image_message_to_array(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding in {"bgr8", "rgb8"}:
        array = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, 3)
        return array[..., ::-1].copy() if encoding == "bgr8" else array.copy()
    if encoding in {"mono8", "8uc1"}:
        return np.frombuffer(message.data, np.uint8).reshape(message.height, message.width).copy()
    if encoding in {"mono16", "16uc1"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        return (
            np.frombuffer(message.data, dtype)
            .reshape(message.height, message.width)
            .astype(np.uint16)
        )
    if encoding == "32fc1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        return (
            np.frombuffer(message.data, dtype)
            .reshape(message.height, message.width)
            .astype(np.float32)
        )
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


def navigation_geometry_primitives(payload: Any, *, segments: int = 64) -> dict[str, Any] | None:
    """Build frame-correct circle, bearing, and stand-off waypoint geometry."""
    if not isinstance(payload, dict) or segments < 8:
        return None
    try:
        frame = str(payload["frame"])
        center = np.asarray(payload["center"], dtype=float).reshape(3)
        radius = float(payload["radius"])
        tolerance = max(0.0, float(payload.get("tolerance", 0.12)))
        robot = np.asarray(payload.get("robot", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        frame not in {"map", "camera"}
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(robot))
        or not np.isfinite(radius)
        or radius <= 0
    ):
        return None

    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    if frame == "map":
        ground_z = float(payload.get("ground_z", robot[2] + 0.03))
        circle = np.column_stack(
            (
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full_like(angles, ground_z),
            )
        )
        distance = float(np.linalg.norm((center - robot)[:2]))
        approach = payload.get("approach")
        try:
            approach = None if approach is None else np.asarray(approach, dtype=float).reshape(3)
        except (TypeError, ValueError):
            approach = None
        if approach is not None:
            approach[2] = ground_z
        root = "world/navigation/safety_geometry"
    else:
        circle = np.column_stack(
            (
                center[0] + radius * np.cos(angles),
                np.full_like(angles, center[1]),
                center[2] + radius * np.sin(angles),
            )
        )
        distance = float(np.linalg.norm(center[[0, 2]]))
        scale = max(0.0, (distance - radius) / distance) if distance else 0.0
        approach = center * scale
        root = "world/robot/camera/navigation/safety_geometry"

    if distance < radius - tolerance:
        condition, color = "inside radius", [255, 60, 60]
    elif distance <= radius + tolerance:
        condition, color = "holding radius", [80, 255, 80]
    else:
        condition, color = "approaching", [0, 200, 255]
    if approach is not None and not np.all(np.isfinite(approach)):
        approach = None
    return {
        "root": root,
        "circle": circle,
        "bearing": np.asarray([robot, center]),
        "approach": approach,
        "color": color,
        "condition": condition,
        "distance": distance,
        "radius": radius,
        "frame": frame,
        "target": payload.get("target"),
        "waypoint": payload.get("waypoint"),
    }


def predicted_swept_clearance(
    linear_x: float,
    linear_y: float,
    angular_z: float,
    clearance_radius: float,
    *,
    horizon: float = 2.0,
    steps: int = 20,
    circle_segments: int = 32,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Integrate a planar body command and return its swept clearance rings."""
    if steps < 1 or circle_segments < 8 or horizon <= 0 or clearance_radius <= 0:
        raise ValueError("swept-clearance dimensions must be positive")
    dt = horizon / steps
    pose = np.zeros(3, dtype=float)
    centers = [pose[:2].copy()]
    for _ in range(steps):
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        pose[0] += (cosine * linear_x - sine * linear_y) * dt
        pose[1] += (sine * linear_x + cosine * linear_y) * dt
        pose[2] += angular_z * dt
        centers.append(pose[:2].copy())
    centers_3d = np.column_stack((np.asarray(centers), np.full(len(centers), 0.03)))
    angles = np.linspace(0.0, 2.0 * np.pi, circle_segments + 1)
    rings = []
    for center in centers_3d[:: max(1, steps // 6)]:
        rings.append(
            np.column_stack(
                (
                    center[0] + clearance_radius * np.cos(angles),
                    center[1] + clearance_radius * np.sin(angles),
                    np.full_like(angles, center[2]),
                )
            )
        )
    if not np.array_equal(rings[-1][0, :2], centers_3d[-1, :2] + [clearance_radius, 0]):
        center = centers_3d[-1]
        rings.append(
            np.column_stack(
                (
                    center[0] + clearance_radius * np.cos(angles),
                    center[1] + clearance_radius * np.sin(angles),
                    np.full_like(angles, center[2]),
                )
            )
        )
    return centers_3d, rings


def occupancy_grid_points(message: Any) -> dict[str, np.ndarray]:
    """Convert ROS occupancy values into world-frame cell-center point sets."""
    width, height = int(message.info.width), int(message.info.height)
    values = np.asarray(message.data, dtype=np.int16).reshape(height, width)
    origin = message.info.origin.position
    resolution = float(message.info.resolution)
    result = {}
    for name, selected in (
        ("occupied", values >= 100),
        ("inflated", (values > 0) & (values < 100)),
    ):
        ys, xs = np.where(selected)
        result[name] = np.column_stack(
            (
                float(origin.x) + (xs + 0.5) * resolution,
                float(origin.y) + (ys + 0.5) * resolution,
                np.full(len(xs), 0.015),
            )
        )
    return result


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
        from geometry_msgs.msg import PointStamped, PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
        from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2
        from std_msgs.msg import String

        self._rr = rr
        self._recording = recording
        self._topics = topics or ObservabilityTopics()
        self._node = rclpy.create_node("nero_rerun_bridge")
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        subscribe = self._node.create_subscription
        specs = [
            (Image, self._topics.rgb, self._on_rgb, sensor_qos),
            (Image, self._topics.depth, self._on_depth, sensor_qos),
            (CameraInfo, self._topics.camera_info, self._on_camera_info, 1),
            (Imu, self._topics.imu, self._on_imu, sensor_qos),
            (Odometry, self._topics.odometry, self._on_odometry, sensor_qos),
            (JointState, self._topics.joint_states, self._on_joint_states, sensor_qos),
            (PoseStamped, self._topics.pose, self._on_pose, 10),
            (RosPath, self._topics.path, self._on_path, 1),
            (PointCloud2, self._topics.map_points, self._on_map, sensor_qos),
            (String, self._topics.tracking, self._on_tracking, 10),
            (String, self._topics.status, self._on_status, 10),
            (Twist, self._topics.command, self._on_command, 10),
            (RosPath, self._topics.plan, self._on_plan, 1),
            (PoseStamped, self._topics.goal_pose, self._on_goal_pose, 10),
            (PointStamped, self._topics.object_position, self._on_object_position, 10),
            (PoseStamped, self._topics.reference_pose, self._on_reference_pose, 10),
            (RosPath, self._topics.reference_path, self._on_reference_path, 1),
            (PointCloud2, self._topics.reference_map, self._on_reference_map, sensor_qos),
            (String, self._topics.detections, self._on_detections, sensor_qos),
            (Image, self._topics.obstacle_mask, self._on_obstacle_mask, sensor_qos),
            (
                PointCloud2,
                self._topics.obstacle_points,
                self._on_obstacle_points,
                sensor_qos,
            ),
            (OccupancyGrid, self._topics.occupancy_grid, self._on_occupancy_grid, 1),
        ]
        self._subscription_topics = tuple(spec[1] for spec in specs)
        expected_topics = set(vars(self._topics).values())
        if set(self._subscription_topics) != expected_topics:
            missing = expected_topics - set(self._subscription_topics)
            extra = set(self._subscription_topics) - expected_topics
            raise RuntimeError(
                f"Rerun topic contract mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self._subscriptions = [subscribe(*spec) for spec in specs]
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        transform = np.eye(4) if body_to_camera is None else np.asarray(body_to_camera)
        if transform.shape != (4, 4):
            raise ValueError("body_to_camera must be a 4x4 transform")
        mount_rotation = Rotation.from_matrix(transform[:3, :3]).as_quat()
        recording.log(
            "world/robot/camera",
            rr.Transform3D(translation=transform[:3, 3], quaternion=mount_rotation),
            static=True,
        )
        self._target_history: list[np.ndarray] = []
        self._target_history_key: tuple[str, str | None] | None = None
        self._safety_clearance_radius = 0.25

    @property
    def node(self) -> Any:
        return self._node

    @property
    def subscription_topics(self) -> tuple[str, ...]:
        return self._subscription_topics

    def _time(self, message: Any) -> None:
        self._recording.set_time_nanos("ros_time", _timestamp_ns(message))

    def _receipt_time(self) -> None:
        """Timestamp headerless ROS messages on receipt."""
        self._recording.set_time_nanos("wall_time", time.time_ns())

    def _on_rgb(self, message: Any) -> None:
        self._time(message)
        self._recording.log(
            "world/robot/camera/rgb", self._rr.Image(image_message_to_array(message))
        )

    def _on_depth(self, message: Any) -> None:
        self._time(message)
        depth = image_message_to_array(message)
        meter = 1000.0 if depth.dtype == np.uint16 else 1.0
        self._recording.log("world/robot/camera/depth", self._rr.DepthImage(depth, meter=meter))

    def _on_obstacle_mask(self, message: Any) -> None:
        self._time(message)
        self._recording.log(
            "world/robot/camera/rgb/obstacles",
            self._rr.SegmentationImage(image_message_to_array(message)),
        )

    def _on_obstacle_points(self, message: Any) -> None:
        self._time(message)
        self._recording.log(
            "world/robot/camera/obstacles",
            self._rr.Points3D(pointcloud2_to_xyz(message), colors=[255, 80, 40], radii=0.012),
        )

    def _on_occupancy_grid(self, message: Any) -> None:
        self._time(message)
        groups = occupancy_grid_points(message)
        radius = max(0.005, float(message.info.resolution) * 0.45)
        for name, color in (("occupied", [30, 30, 30]), ("inflated", [255, 140, 20])):
            self._recording.log(
                f"world/navigation/map/{name}",
                self._rr.Points3D(groups[name], colors=color, radii=radius),
            )

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
        orientation = message.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        if np.linalg.norm(quaternion) > 0:
            roll, pitch, yaw = Rotation.from_quat(quaternion).as_euler("xyz")
            for name, value in zip(("roll", "pitch", "yaw"), (roll, pitch, yaw)):
                self._recording.log(
                    f"metrics/imu/orientation/{name}", self._rr.Scalar(float(value))
                )

    def _on_odometry(self, message: Any) -> None:
        self._time(message)
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        yaw = (
            Rotation.from_quat(quaternion).as_euler("xyz")[2]
            if np.linalg.norm(quaternion) > 0
            else 0.0
        )
        for name, value in (("x", position.x), ("y", position.y), ("yaw", yaw)):
            self._recording.log(f"metrics/odometry/{name}", self._rr.Scalar(float(value)))

    def _on_joint_states(self, message: Any) -> None:
        self._time(message)
        names = list(message.name)
        for field in ("position", "velocity", "effort"):
            values = list(getattr(message, field, []))
            for index, value in enumerate(values):
                name = names[index] if index < len(names) else f"joint_{index}"
                safe_name = str(name).replace("/", "_").replace(" ", "_")
                self._recording.log(
                    f"metrics/joints/{safe_name}/{field}",
                    self._rr.Scalar(float(value)),
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

    def _on_goal_pose(self, message: Any) -> None:
        self._log_pose(message, "world/navigation/goal")

    def _on_object_position(self, message: Any) -> None:
        self._time(message)
        point = message.point
        self._recording.log(
            "world/navigation/object",
            self._rr.Points3D([[point.x, point.y, point.z]], colors=[255, 80, 80], radii=0.08),
        )

    def _log_path(self, message: Any, entity: str, color: list[int]) -> None:
        self._time(message)
        points = [
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in message.poses
        ]
        if len(points) >= 2:
            self._recording.log(entity, self._rr.LineStrips3D([points], colors=color, radii=0.01))
        else:
            self._recording.log(entity, self._rr.Clear(recursive=False))

    def _on_path(self, message: Any) -> None:
        self._log_path(message, "world/slam/trajectory", [0, 200, 255])

    def _on_plan(self, message: Any) -> None:
        self._log_path(message, "world/navigation/plan", [80, 255, 80])

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
        try:
            payload = json.loads(message.data)
            timestamp = float(payload.get("timestamp", time.time()))
            items = payload.get("detections", [])
            if not isinstance(items, list):
                raise ValueError("detections must be a list")
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Ignoring malformed detection telemetry")
            return
        self._recording.set_time_nanos("ros_time", int(timestamp * 1_000_000_000))
        centers, sizes, labels, points, point_labels = [], [], [], [], []
        for detection in items:
            try:
                x0, y0, x1, y1 = (float(value) for value in detection["bbox"])
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("invalid detection bounds")
                label = str(detection.get("label", "object"))
                confidence = float(detection.get("confidence", 0.0))
                distance = float(detection.get("distance", 0.0))
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed detection item")
                continue
            centers.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            sizes.append([x1 - x0, y1 - y0])
            labels.append(f"{label} {confidence:.2f} {distance:.2f}m")
            position = detection.get("position_3d")
            if (
                isinstance(position, list)
                and len(position) == 3
                and detection.get("coordinate_frame") == "camera"
            ):
                points.append(position)
                point_labels.append(label)
        self._recording.log(
            "world/robot/camera/rgb/detections",
            self._rr.Boxes2D(centers=centers, sizes=sizes, labels=labels),
        )
        self._recording.log(
            "world/robot/camera/detection_centroids",
            self._rr.Points3D(
                points,
                labels=point_labels,
                colors=[255, 80, 80],
                radii=0.04,
            ),
        )

    def _on_status(self, message: Any) -> None:
        self._receipt_time()
        data = None
        try:
            data = json.loads(message.data)
            text = f"{data.get('state', 'unknown')}: {data.get('message', '')}"
        except (AttributeError, json.JSONDecodeError, TypeError):
            text = str(getattr(message, "data", message))
        self._recording.log("status/navigation", self._rr.TextLog(text))
        self._log_navigation_geometry(
            data.get("navigation_geometry") if isinstance(data, dict) else None
        )
        self._log_safety(data.get("safety") if isinstance(data, dict) else None)
        self._log_detector(data.get("detector") if isinstance(data, dict) else None)

    def _log_navigation_geometry(self, payload: Any) -> None:
        world_root = "world/navigation/safety_geometry"
        camera_root = "world/robot/camera/navigation/safety_geometry"
        geometry = navigation_geometry_primitives(payload)
        for root in (world_root, camera_root):
            self._recording.log(root, self._rr.Clear(recursive=True))
        if geometry is None:
            return
        root = geometry["root"]
        color = geometry["color"]
        self._recording.log(
            f"{root}/radius",
            self._rr.LineStrips3D([geometry["circle"]], colors=color, radii=0.015),
        )
        self._recording.log(
            f"{root}/target_bearing",
            self._rr.LineStrips3D([geometry["bearing"]], colors=[255, 210, 40], radii=0.01),
        )
        if geometry["approach"] is not None:
            self._recording.log(
                f"{root}/stand_off_waypoint",
                self._rr.Points3D(
                    [geometry["approach"]],
                    colors=color,
                    radii=0.07,
                    labels=[geometry["condition"]],
                ),
            )
        waypoint = geometry.get("waypoint")
        if waypoint is not None:
            try:
                waypoint = np.asarray(waypoint, dtype=float).reshape(3)
            except (TypeError, ValueError):
                waypoint = None
        if waypoint is not None and np.all(np.isfinite(waypoint)):
            self._recording.log(
                f"{root}/active_planner_waypoint",
                self._rr.Points3D(
                    [waypoint], colors=[190, 90, 255], radii=0.08, labels=["active waypoint"]
                ),
            )
        history_key = (geometry["frame"], geometry.get("target"))
        if history_key != getattr(self, "_target_history_key", None):
            self._target_history = []
            self._target_history_key = history_key
            self._recording.log(f"{root}/target_history", self._rr.Clear(recursive=False))
        center = np.asarray(payload["center"], dtype=float)
        if not self._target_history or np.linalg.norm(center - self._target_history[-1]) > 0.01:
            self._target_history.append(center)
            self._target_history = self._target_history[-100:]
        if len(self._target_history) >= 2:
            self._recording.log(
                f"{root}/target_history",
                self._rr.LineStrips3D([self._target_history], colors=[255, 80, 180], radii=0.008),
            )
        self._recording.log(
            "metrics/navigation/target_range",
            self._rr.Scalar(geometry["distance"]),
        )
        self._recording.log(
            "metrics/navigation/stand_off_radius",
            self._rr.Scalar(geometry["radius"]),
        )

    def _log_safety(self, payload: Any) -> None:
        root = "world/robot/safety"
        self._recording.log(root, self._rr.Clear(recursive=True))
        if not isinstance(payload, dict):
            return
        is_safe = bool(payload.get("is_safe", True))
        enforced = bool(payload.get("enforced", True))
        emergency = bool(payload.get("emergency_stop", False))
        sensor_blind = bool(payload.get("depth_sensor_blind", False))
        warnings = [str(value) for value in payload.get("warnings", [])]
        reason = str(payload.get("reason", ""))
        if not enforced:
            color, condition = [210, 80, 255], "BYPASSED"
        elif emergency or not is_safe:
            color, condition = [255, 60, 60], "E-STOP"
        elif sensor_blind or warnings:
            color, condition = [255, 170, 40], "WARNING"
        else:
            color, condition = [80, 255, 80], "SAFE"

        limit = payload.get("min_obstacle_distance")
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            limit = 0.25
        if not np.isfinite(limit) or limit <= 0:
            limit = 0.25
        self._safety_clearance_radius = limit
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        ring = np.column_stack(
            (
                limit * np.cos(angles),
                limit * np.sin(angles),
                np.full_like(angles, 0.03),
            )
        )
        clearance = payload.get("obstacle_distance")
        clearance_text = "unknown" if clearance is None else f"{float(clearance):.2f}m"
        label = f"{condition} | clearance {clearance_text}"
        if reason:
            label += f" | {reason}"
        self._recording.log(
            f"{root}/clearance_limit",
            self._rr.LineStrips3D([ring], colors=color, radii=0.018),
        )
        self._recording.log(
            f"{root}/state",
            self._rr.Points3D(
                [[0.0, 0.0, 0.12]],
                colors=color,
                radii=0.055,
                labels=[label],
                show_labels=True,
            ),
        )
        safety_text = label
        if warnings:
            safety_text += " | " + "; ".join(warnings)
        if safety_text != getattr(self, "_last_safety_text", None):
            self._recording.log("status/safety", self._rr.TextLog(safety_text))
            self._last_safety_text = safety_text

        scalar_values = {
            "safe": float(is_safe),
            "enforced": float(enforced),
            "emergency_stop": float(emergency),
            "depth_sensor_blind": float(sensor_blind),
            "has_obstacle": float(bool(payload.get("has_obstacle", False))),
            "left_clear": float(bool(payload.get("left_clear", True))),
            "center_clear": float(bool(payload.get("center_clear", True))),
            "right_clear": float(bool(payload.get("right_clear", True))),
        }
        optional_values = {
            "obstacle_distance_m": payload.get("obstacle_distance"),
            "battery_percent": payload.get("battery_percent"),
            "roll_degrees": (
                None if payload.get("roll_rad") is None else np.degrees(float(payload["roll_rad"]))
            ),
            "pitch_degrees": (
                None
                if payload.get("pitch_rad") is None
                else np.degrees(float(payload["pitch_rad"]))
            ),
        }
        for name, value in optional_values.items():
            if value is not None and np.isfinite(float(value)):
                scalar_values[name] = float(value)
        for name, value in scalar_values.items():
            self._recording.log(f"metrics/safety/{name}", self._rr.Scalar(value))

    def _log_detector(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for name in (
            "inference_ms_ema",
            "inference_fps",
            "result_age_seconds",
            "result_revision",
            "confidence_threshold",
        ):
            value = payload.get(name)
            if value is not None and np.isfinite(float(value)):
                self._recording.log(f"metrics/detector/{name}", self._rr.Scalar(float(value)))
        summary = f"{payload.get('backend', 'unknown')} | target={payload.get('target') or 'none'}"
        if summary != getattr(self, "_last_detector_text", None):
            self._recording.log("status/detector", self._rr.TextLog(summary))
            self._last_detector_text = summary

    def _on_tracking(self, message: Any) -> None:
        self._receipt_time()
        try:
            data = json.loads(message.data)
        except json.JSONDecodeError:
            data = {"status": message.data}
        self._recording.log("status/tracking", self._rr.TextLog(str(data.get("status", "unknown"))))
        if "map_points" in data:
            self._recording.log(
                "metrics/slam/map_points", self._rr.Scalar(float(data["map_points"]))
            )

    def _on_command(self, message: Any) -> None:
        self._receipt_time()
        values = {
            "linear_x": message.linear.x,
            "linear_y": message.linear.y,
            "angular_z": message.angular.z,
        }
        for name, value in values.items():
            self._recording.log(f"metrics/command/{name}", self._rr.Scalar(float(value)))
        root = "world/robot/command_preview"
        self._recording.log(root, self._rr.Clear(recursive=True))
        linear = np.asarray([message.linear.x, message.linear.y, 0.0], dtype=float)
        speed = float(np.linalg.norm(linear[:2]))
        if speed > 1e-4:
            self._recording.log(
                f"{root}/linear_2s",
                self._rr.Arrows3D(
                    origins=[[0.0, 0.0, 0.08]],
                    vectors=[linear * 2.0],
                    colors=[80, 255, 80],
                    radii=0.015,
                    labels=[f"{speed:.2f} m/s"],
                ),
            )
        yaw_rate = float(message.angular.z)
        if abs(yaw_rate) > 1e-4:
            angles = np.linspace(0.0, yaw_rate * 2.0, 24)
            radius = 0.3
            arc = np.column_stack(
                (
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    np.full_like(angles, 0.08),
                )
            )
            self._recording.log(
                f"{root}/yaw_2s",
                self._rr.LineStrips3D([arc], colors=[255, 170, 40], radii=0.012),
            )
        centers, rings = predicted_swept_clearance(
            float(message.linear.x),
            float(message.linear.y),
            yaw_rate,
            float(getattr(self, "_safety_clearance_radius", 0.25)),
        )
        self._recording.log(
            f"{root}/swept_centerline",
            self._rr.LineStrips3D([centers], colors=[80, 220, 255], radii=0.008),
        )
        self._recording.log(
            f"{root}/swept_clearance",
            self._rr.LineStrips3D(rings, colors=[255, 190, 40], radii=0.006),
        )


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
    sink.add_argument(
        "--serve-web",
        action="store_true",
        help="Serve a robot-hosted browser viewer",
    )
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-path", default="/rerun")
    parser.add_argument("--viewer-port", type=int, default=8081)
    parser.add_argument("--websocket-port", type=int, default=9877)
    parser.add_argument("--server-memory-limit", default="256MB")
    parser.add_argument(
        "--sensor-calibration",
        type=Path,
        default=Path("config/k1_calibration.json"),
        help="K1 calibration used for the body-to-camera transform",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--print-topics",
        action="store_true",
        help="Print the complete ROS subscription contract and exit",
    )
    args = parser.parse_args()
    for name in ("web_port", "viewer_port", "websocket_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if len({args.web_port, args.viewer_port, args.websocket_port}) != 3:
        parser.error("web, viewer, and WebSocket ports must be different")
    if not args.web_path.startswith("/") or "?" in args.web_path or "#" in args.web_path:
        parser.error("--web-path must be an absolute URL path without a query or fragment")
    if args.web_path.rstrip("/") in {"", "/"}:
        parser.error("--web-path must name a path such as /rerun")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.print_topics:
        for name, topic in vars(ObservabilityTopics()).items():
            print(f"{name}: {topic}")
        return
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        import rclpy
        from rclpy.executors import ExternalShutdownException
    except ImportError as exc:
        raise SystemExit("Install the visualization extra with: uv sync --extra viz") from exc

    rclpy.init(args=None)
    recording = rr.new_recording("nero_k1", make_default=False)
    gateway = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(args.save))
    elif args.serve_web:
        rr.serve_web(
            open_browser=False,
            web_port=args.viewer_port,
            ws_port=args.websocket_port,
            recording=recording,
            server_memory_limit=args.server_memory_limit,
        )
        gateway = start_web_gateway(
            args.web_port,
            viewer_port=args.viewer_port,
            websocket_port=args.websocket_port,
            path=args.web_path,
        )
        logger.info(
            "Robot-hosted Rerun available at http://<robot-ip>:%d%s",
            args.web_port,
            args.web_path,
        )
    elif args.spawn or not args.connect:
        recording.spawn()
    else:
        recording.connect(args.connect or "127.0.0.1:9876")
    calibration_path = args.sensor_calibration
    if not calibration_path.is_file():
        calibration_path = Path("config/k1_geek_nominal_calibration.json")
    calibration = K1Calibration.load(calibration_path)
    bridge = RerunRosBridge(recording, body_to_camera=np.asarray(calibration.tbc))
    recording.send_blueprint(
        create_default_blueprint(rrb),
        make_active=True,
        make_default=True,
    )
    logger.info(
        "Rerun bridge subscribed to: %s",
        ", ".join(bridge.subscription_topics),
    )
    try:
        rclpy.spin(bridge.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        bridge.node.destroy_node()
        if gateway is not None:
            gateway.shutdown()
            gateway.server_close()
        recording.flush()
        recording.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
