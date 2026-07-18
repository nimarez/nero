"""Booster Studio K1 sensor and locomotion adapter.

This module is imported only inside Booster Studio's Linux virtual robot terminal.
It consumes the simulator's ROS 2 RGB-D, CameraInfo, IMU, and pose topics while
sending motion through the same official Booster locomotion client used on a K1.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from nero.robot import RobotState
from nero.perception.object_detector import ObjectDetection, ObjectDetector
from nero.slam.k1_calibration import K1Calibration, estimate_frequency

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoosterStudioTopics:
    """Single-robot topics published by Booster Studio's K1 simulator."""

    IMU_CANDIDATES: ClassVar[tuple[str, ...]] = (
        "/imu/data",
        "/booster/ros2_k2_imu/robot1",
    )

    rgb: str = "/rgbd_camera/rgb/image_compressed"
    depth: str = "/rgbd_camera/depth/image_raw"
    camera_info: str = "/rgbd_camera/rgb/camera_info"
    # Studio 1.9.10 exposes /imu/data with its renderer connected, while its
    # headless startup can briefly expose the legacy robot-specific topic.
    # None means subscribe to both so the K1's built-in IMU stays implicit.
    imu: str | None = None
    pose: str = "/soccer/sim/localization/robot_pose"
    detections: str = "/soccer/sim/vision/detections"


class BoosterStudioObjectDetector(ObjectDetector):
    """Expose Booster Studio's live simulated detections to the shared policy."""

    def __init__(self, robot: "BoosterStudioRobotInterface"):
        super().__init__()
        self._robot = robot

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def detect(self, rgb: np.ndarray, depth: np.ndarray, camera_info=None):
        return self._robot.get_detections()


class BoosterStudioRobotInterface:
    """RobotInterface-compatible adapter for a Booster Studio virtual K1."""

    def __init__(
        self,
        *,
        topics: BoosterStudioTopics | None = None,
        network_interface: str = "127.0.0.1",
        robot_name: str | None = None,
        timeout: float = 15.0,
    ):
        try:
            import rclpy
            from geometry_msgs.msg import Pose2D
            from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
            from vision_msgs.msg import Detection2DArray
            from booster_robotics_sdk_python import B1LocoClient, ChannelFactory
        except ImportError as exc:
            raise RuntimeError(
                "Booster Studio integration must run in its Linux virtual robot "
                "terminal, where ROS 2 and the Booster SDK are preinstalled"
            ) from exc

        self._rclpy = rclpy
        self._timeout = timeout
        self._topics = topics or BoosterStudioTopics()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._rgb: Any = None
        self._depth: Any = None
        self._camera_info: Any = None
        self._imu: Any = None
        self._pose = np.zeros(3, dtype=float)
        self._imu_samples: list[tuple[float, ...]] = []
        self._rgb_timestamps: list[float] = []
        self._detections: list[ObjectDetection] = []
        self._last_frame_timestamp: float | None = None
        self._last_frame_samples: list[tuple[float, ...]] = []
        self._initialized = False
        self._closed = False

        ChannelFactory.Instance().Init(0, network_interface)
        self._loco = B1LocoClient()
        if robot_name and hasattr(self._loco, "InitWithName"):
            self._loco.InitWithName(robot_name)
        else:
            self._loco.Init()

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("nero_booster_studio")
        imu_topics = (
            (self._topics.imu,)
            if self._topics.imu is not None
            else BoosterStudioTopics.IMU_CANDIDATES
        )
        self._subscriptions = [
            self._node.create_subscription(CompressedImage, self._topics.rgb, self._on_rgb, 10),
            self._node.create_subscription(Image, self._topics.depth, self._on_depth, 10),
            self._node.create_subscription(
                CameraInfo, self._topics.camera_info, self._on_camera_info, 10
            ),
            *(self._node.create_subscription(Imu, topic, self._on_imu, 50) for topic in imu_topics),
            self._node.create_subscription(Pose2D, self._topics.pose, self._on_pose, 10),
            self._node.create_subscription(
                Detection2DArray,
                self._topics.detections,
                self._on_detections,
                10,
            ),
        ]
        self._spin_thread = threading.Thread(
            target=self._spin, name="nero-booster-studio-ros", daemon=True
        )
        self._spin_thread.start()
        self._info = SimpleNamespace(
            manufacturer="Booster Robotics",
            model="K1 (Booster Studio)",
            serial_number=robot_name or "robot0",
        )

    def _spin(self) -> None:
        while not self._closed and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    @staticmethod
    def _stamp(message: Any) -> float:
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_rgb(self, message: Any) -> None:
        image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        timestamp = time.monotonic()
        with self._ready:
            self._rgb = SimpleNamespace(data=image, header=message.header, nero_timestamp=timestamp)
            self._rgb_timestamps.append(timestamp)
            if len(self._rgb_timestamps) > 1000:
                del self._rgb_timestamps[:-500]
            self._ready.notify_all()

    def _on_depth(self, message: Any) -> None:
        encoding = message.encoding.lower()
        dtype = np.uint16 if encoding in {"16uc1", "mono16"} else np.float32
        depth = np.frombuffer(message.data, dtype=dtype).reshape(
            int(message.height), int(message.width)
        )
        if bool(message.is_bigendian) != (depth.dtype.byteorder == ">"):
            depth = depth.byteswap().newbyteorder()
        with self._ready:
            self._depth = SimpleNamespace(data=depth.copy(), header=message.header)
            self._ready.notify_all()

    def _on_camera_info(self, message: Any) -> None:
        with self._ready:
            self._camera_info = SimpleNamespace(
                k=np.asarray(message.k, dtype=float).reshape(3, 3),
                d=list(message.d),
                width=int(message.width),
                height=int(message.height),
                header=message.header,
            )
            self._ready.notify_all()

    def _on_imu(self, message: Any) -> None:
        quaternion = [
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        ]
        rpy = Rotation.from_quat(quaternion).as_euler("xyz")
        gyro = np.array(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ]
        )
        acceleration = np.array(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ]
        )
        # Booster Studio currently stamps camera frames in simulation time and its
        # ROS IMU publisher in wall time. Receipt time gives ORB one shared,
        # monotonic clock domain for synchronizing the two streams.
        timestamp = time.monotonic()
        sample = (*acceleration.tolist(), *gyro.tolist(), timestamp)
        with self._ready:
            self._imu = SimpleNamespace(
                rpy=rpy,
                angular_velocity=gyro,
                linear_acceleration=acceleration,
            )
            self._imu_samples.append(sample)
            if len(self._imu_samples) > 2000:
                del self._imu_samples[:-1000]
            self._ready.notify_all()

    def _on_pose(self, message: Any) -> None:
        with self._lock:
            self._pose = np.array([message.x, message.y, message.theta], dtype=float)

    def _on_detections(self, message: Any) -> None:
        detections = []
        for item in message.detections:
            if not item.results:
                continue
            result = item.results[0]
            hypothesis = result.hypothesis
            position = result.pose.pose.position
            # Booster Studio publishes the detection position in the K1 trunk
            # frame: +x forward, +y left. Nero's controller consumes
            # [lateral, vertical, forward].
            position_3d = np.array([position.y, 0.0, position.x], dtype=float)
            center = getattr(item.bbox.center, "position", item.bbox.center)
            half_width = float(item.bbox.size_x) / 2.0
            half_height = float(item.bbox.size_y) / 2.0
            bbox = (
                int(round(float(center.x) - half_width)),
                int(round(float(center.y) - half_height)),
                int(round(float(center.x) + half_width)),
                int(round(float(center.y) + half_height)),
            )
            detections.append(
                ObjectDetection(
                    label=str(hypothesis.class_id),
                    confidence=float(hypothesis.score),
                    bbox=bbox,
                    position_3d=position_3d,
                    distance=float(np.hypot(position.x, position.y)),
                )
            )
        with self._lock:
            self._detections = detections

    def initialize(self) -> None:
        """Wait until every required simulated sensor has produced data."""
        deadline = time.monotonic() + self._timeout
        with self._ready:
            while not all(
                value is not None
                for value in (self._rgb, self._depth, self._camera_info, self._imu)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        name
                        for name, value in (
                            ("rgb", self._rgb),
                            ("depth", self._depth),
                            ("camera_info", self._camera_info),
                            ("imu", self._imu),
                        )
                        if value is None
                    ]
                    raise RuntimeError(
                        "Booster Studio sensor timeout; missing: " + ", ".join(missing)
                    )
                self._ready.wait(remaining)
            expected_shape = (self._camera_info.height, self._camera_info.width)
            if self._rgb.data.shape[:2] != expected_shape:
                raise RuntimeError(
                    "Booster Studio RGB dimensions do not match CameraInfo: "
                    f"{self._rgb.data.shape[:2]} != {expected_shape}"
                )
            if self._depth.data.shape != expected_shape:
                raise RuntimeError(
                    "Booster Studio depth dimensions do not match CameraInfo: "
                    f"{self._depth.data.shape} != {expected_shape}"
                )
            if self._depth.data.dtype != np.uint16:
                raise RuntimeError(
                    "Booster Studio K1 depth must be 16-bit millimetres, got "
                    f"{self._depth.data.dtype}"
                )
        self.set_velocity(0.0, 0.0, 0.0)
        self._initialized = True

    def measure_sensor_rates(self, duration: float = 2.0) -> tuple[float, float]:
        """Measure rendered camera and IMU rates instead of assuming target rates."""
        start = time.monotonic()
        with self._ready:
            self._ready.wait_for(
                lambda: time.monotonic() >= start + duration,
                timeout=duration + 0.5,
            )
            rgb_times = [value for value in self._rgb_timestamps if value >= start]
            imu_times = [sample[6] for sample in self._imu_samples if sample[6] >= start]
        return (
            estimate_frequency(rgb_times, "simulated camera"),
            estimate_frequency(imu_times, "simulated IMU"),
        )

    @property
    def robot_info(self) -> Any:
        return self._info

    def get_state(self, include_images: bool = True) -> RobotState:
        if not self._initialized:
            raise RuntimeError("initialize() must be called before get_state()")
        with self._lock:
            frame_timestamp = self.image_timestamp(self._rgb)
            if self._last_frame_timestamp != frame_timestamp:
                self._last_frame_samples = [
                    sample
                    for sample in self._imu_samples
                    if (
                        self._last_frame_timestamp is None or sample[6] > self._last_frame_timestamp
                    )
                    and sample[6] <= frame_timestamp
                ]
                self._last_frame_timestamp = frame_timestamp
            samples = list(self._last_frame_samples)
            return RobotState(
                mode="walk",
                imu=self._imu,
                odom=SimpleNamespace(pose_2d=self._pose.copy()),
                rgb=self._rgb if include_images else None,
                depth=self._depth if include_images else None,
                camera_info=self._camera_info,
                imu_samples=samples,
            )

    def get_camera_info(self) -> Any:
        with self._lock:
            return self._camera_info

    def get_detections(self) -> list[ObjectDetection]:
        with self._lock:
            return list(self._detections)

    @staticmethod
    def image_to_array(image: Any) -> np.ndarray:
        return np.asarray(getattr(image, "data", image))

    @staticmethod
    def image_timestamp(image: Any) -> float:
        return float(getattr(image, "nero_timestamp", BoosterStudioRobotInterface._stamp(image)))

    def speak(self, text: str) -> None:
        print(f"\n[SIMULATED K1 SPEAKER] {text}", flush=True)

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        error = self._loco.Move(float(vx), float(vy), float(vyaw))
        if error not in (None, 0):
            raise RuntimeError(
                f"Booster Studio rejected velocity command ({error}); set the K1 to WALK mode"
            )

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            self._closed = True
            self._spin_thread.join(timeout=2.0)
            self._node.destroy_node()


def write_booster_studio_calibration(
    camera_info: Any,
    path: Path,
    *,
    camera_fps: float,
    imu_frequency: float,
) -> K1Calibration:
    """Create IMU_RGBD settings from live intrinsics and the K1 simulator MJCF."""
    body_to_camera = np.eye(4)
    body_to_camera[:3, :3] = Rotation.from_euler("xyz", [0.0, -1.5708, -1.5708]).as_matrix()
    body_to_camera[:3, 3] = [0.0669, 0.0, 0.3559]
    calibration = K1Calibration(
        camera_frame=str(camera_info.header.frame_id),
        imu_frame="imu_link",
        width=int(camera_info.width),
        height=int(camera_info.height),
        camera_fps=camera_fps,
        camera_matrix=np.asarray(camera_info.k).reshape(-1).tolist(),
        distortion=list(camera_info.d),
        depth_map_factor=1000.0,
        camera_rgb=False,
        tbc=body_to_camera.tolist(),
        imu_frequency=imu_frequency,
        imu_noise_gyro=0.005,
        imu_noise_acc=0.01,
        imu_gyro_walk=0.0001,
        imu_acc_walk=0.001,
        source="Booster Studio K1 MJCF + live simulated CameraInfo",
    )
    calibration.save(path)
    return calibration
