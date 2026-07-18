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
    clock: str = "/clock"
    odom: str = "/odom"
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
        expected_calibration: K1Calibration | None = None,
    ):
        try:
            import rclpy
            from geometry_msgs.msg import Pose2D
            from nav_msgs.msg import Odometry
            from rosgraph_msgs.msg import Clock
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
        self._expected_calibration = expected_calibration
        self._topics = topics or BoosterStudioTopics()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._rgb: Any = None
        self._depth: Any = None
        self._pending_rgb: dict[int, tuple[Any, float]] = {}
        self._pending_depth: dict[int, Any] = {}
        self._camera_info: Any = None
        self._imu: Any = None
        self._pose = np.zeros(3, dtype=float)
        self._pose_samples: list[tuple[float, np.ndarray]] = []
        self._sim_time: float | None = None
        self._odom: Any = None
        self._imu_samples: list[tuple[float, ...]] = []
        self._orientation_samples: list[tuple[float, np.ndarray]] = []
        self._rgb_timestamps: list[float] = []
        self._frame_tokens = 1.0
        self._last_source_frame_timestamp: float | None = None
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
            self._node.create_subscription(
                Image,
                self._topics.depth,
                self._on_depth,
                rclpy.qos.qos_profile_sensor_data,
            ),
            self._node.create_subscription(
                CameraInfo, self._topics.camera_info, self._on_camera_info, 10
            ),
            *(self._node.create_subscription(Imu, topic, self._on_imu, 50) for topic in imu_topics),
            self._node.create_subscription(Pose2D, self._topics.pose, self._on_pose, 10),
            self._node.create_subscription(Clock, self._topics.clock, self._on_clock, 50),
            self._node.create_subscription(Odometry, self._topics.odom, self._on_odom, 10),
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

    @staticmethod
    def _stamp_ns(message: Any) -> int:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _commit_synchronized_frame(self, stamp_ns: int) -> None:
        rgb_entry = self._pending_rgb.get(stamp_ns)
        depth = self._pending_depth.get(stamp_ns)
        if rgb_entry is None or depth is None:
            return
        rgb, _ = rgb_entry
        sensor_timestamp = self._stamp(rgb)
        del self._pending_rgb[stamp_ns]
        del self._pending_depth[stamp_ns]

        # Studio's host renderer commonly runs at 30 Hz and its container relay
        # does not forward camera-control commands. Deliver synchronized pairs
        # to policies at the real K1 rate using a timestamp-driven token bucket.
        expected = self._expected_calibration
        if expected is not None and self._last_source_frame_timestamp is not None:
            elapsed = max(0.0, sensor_timestamp - self._last_source_frame_timestamp)
            self._frame_tokens = min(
                2.0, self._frame_tokens + elapsed * expected.camera_fps
            )
        self._last_source_frame_timestamp = sensor_timestamp
        if expected is not None and self._frame_tokens < 1.0 - 1e-9:
            return
        if expected is not None:
            self._frame_tokens -= 1.0

        if expected is not None:
            minimum = expected.depth_min_m
            maximum = expected.depth_max_m
            if minimum is not None or maximum is not None:
                depth_values = depth.data
                scale = expected.depth_map_factor
                invalid = np.zeros(depth_values.shape, dtype=bool)
                if minimum is not None:
                    invalid |= depth_values < minimum * scale
                if maximum is not None:
                    invalid |= depth_values > maximum * scale
                depth_values[invalid] = 0
        rgb.nero_timestamp = sensor_timestamp
        depth.nero_timestamp = sensor_timestamp
        self._rgb = rgb
        self._depth = depth
        self._rgb_timestamps.append(sensor_timestamp)
        if len(self._rgb_timestamps) > 1000:
            del self._rgb_timestamps[:-500]
        self._ready.notify_all()

    def _bound_pending_frames(self) -> None:
        """Bound unmatched image queues even if synchronization never succeeds."""
        for pending in (self._pending_rgb, self._pending_depth):
            while len(pending) > 10:
                del pending[min(pending)]

    def _on_rgb(self, message: Any) -> None:
        image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        stamp_ns = self._stamp_ns(message)
        with self._ready:
            self._pending_rgb[stamp_ns] = (
                SimpleNamespace(data=image, header=message.header),
                time.monotonic(),
            )
            self._bound_pending_frames()
            self._commit_synchronized_frame(stamp_ns)

    def _on_depth(self, message: Any) -> None:
        encoding = message.encoding.lower()
        dtype = np.uint16 if encoding in {"16uc1", "mono16"} else np.float32
        depth = np.frombuffer(message.data, dtype=dtype).reshape(
            int(message.height), int(message.width)
        )
        if bool(message.is_bigendian) != (depth.dtype.byteorder == ">"):
            depth = depth.byteswap().newbyteorder()
        stamp_ns = self._stamp_ns(message)
        with self._ready:
            self._pending_depth[stamp_ns] = SimpleNamespace(
                data=depth.copy(), header=message.header
            )
            self._bound_pending_frames()
            self._commit_synchronized_frame(stamp_ns)

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
        with self._ready:
            # The IMU header uses wall time, while camera headers use simulation
            # time. The latest /clock sample places both in one sensor domain.
            timestamp = self._sim_time if self._sim_time is not None else time.monotonic()
            if self._imu_samples and timestamp <= self._imu_samples[-1][6]:
                return
            sample = (*acceleration.tolist(), *gyro.tolist(), timestamp)
            self._imu = SimpleNamespace(
                rpy=rpy,
                angular_velocity=gyro,
                linear_acceleration=acceleration,
            )
            self._imu_samples.append(sample)
            self._orientation_samples.append((timestamp, rpy))
            if len(self._imu_samples) > 2000:
                del self._imu_samples[:-1000]
            if len(self._orientation_samples) > 2000:
                del self._orientation_samples[:-1000]
            self._ready.notify_all()

    def _on_pose(self, message: Any) -> None:
        with self._lock:
            self._pose = np.array([message.x, message.y, message.theta], dtype=float)
            timestamp = self._sim_time if self._sim_time is not None else time.monotonic()
            self._pose_samples.append((timestamp, self._pose.copy()))
            if len(self._pose_samples) > 5000:
                del self._pose_samples[:-2500]

    def _on_clock(self, message: Any) -> None:
        with self._lock:
            if self._sim_time is None:
                # Discard any startup samples recorded with receipt time so the
                # interpolation history contains exactly one clock domain.
                self._pose_samples.clear()
            self._sim_time = float(message.clock.sec) + float(message.clock.nanosec) * 1e-9

    def _on_odom(self, message: Any) -> None:
        orientation = message.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("xyz")[2]
        position = message.pose.pose.position
        with self._ready:
            self._odom = SimpleNamespace(
                pose_2d=np.array([position.x, position.y, yaw], dtype=float),
                timestamp=time.monotonic(),
            )
            self._ready.notify_all()

    def _ground_truth_pose_locked(self, timestamp: float) -> np.ndarray:
        if not self._pose_samples:
            return self._pose.copy()
        times = np.fromiter((sample[0] for sample in self._pose_samples), dtype=float)
        index = int(np.searchsorted(times, timestamp))
        if index <= 0:
            return self._pose_samples[0][1].copy()
        if index >= len(self._pose_samples):
            return self._pose_samples[-1][1].copy()
        before_time, before = self._pose_samples[index - 1]
        after_time, after = self._pose_samples[index]
        weight = (timestamp - before_time) / (after_time - before_time)
        result = before + weight * (after - before)
        yaw_delta = np.arctan2(np.sin(after[2] - before[2]), np.cos(after[2] - before[2]))
        result[2] = before[2] + weight * yaw_delta
        return result

    def _orientation_locked(self, timestamp: float) -> np.ndarray:
        if not self._orientation_samples:
            return np.asarray(self._imu.rpy, dtype=float).copy()
        times = np.fromiter(
            (sample[0] for sample in self._orientation_samples), dtype=float
        )
        index = int(np.searchsorted(times, timestamp))
        if index <= 0:
            return self._orientation_samples[0][1].copy()
        if index >= len(self._orientation_samples):
            return self._orientation_samples[-1][1].copy()
        before_time, before = self._orientation_samples[index - 1]
        after_time, after = self._orientation_samples[index]
        weight = (timestamp - before_time) / (after_time - before_time)
        delta = np.arctan2(np.sin(after - before), np.cos(after - before))
        return before + weight * delta

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
                for value in (self._rgb, self._depth, self._camera_info, self._imu, self._odom)
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
                            ("odom", self._odom),
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
            if self._expected_calibration is not None:
                expected = (
                    self._expected_calibration.depth_height or self._expected_calibration.height,
                    self._expected_calibration.depth_width or self._expected_calibration.width,
                )
                if expected_shape != expected:
                    raise RuntimeError(
                        "Booster Studio camera does not match the K1 Geek profile: "
                        f"{expected_shape[::-1]} != {expected[::-1]}. Re-run "
                        "nero-setup-booster-room --activate with the same calibration "
                        "and reload the scene."
                    )
        self.set_velocity(0.0, 0.0, 0.0)
        self._initialized = True

    def validate_sensor_profile(self, camera_fps: float, imu_frequency: float) -> None:
        """Reject a sim whose delivered sensors differ from the real K1 profile."""
        expected = self._expected_calibration
        if expected is None:
            return
        camera_error = abs(camera_fps - expected.camera_fps) / expected.camera_fps
        imu_error = abs(imu_frequency - expected.imu_frequency) / expected.imu_frequency
        if camera_error > 0.12 or imu_error > 0.12:
            raise RuntimeError(
                "Booster Studio sensor-rate mismatch: "
                f"camera {camera_fps:.1f}/{expected.camera_fps:.1f} Hz, "
                f"IMU {imu_frequency:.1f}/{expected.imu_frequency:.1f} Hz"
            )

    def measure_sensor_rates(self, duration: float = 2.0) -> tuple[float, float]:
        """Measure rendered camera and IMU rates instead of assuming target rates."""
        start = time.monotonic()
        with self._ready:
            # Camera and IMU timestamps are deliberately expressed in the
            # simulator's /clock domain.  Use sensor-domain markers here;
            # comparing them with the host's monotonic clock would discard
            # every sample once simulation time starts near zero.
            rgb_marker = self._rgb_timestamps[-1] if self._rgb_timestamps else -np.inf
            imu_marker = self._imu_samples[-1][6] if self._imu_samples else -np.inf
            self._ready.wait_for(
                lambda: time.monotonic() >= start + duration,
                timeout=duration + 0.5,
            )
            rgb_times = [value for value in self._rgb_timestamps if value > rgb_marker]
            imu_times = [sample[6] for sample in self._imu_samples if sample[6] > imu_marker]
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
            synchronized_imu = SimpleNamespace(
                rpy=self._orientation_locked(frame_timestamp),
                angular_velocity=self._imu.angular_velocity,
                linear_acceleration=self._imu.linear_acceleration,
            )
            return RobotState(
                mode="walk",
                imu=synchronized_imu,
                odom=self._odom,
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

    def get_ground_truth_pose(self, timestamp: float | None = None) -> np.ndarray:
        """Return Studio's planar reference pose synchronized to a camera frame."""
        with self._lock:
            if timestamp is None:
                timestamp = self.image_source_timestamp(self._rgb)
            return self._ground_truth_pose_locked(float(timestamp))

    @staticmethod
    def image_to_array(image: Any) -> np.ndarray:
        return np.asarray(getattr(image, "data", image))

    @staticmethod
    def image_timestamp(image: Any) -> float:
        synchronized = getattr(image, "nero_timestamp", None)
        return float(
            synchronized
            if synchronized is not None
            else BoosterStudioRobotInterface._stamp(image)
        )

    @staticmethod
    def image_source_timestamp(image: Any) -> float:
        """Return the renderer's simulation timestamp for reference alignment."""
        return BoosterStudioRobotInterface._stamp(image)

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
            try:
                self.stop()
            except Exception as exc:
                # A K1 outside WALK mode rejects even a zero Move command. Do
                # not let that prevent ROS teardown or hide an earlier error.
                logger.warning("Could not send final zero velocity: %s", exc)
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
    reference_calibration: K1Calibration | None = None,
) -> K1Calibration:
    """Create IMU_RGBD settings from live intrinsics and the K1 simulator MJCF."""
    reference = reference_calibration
    body_to_camera = (
        np.asarray(reference.tbc, dtype=float).copy()
        if reference is not None
        else np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0669],
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.3559],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
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
        shutter_type=reference.shutter_type if reference else "global",
        rgb_fov_degrees=reference.rgb_fov_degrees if reference else None,
        depth_width=reference.depth_width if reference else int(camera_info.width),
        depth_height=reference.depth_height if reference else int(camera_info.height),
        depth_fps=reference.depth_fps if reference else camera_fps,
        depth_fov_degrees=reference.depth_fov_degrees if reference else None,
        depth_accuracy_at_1m=(reference.depth_accuracy_at_1m if reference else None),
        depth_min_m=reference.depth_min_m if reference else None,
        depth_max_m=reference.depth_max_m if reference else None,
    )
    calibration.save(path)
    return calibration
