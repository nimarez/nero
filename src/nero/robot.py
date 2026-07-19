"""Physical Booster K1 adapter using the public SDK and native ROS 2 topics."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

K1_HEAD_YAW_LIMITS = (-1.0, 1.0)
K1_HEAD_PITCH_LIMITS = (-0.349, 0.855)
K1_MAX_RGBD_RESOLUTION = (544, 448)


@dataclass(frozen=True)
class K1Topics:
    """Sensor topics provided by the production K1 Geek image."""

    rgb: str = "/boostercamera/head/rgb"
    depth: str = "/boostercamera/head/depth"
    camera_info: str = "/boostercamera/head/rgb/camera_info"
    raw_rgb: str = "/boostercamera/head/raw/rgb"
    odom: str = "/odometer_state"
    joints: str = "/joint_states"


@dataclass
class RobotState:
    """Aggregated robot state snapshot."""

    mode: Any = None
    imu: Any = None
    odom: Any = None
    joints: Any = None
    rgb: Any = None
    depth: Any = None
    camera_info: Any = None
    imu_samples: Optional[list[tuple[float, ...]]] = None
    battery_level: float | None = None

    @property
    def position_2d(self) -> np.ndarray:
        return np.asarray(self.odom.pose_2d, dtype=float) if self.odom is not None else np.zeros(3)

    @property
    def orientation_rpy(self) -> np.ndarray:
        return np.asarray(self.imu.rpy, dtype=float) if self.imu is not None else np.zeros(3)

    @property
    def angular_velocity(self) -> np.ndarray:
        return (
            np.asarray(self.imu.angular_velocity, dtype=float)
            if self.imu is not None
            else np.zeros(3)
        )

    @property
    def linear_acceleration(self) -> np.ndarray:
        return (
            np.asarray(self.imu.linear_acceleration, dtype=float)
            if self.imu is not None
            else np.zeros(3)
        )


class RobotAdapter(Protocol):
    def initialize(self) -> None: ...
    def get_state(self, include_images: bool = True) -> RobotState: ...
    def get_camera_info(self) -> Any: ...
    def image_to_array(self, image: Any) -> np.ndarray: ...
    def image_timestamp(self, image: Any) -> float: ...
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None: ...
    def set_head_pose(self, pitch: float, yaw: float, duration: float = 0.35) -> None: ...
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class RobotInterface:
    """Real K1 transport: ROS 2 sensors plus official B1 locomotion RPC."""

    def __init__(
        self,
        network_interface: str = "",
        virtual_robot_name: str = "",
        timeout: float = 10.0,
        sensor_startup_timeout: float = 90.0,
        rgbd_sync_tolerance: float = 0.02,
        topics: K1Topics | None = None,
    ):
        try:
            import rclpy
            from booster_interface.msg import Odometer
            from booster_robotics_sdk_python import (
                B1BatteryStateSubscriber,
                B1LocoClient,
                B1LowStateSubscriber,
                ChannelFactory,
                LuiClient,
            )
            from sensor_msgs.msg import CameraInfo, Image, JointState
        except ImportError as exc:
            raise RuntimeError(
                "K1 hardware control requires the robot's ROS 2 environment and "
                "booster-robotics-sdk-python"
            ) from exc

        self._rclpy = rclpy
        self._timeout = float(timeout)
        self._sensor_startup_timeout = float(sensor_startup_timeout)
        if self._timeout <= 0 or self._sensor_startup_timeout <= 0:
            raise ValueError("K1 timeouts must be positive")
        if rgbd_sync_tolerance < 0:
            raise ValueError("RGB-D synchronization tolerance must be non-negative")
        self._rgbd_sync_tolerance_ns = int(rgbd_sync_tolerance * 1_000_000_000)
        self._topics = topics or K1Topics()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._rgb: Any = None
        self._depth: Any = None
        self._camera_info: Any = None
        self._imu: Any = None
        self._odom: Any = None
        self._joints: Any = None
        self._pending_rgb: dict[int, Any] = {}
        self._pending_depth: dict[int, Any] = {}
        self._rgb_message_count = 0
        self._depth_message_count = 0
        self._camera_info_message_count = 0
        self._closest_rgbd_offset_ns: int | None = None
        self._imu_samples: list[tuple[float, ...]] = []
        self._battery_level: float | None = None
        self._last_frame_timestamp: float | None = None
        self._mode = -1
        self._initialized = False
        self._closed = False

        self._network_interface = network_interface or os.getenv("BOOSTER_NET_IF", "lo")
        ChannelFactory.Instance().Init(0, self._network_interface)
        self._loco = B1LocoClient()
        if virtual_robot_name and hasattr(self._loco, "InitWithName"):
            self._loco.InitWithName(virtual_robot_name)
        else:
            self._loco.Init()
        if not self._loco.WaitForService(int(self._timeout * 1000)):
            raise RuntimeError(f"K1 locomotion service unavailable on {self._network_interface!r}")

        info = self._json_body(self._loco.GetRobotInfo())
        self._info = SimpleNamespace(
            manufacturer="Booster Robotics",
            model=str(info.get("model", info.get("robot_model", "K1"))),
            serial_number=str(info.get("serial_number", info.get("serial", "unknown"))),
            raw=info,
        )
        self._lui = LuiClient()
        self._lui_tts_failed = False
        if virtual_robot_name and hasattr(self._lui, "InitWithName"):
            self._lui.InitWithName(virtual_robot_name)
        else:
            self._lui.Init()

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("nero_k1_hardware")
        qos = rclpy.qos.QoSProfile(
            depth=100,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )
        self._subscriptions = [
            self._node.create_subscription(Image, self._topics.rgb, self._on_rgb, qos),
            self._node.create_subscription(Image, self._topics.depth, self._on_depth, qos),
            self._node.create_subscription(
                CameraInfo, self._topics.camera_info, self._on_camera_info, qos
            ),
            self._node.create_subscription(Odometer, self._topics.odom, self._on_odom, qos),
            self._node.create_subscription(JointState, self._topics.joints, self._on_joints, qos),
        ]
        self._low_state_subscriber = B1LowStateSubscriber(self._on_low_state)
        self._low_state_subscriber.InitChannel()
        self._battery_state_subscriber = B1BatteryStateSubscriber(self._on_battery_state)
        self._battery_state_subscriber.InitChannel()
        self._spin_thread = threading.Thread(target=self._spin, name="nero-k1-ros", daemon=True)
        self._spin_thread.start()
        logger.info("Connected to %s (%s)", self._info.model, self._info.serial_number)

    @staticmethod
    def _json_body(value: Any) -> dict[str, Any]:
        if hasattr(value, "to_json_str"):
            value = value.to_json_str()
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict) and "body" in value:
            value = value["body"]
            if isinstance(value, str):
                value = json.loads(value)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _stamp_ns(message: Any) -> int:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _stamp(message: Any) -> float:
        return RobotInterface._stamp_ns(message) * 1e-9

    def _spin(self) -> None:
        while not self._closed and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    def _commit_frame(self, stamp_ns: int) -> None:
        self._prune_pending(stamp_ns)
        if stamp_ns in self._pending_rgb:
            rgb_stamp = stamp_ns
            candidates = self._pending_depth
        elif stamp_ns in self._pending_depth:
            candidates = self._pending_rgb
            rgb_stamp = min(candidates, key=lambda value: abs(value - stamp_ns), default=None)
            if rgb_stamp is None:
                return
            depth_stamp = stamp_ns
            offset = abs(rgb_stamp - depth_stamp)
            self._closest_rgbd_offset_ns = (
                offset
                if self._closest_rgbd_offset_ns is None
                else min(self._closest_rgbd_offset_ns, offset)
            )
            if offset > getattr(self, "_rgbd_sync_tolerance_ns", 0):
                return
            rgb, depth = self._pending_rgb[rgb_stamp], self._pending_depth[depth_stamp]
            self._finish_frame_pair(rgb_stamp, depth_stamp, rgb, depth)
            return
        depth_stamp = min(candidates, key=lambda value: abs(value - rgb_stamp), default=None)
        if depth_stamp is None:
            return
        offset = abs(rgb_stamp - depth_stamp)
        self._closest_rgbd_offset_ns = (
            offset
            if self._closest_rgbd_offset_ns is None
            else min(self._closest_rgbd_offset_ns, offset)
        )
        if offset > getattr(self, "_rgbd_sync_tolerance_ns", 0):
            return
        self._finish_frame_pair(
            rgb_stamp,
            depth_stamp,
            self._pending_rgb[rgb_stamp],
            self._pending_depth[depth_stamp],
        )

    def _finish_frame_pair(self, rgb_stamp: int, depth_stamp: int, rgb: Any, depth: Any) -> None:
        self._rgb, self._depth = rgb, depth
        self._pending_rgb.pop(rgb_stamp, None)
        self._pending_depth.pop(depth_stamp, None)
        self._prune_pending(max(rgb_stamp, depth_stamp))
        self._ready.notify_all()

    def _prune_pending(self, newest_stamp: int) -> None:
        """Bound unmatched full-resolution image storage to roughly one second."""
        cutoff = newest_stamp - 1_000_000_000
        self._pending_rgb = {k: v for k, v in self._pending_rgb.items() if k >= cutoff}
        self._pending_depth = {k: v for k, v in self._pending_depth.items() if k >= cutoff}

    def _on_rgb(self, message: Any) -> None:
        with self._ready:
            self._rgb_message_count += 1
            stamp = self._stamp_ns(message)
            self._pending_rgb[stamp] = message
            self._commit_frame(stamp)

    def _on_depth(self, message: Any) -> None:
        with self._ready:
            self._depth_message_count += 1
            stamp = self._stamp_ns(message)
            self._pending_depth[stamp] = message
            self._commit_frame(stamp)

    def _on_camera_info(self, message: Any) -> None:
        with self._ready:
            self._camera_info_message_count += 1
            self._camera_info = SimpleNamespace(
                header=message.header,
                width=int(message.width),
                height=int(message.height),
                k=np.asarray(message.k, dtype=float).reshape(3, 3),
                d=np.asarray(message.d, dtype=float),
                distortion_model=message.distortion_model,
            )
            self._ready.notify_all()

    def _on_low_state(self, message: Any) -> None:
        """Receive the real 500 Hz body IMU carried in the K1 low-state stream."""
        value = message.imu_state
        timestamp = time.time()
        accel = np.asarray(value.acc, dtype=float)
        gyro = np.asarray(value.gyro, dtype=float)
        sample = (*accel.tolist(), *gyro.tolist(), timestamp)
        with self._ready:
            self._imu = SimpleNamespace(
                rpy=np.asarray(value.rpy, dtype=float),
                angular_velocity=gyro,
                linear_acceleration=accel,
            )
            if not self._imu_samples or timestamp > self._imu_samples[-1][-1]:
                self._imu_samples.append(sample)
                if len(self._imu_samples) > 4000:
                    del self._imu_samples[:-4000]
            self._ready.notify_all()

    def _on_battery_state(self, message: Any) -> None:
        """Receive the K1 state-of-charge percentage from ``rt/battery_state``."""
        level = float(message.soc)
        if not np.isfinite(level) or not 0.0 <= level <= 100.0:
            logger.warning("Ignoring invalid K1 battery state of charge: %r", level)
            return
        with self._ready:
            self._battery_level = level
            self._ready.notify_all()

    def _on_odom(self, message: Any) -> None:
        with self._ready:
            self._odom = SimpleNamespace(
                pose_2d=np.array([message.x, message.y, message.theta], dtype=float)
            )
            self._ready.notify_all()

    def _on_joints(self, message: Any) -> None:
        with self._ready:
            self._joints = message

    def _sensor_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self._rgb,
                self._depth,
                self._camera_info,
                self._imu,
                self._odom,
                self._battery_level,
            )
        )

    def initialize(self) -> None:
        """Verify all sensors and walking mode, then arm velocity output at zero."""
        if self._initialized:
            return
        deadline = time.monotonic() + self._sensor_startup_timeout
        with self._ready:
            while not self._sensor_ready():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "K1 sensor preflight timed out after "
                        f"{self._sensor_startup_timeout:g}s: {self.sensor_diagnostics()}"
                    )
                self._ready.wait(timeout=min(0.25, remaining))
            self._validate_max_camera_resolution()
        self.initialize_locomotion_only()
        logger.info(
            "K1 preflight passed at maximum RGB-D resolution %dx%d; velocity output armed at zero",
            *K1_MAX_RGBD_RESOLUTION,
        )

    def initialize_locomotion_only(self) -> None:
        """Arm locomotion without camera preflight for explicit external-pose agents."""
        if self._initialized:
            return
        self._mode = self.get_mode()
        if self._mode != 2:
            raise RuntimeError(
                f"K1 must already be in walking mode (2); current mode is {self._mode}. "
                "Nero will not change physical robot mode automatically."
            )
        self._loco.Move(0.0, 0.0, 0.0)
        self._initialized = True
        logger.info("K1 walking mode verified; locomotion output armed at zero")

    def _validate_max_camera_resolution(self) -> None:
        """Fail closed unless every registered camera stream uses the K1 maximum."""
        expected = K1_MAX_RGBD_RESOLUTION
        streams = {
            "rgb": self._rgb,
            "depth": self._depth,
            "camera_info": self._camera_info,
        }
        incompatible = []
        for name, message in streams.items():
            try:
                size = (int(message.width), int(message.height))
            except (AttributeError, TypeError, ValueError):
                incompatible.append(f"{name}=unknown")
                continue
            if size != expected:
                incompatible.append(f"{name}={size[0]}x{size[1]}")
        if incompatible:
            actual = ", ".join(incompatible)
            raise RuntimeError(
                "K1 camera is not publishing its maximum registered RGB-D resolution "
                f"{expected[0]}x{expected[1]} ({actual}). Restart Booster perception "
                "and rerun nero-k1-preflight."
            )

    def sensor_diagnostics(self) -> str:
        """Describe which live inputs arrived and whether RGB-D paired."""
        rgb_count = int(getattr(self, "_rgb_message_count", 0))
        depth_count = int(getattr(self, "_depth_message_count", 0))
        info_count = int(getattr(self, "_camera_info_message_count", 0))
        values = [
            f"aligned RGB={'ready' if self._rgb is not None else f'no pair ({rgb_count} messages)'}",
            f"depth={'ready' if self._depth is not None else f'no pair ({depth_count} messages)'}",
            f"camera_info={'ready' if self._camera_info is not None else f'no frames ({info_count} messages)'}",
            f"IMU={'ready' if self._imu is not None else 'missing'}",
            f"odometry={'ready' if self._odom is not None else 'missing'}",
            f"battery={'ready' if self._battery_level is not None else 'missing'}",
        ]
        offset = getattr(self, "_closest_rgbd_offset_ns", None)
        if rgb_count and depth_count and self._rgb is None and offset is not None:
            values.append(f"closest RGB-D offset={offset / 1_000_000.0:.1f}ms")
        return ", ".join(values)

    @property
    def robot_info(self) -> Any:
        return self._info

    def get_mode(self) -> int:
        body = self._json_body(self._loco.GetMode())
        return int(body.get("mode", -1))

    def get_state(self, include_images: bool = True) -> RobotState:
        deadline = time.monotonic() + self._timeout
        with self._ready:
            if not self._sensor_ready():
                raise RuntimeError("K1 sensor snapshot is not ready")
            while (
                self._last_frame_timestamp is not None
                and self._stamp(self._rgb) <= self._last_frame_timestamp
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("K1 synchronized RGB-D frame timed out")
                self._ready.wait(timeout=min(0.25, remaining))
            timestamp = self._stamp(self._rgb)
            start = self._last_frame_timestamp
            samples = [
                sample
                for sample in self._imu_samples
                if (start is None or sample[-1] > start) and sample[-1] <= timestamp
            ]
            self._imu_samples = [s for s in self._imu_samples if s[-1] > timestamp]
            self._last_frame_timestamp = timestamp
            return RobotState(
                mode=self._mode,
                imu=self._imu,
                odom=self._odom,
                joints=self._joints,
                rgb=self._rgb if include_images else None,
                depth=self._depth if include_images else None,
                camera_info=self._camera_info,
                imu_samples=samples,
                battery_level=self._battery_level,
            )

    def peek_state(self, include_images: bool = True) -> RobotState:
        """Return the latest snapshot without consuming synchronized IMU samples."""
        with self._lock:
            if not self._sensor_ready():
                raise RuntimeError("K1 sensor snapshot is not ready")
            return RobotState(
                mode=self._mode,
                imu=self._imu,
                odom=self._odom,
                joints=self._joints,
                rgb=self._rgb if include_images else None,
                depth=self._depth if include_images else None,
                camera_info=self._camera_info,
                imu_samples=[],
                battery_level=self._battery_level,
            )

    def get_camera_info(self) -> Any:
        with self._lock:
            if self._camera_info is None:
                raise RuntimeError("K1 CameraInfo is not ready")
            return self._camera_info

    @staticmethod
    def image_to_array(image: Any) -> np.ndarray:
        if image is None:
            raise ValueError("K1 returned no image data")
        if isinstance(image, np.ndarray):
            return image
        encoding = str(getattr(image, "encoding", "")).lower()
        if not encoding or not hasattr(image, "height"):
            return np.asarray(getattr(image, "data", image))
        height, width = int(image.height), int(image.width)
        data = np.frombuffer(image.data, dtype=np.uint8)
        if encoding == "nv12":
            yuv = data.reshape(height * 3 // 2, width)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if encoding in {"mono16", "16uc1"}:
            return np.frombuffer(image.data, dtype=np.uint16).reshape(height, width)
        if encoding in {"rgb8", "bgr8"}:
            result = data.reshape(height, width, 3)
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else result
        return np.asarray(getattr(image, "data", image))

    @staticmethod
    def image_timestamp(image: Any) -> float:
        header = getattr(image, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return time.time()
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def speak(self, text: str) -> None:
        # Retain compatibility with injected adapters used by downstream tests.
        legacy = getattr(self, "_robot", None)
        speaker = getattr(legacy, "speaker", None)
        synthesize = getattr(speaker, "synthesize", None)
        if callable(synthesize):
            synthesize(text)
            return
        if not getattr(self, "_lui_tts_failed", False):
            try:
                import booster_robotics_sdk_python as booster

                config = booster.LuiTtsConfig()
                parameter = booster.LuiTtsParameter()
                parameter.text = text
                self._lui.StartTts(config)
                try:
                    self._lui.SendTtsText(parameter)
                finally:
                    self._lui.StopTts()
                return
            except Exception:
                self._lui_tts_failed = True
                logger.warning(
                    "K1 LUI TTS is unavailable; falling back to flite",
                    exc_info=True,
                )

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
                subprocess.run(
                    ["flite", "-t", text, "-o", audio_file.name],
                    check=True,
                    timeout=30,
                )
                subprocess.run(
                    ["aplay", "-D", "plughw:0,0", audio_file.name],
                    check=True,
                    timeout=30,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Speech playback requires K1 LUI TTS or flite and aplay") from exc

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if not self._initialized:
            raise RuntimeError("initialize() must pass before velocity commands")
        values = np.asarray([vx, vy, vyaw], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("velocity command must be finite")
        error = self._loco.Move(float(vx), float(vy), float(vyaw))
        if error not in (None, 0):
            raise RuntimeError(f"K1 rejected velocity command ({error})")

    def set_head_pose(self, pitch: float, yaw: float, duration: float = 0.35) -> None:
        """Move the K1 head to an absolute pitch/yaw pose within ``duration``."""
        values = np.asarray([pitch, yaw, duration], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("head pose command must be finite")
        if not K1_HEAD_PITCH_LIMITS[0] <= pitch <= K1_HEAD_PITCH_LIMITS[1]:
            raise ValueError(f"head pitch must be within {K1_HEAD_PITCH_LIMITS} radians")
        if not K1_HEAD_YAW_LIMITS[0] <= yaw <= K1_HEAD_YAW_LIMITS[1]:
            raise ValueError(f"head yaw must be within {K1_HEAD_YAW_LIMITS} radians")
        if duration <= 0:
            raise ValueError("head motion duration must be positive")
        if not self._initialized:
            raise RuntimeError("initialize() must pass before head commands")
        rotate = getattr(self._loco, "RotateHeadWithTime", None)
        if rotate is None:
            raise RuntimeError("Booster SDK does not provide RotateHeadWithTime")
        error = rotate(float(pitch), float(yaw), max(1, int(round(duration * 1000.0))))
        if error not in (None, 0):
            raise RuntimeError(f"K1 rejected head pose command ({error})")

    def stop(self) -> None:
        if self._initialized:
            self.set_velocity(0.0, 0.0, 0.0)

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self.stop()
            except RuntimeError as exc:
                logger.warning("Could not send final zero velocity: %s", exc)
        finally:
            self._initialized = False
            self._closed = True
            if hasattr(self, "_low_state_subscriber"):
                self._low_state_subscriber.CloseChannel()
            if hasattr(self, "_battery_state_subscriber"):
                self._battery_state_subscriber.CloseChannel()
            if hasattr(self, "_spin_thread"):
                self._spin_thread.join(timeout=1.0)
            if hasattr(self, "_node"):
                self._node.destroy_node()
