"""Robot interface wrapper for Booster K1."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from boosteros.robots.booster import BoosterRobot, RobotModeName
    from boosteros.types import (
        AnyImage,
        CameraInfo,
        IMUState,
        JointStates,
        OdomState,
        RobotInfo,
    )
except ImportError as exc:
    BoosterRobot = None
    RobotModeName = str
    AnyImage = CameraInfo = IMUState = JointStates = OdomState = RobotInfo = Any
    _BOOSTEROS_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _BOOSTEROS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Aggregated robot state snapshot."""

    mode: Optional[RobotModeName] = None
    imu: Optional[IMUState] = None
    odom: Optional[OdomState] = None
    joints: Optional[JointStates] = None
    rgb: Optional[AnyImage] = None
    depth: Optional[AnyImage] = None
    camera_info: Optional[CameraInfo] = None

    @property
    def position_2d(self) -> np.ndarray:
        """Get 2D position [x, y, yaw] from odometry."""
        if self.odom is None:
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.odom.pose_2d)

    @property
    def orientation_rpy(self) -> np.ndarray:
        """Get orientation as RPY from IMU."""
        if self.imu is None:
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.imu.rpy)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Get angular velocity from IMU."""
        if self.imu is None:
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.imu.angular_velocity)

    @property
    def linear_acceleration(self) -> np.ndarray:
        """Get linear acceleration from IMU."""
        if self.imu is None:
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.imu.linear_acceleration)


class RobotInterface:
    """High-level robot interface wrapping BoosterRobot SDK."""

    def __init__(
        self,
        network_interface: str = "",
        virtual_robot_name: str = "",
        timeout: float = 10.0,
    ):
        if BoosterRobot is None:
            raise RuntimeError(
                "BoosterOS is unavailable. Install the vendor SDK on the supported "
                "Linux/ROS robot environment to enable hardware control."
            ) from _BOOSTEROS_IMPORT_ERROR

        self._robot = BoosterRobot(
            network_interface=network_interface,
            virtual_robot_name=virtual_robot_name,
            timeout=timeout,
            enable_tf_listener=True,
        )
        self._info = self._robot.robot_info
        self._initialized = False
        logger.info(
            f"Connected to {self._info.manufacturer} {self._info.model} ({self._info.serial_number})"
        )

    def initialize(self) -> None:
        """Initialize robot for navigation.

        Sets robot mode to 'walk' and ensures velocity is zeroed.
        Must be called before sending velocity commands.
        """
        if self._initialized:
            return

        logger.info("Initializing robot for navigation...")
        self._robot.set_mode("walk")
        self._robot.set_velocity(0.0, 0.0, 0.0)
        self._initialized = True
        logger.info("Robot initialized in walk mode")

    @property
    def robot_info(self) -> RobotInfo:
        return self._info

    def get_mode(self) -> RobotModeName:
        return self._robot.get_mode()

    def set_mode(self, mode: str) -> None:
        """Set robot mode: 'prepare', 'walk', 'damping', 'custom'."""
        self._robot.set_mode(mode)
        logger.info(f"Mode set to {mode}")

    def list_gaits(self) -> list[str]:
        return self._robot.list_gaits()

    def set_gait(self, gait: str) -> None:
        self._robot.set_gait(gait)
        logger.info(f"Gait set to {gait}")

    def get_state(self, include_images: bool = True) -> RobotState:
        """Get full robot state snapshot."""
        state = RobotState(
            mode=self.get_mode(),
            imu=self._robot.get_imu(),
            odom=self._robot.get_odom(),
            joints=self._robot.get_joint_states(),
            camera_info=self._robot.get_camera_info(),
        )
        if include_images:
            state.rgb = self._robot.get_image(img_type="rgb")
            state.depth = self._robot.get_image(img_type="depth")
        return state

    def get_rgb(self) -> AnyImage:
        return self._robot.get_image(img_type="rgb")

    def get_depth(self) -> AnyImage:
        return self._robot.get_image(img_type="depth")

    def get_rgb_frame(self) -> np.ndarray:
        """Get the K1 RGB image as a NumPy array."""
        return self.image_to_array(self.get_rgb())

    def get_depth_frame(self) -> np.ndarray:
        """Get the K1 depth image as a NumPy array."""
        return self.image_to_array(self.get_depth())

    @staticmethod
    def image_to_array(image: AnyImage) -> np.ndarray:
        """Normalize Booster image wrappers and raw arrays."""
        if image is None:
            raise ValueError("K1 returned no image data")
        data = getattr(image, "data", image)
        return np.asarray(data)

    @staticmethod
    def image_timestamp(image: AnyImage) -> float:
        """Return the camera's ROS timestamp, or the local receipt time."""
        header = getattr(image, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return time.time()
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def get_imu(self) -> IMUState:
        return self._robot.get_imu()

    def get_odom(self) -> OdomState:
        return self._robot.get_odom()

    def get_camera_info(self) -> CameraInfo:
        return self._robot.get_camera_info()

    def speak(self, text: str) -> None:
        """Speak text through the K1 audio service."""
        services = [
            getattr(self._robot, "speaker", None),
            getattr(self._robot, "audio", None),
            self._robot,
        ]
        for service in services:
            if service is None:
                continue
            for method_name in ("synthesize", "speak", "say", "text_to_speech"):
                method = getattr(service, method_name, None)
                if callable(method):
                    method(text)
                    return
        raise RuntimeError(
            "The connected Booster runtime does not expose a speaker API"
        )

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set robot velocity.

        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            vyaw: Yaw angular velocity (rad/s)
        """
        self._robot.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        """Emergency stop."""
        self._robot.set_velocity(0.0, 0.0, 0.0)
        logger.info("Emergency stop triggered")

    def reset_odom(self) -> None:
        self._robot.reset_odom()
        logger.info("Odometry reset")

    def get_transform(
        self, target_frame: str, source_frame: str = ""
    ) -> Optional[np.ndarray]:
        """Get transform between frames as 4x4 matrix."""
        try:
            return self._robot.get_transform(target_frame, source_frame)
        except Exception as e:
            logger.warning(f"Transform {source_frame} -> {target_frame} failed: {e}")
            return None

    def list_frames(self) -> list[str]:
        return self._robot.list_frames()

    def list_actions(self) -> list:
        return self._robot.list_actions()

    def do_action(self, action_id: str) -> None:
        self._robot.do_action(action_id)
        logger.info(f"Action {action_id} triggered")

    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        logger.info("Robot interface closed")
