"""Publish the POS Vive pose endpoint as standard ROS 2 messages."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from nero.observability.topics import ObservabilityTopics
from nero.vive.udp_transport import DEFAULT_STALE_AFTER_S, PosePacket

logger = logging.getLogger(__name__)

DEFAULT_LATEST_FILE = "/run/nero/vive_pose.json"
DEFAULT_RATE_HZ = 100.0
POSE_TOPIC = "/nero/localization/vive/controller_pose"
ODOMETRY_TOPIC = "/nero/localization/vive/controller_odometry"
TRACKING_TOPIC = "/nero/localization/vive/tracking"
DIAGNOSTICS_TOPIC = "/nero/localization/vive/diagnostics"
REFERENCE_POSE_TOPIC = ObservabilityTopics().reference_pose
REFERENCE_PATH_TOPIC = ObservabilityTopics().reference_path
MAX_REFERENCE_PATH_LENGTH = 10_000


@dataclass(frozen=True, slots=True)
class VivePoseState:
    """A validated snapshot read from the POS atomic pose endpoint."""

    packet: PosePacket
    received_at: float
    age_s: float


@dataclass(frozen=True, slots=True)
class ViveCalibration:
    """Fixed transforms required to turn a tracker pose into a K1 body pose."""

    map_from_lighthouse: np.ndarray
    base_from_controller: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ViveCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Vive calibration must be a JSON object")
        return cls(
            map_from_lighthouse=_transform_from_json(
                payload.get("map_from_lighthouse"), "map_from_lighthouse"
            ),
            base_from_controller=_transform_from_json(
                payload.get("base_from_controller"), "base_from_controller"
            ),
        )

    def map_from_base(self, packet: PosePacket) -> np.ndarray:
        lighthouse_from_controller = np.eye(4)
        lighthouse_from_controller[:3, :3] = Rotation.from_quat(packet.quaternion_xyzw).as_matrix()
        lighthouse_from_controller[:3, 3] = packet.position
        return (
            self.map_from_lighthouse
            @ lighthouse_from_controller
            @ np.linalg.inv(self.base_from_controller)
        )


def _transform_from_json(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    translation = np.asarray(value.get("translation"), dtype=float)
    quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=float)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError(f"{field}.translation must contain three finite numbers")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{field}.quaternion_xyzw must contain four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError(f"{field}.quaternion_xyzw must be nonzero")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = translation
    return transform


def read_latest_pose(
    path: str | Path,
    *,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    wall_clock: Callable[[], float] = time.time,
) -> VivePoseState:
    """Read and fail-close a POS pose snapshot without requiring ROS."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    packet = PosePacket.decode(json.dumps(payload, separators=(",", ":")).encode())
    received_at = float(payload["transport"]["received_at"])
    if not math.isfinite(received_at) or received_at <= 0:
        raise ValueError("transport.received_at must be a positive finite time")
    age_s = max(0.0, wall_clock() - received_at)
    quaternion_norm = math.sqrt(sum(value * value for value in packet.quaternion_xyzw))
    valid = packet.tracking_valid and age_s <= stale_after_s and quaternion_norm > 1e-9
    quaternion = (
        tuple(value / quaternion_norm for value in packet.quaternion_xyzw)
        if quaternion_norm > 1e-9
        else packet.quaternion_xyzw
    )
    return VivePoseState(
        packet=replace(packet, quaternion_xyzw=quaternion, tracking_valid=valid),
        received_at=received_at,
        age_s=age_s,
    )


class ViveRosBridge:
    """ROS 2 adapter for the transport-neutral POS latest-pose file."""

    def __init__(
        self,
        *,
        latest_file: str = DEFAULT_LATEST_FILE,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        rate_hz: float = DEFAULT_RATE_HZ,
        frame_id: str = "lighthouse_world",
        child_frame_id: str = "vive_controller",
        calibration: ViveCalibration | None = None,
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry, Path as RosPath
        from std_msgs.msg import Bool, String

        self._rclpy = rclpy
        self._node = rclpy.create_node("nero_vive_pose_bridge")
        self._types = {
            "PoseStamped": PoseStamped,
            "Odometry": Odometry,
            "Path": RosPath,
            "Bool": Bool,
            "String": String,
        }
        self._latest_file = latest_file
        self._stale_after_s = stale_after_s
        self._frame_id = frame_id
        self._child_frame_id = child_frame_id
        self._calibration = calibration
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        self._pose_publisher = self._node.create_publisher(PoseStamped, POSE_TOPIC, sensor_qos)
        self._odometry_publisher = self._node.create_publisher(Odometry, ODOMETRY_TOPIC, sensor_qos)
        self._tracking_publisher = self._node.create_publisher(Bool, TRACKING_TOPIC, 10)
        self._diagnostics_publisher = self._node.create_publisher(String, DIAGNOSTICS_TOPIC, 10)
        self._reference_pose_publisher = self._node.create_publisher(
            PoseStamped, REFERENCE_POSE_TOPIC, 10
        )
        self._reference_path_publisher = self._node.create_publisher(
            RosPath, REFERENCE_PATH_TOPIC, 1
        )
        self._reference_path = RosPath()
        self._reference_path.header.frame_id = "map"
        self._last_sequence: int | None = None
        self._last_valid: bool | None = None
        self._timer = self._node.create_timer(1.0 / rate_hz, self._tick)

    @property
    def node(self) -> Any:
        return self._node

    def _publish_status(self, state: VivePoseState | None, error: str | None = None) -> None:
        valid = state is not None and state.packet.tracking_valid
        tracking = self._types["Bool"]()
        tracking.data = valid
        self._tracking_publisher.publish(tracking)

        if valid != self._last_valid or error is not None:
            diagnostics = self._types["String"]()
            diagnostics.data = json.dumps(
                {
                    "valid": valid,
                    "controller_id": state.packet.controller_id if state else None,
                    "sequence": state.packet.sequence if state else None,
                    "age_ms": round(state.age_s * 1000.0, 3) if state else None,
                    "source_timestamp": state.packet.timestamp if state else None,
                    "error": error,
                },
                separators=(",", ":"),
            )
            self._diagnostics_publisher.publish(diagnostics)
            self._last_valid = valid

    def _tick(self) -> None:
        try:
            state = read_latest_pose(
                self._latest_file,
                stale_after_s=self._stale_after_s,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._publish_status(None, str(error))
            return

        self._publish_status(state)
        packet = state.packet
        if not packet.tracking_valid or packet.sequence == self._last_sequence:
            return
        self._last_sequence = packet.sequence
        stamp = self._node.get_clock().now().to_msg()

        pose = self._types["PoseStamped"]()
        pose.header.stamp = stamp
        pose.header.frame_id = self._frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = packet.position
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = packet.quaternion_xyzw
        self._pose_publisher.publish(pose)

        odometry = self._types["Odometry"]()
        odometry.header = pose.header
        odometry.child_frame_id = self._child_frame_id
        odometry.pose.pose = pose.pose
        (
            odometry.twist.twist.linear.x,
            odometry.twist.twist.linear.y,
            odometry.twist.twist.linear.z,
        ) = packet.linear_velocity
        (
            odometry.twist.twist.angular.x,
            odometry.twist.twist.angular.y,
            odometry.twist.twist.angular.z,
        ) = packet.angular_velocity
        self._odometry_publisher.publish(odometry)

        if self._calibration is not None:
            self._publish_reference_pose(packet, stamp)

    def _publish_reference_pose(self, packet: PosePacket, stamp: Any) -> None:
        """Publish calibrated ground truth without feeding it into navigation control."""
        transform = self._calibration.map_from_base(packet)
        message = self._types["PoseStamped"]()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = transform[:3, 3]
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = quaternion
        self._reference_pose_publisher.publish(message)
        self._reference_path.header = message.header
        self._reference_path.poses.append(message)
        if len(self._reference_path.poses) > MAX_REFERENCE_PATH_LENGTH:
            del self._reference_path.poses[: MAX_REFERENCE_PATH_LENGTH // 2]
        self._reference_path_publisher.publish(self._reference_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-file", default=DEFAULT_LATEST_FILE)
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--frame-id", default="lighthouse_world")
    parser.add_argument("--child-frame-id", default="vive_controller")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=os.environ.get("NERO_VIVE_CALIBRATION"),
        help=(
            "JSON containing map_from_lighthouse and base_from_controller transforms; "
            "enables calibrated /nero/reference pose publication"
        ),
    )
    args = parser.parse_args(argv)
    if args.stale_after <= 0 or args.rate <= 0:
        parser.error("--stale-after and --rate must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calibration = ViveCalibration.load(args.calibration) if args.calibration else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("Invalid Vive calibration: %s", error)
        return 1
    try:
        import rclpy
    except ImportError:
        logger.error("ROS 2/rclpy is unavailable; run this inside Nima's ROS-equipped runtime")
        return 1
    rclpy.init(args=None)
    bridge = ViveRosBridge(
        latest_file=args.latest_file,
        stale_after_s=args.stale_after,
        rate_hz=args.rate,
        frame_id=args.frame_id,
        child_frame_id=args.child_frame_id,
        calibration=calibration,
    )
    try:
        rclpy.spin(bridge.node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
