"""Publish the POS Vive pose endpoint as standard ROS 2 messages."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from nero.vive.udp_transport import DEFAULT_STALE_AFTER_S, PosePacket

logger = logging.getLogger(__name__)

DEFAULT_LATEST_FILE = "/run/nero/vive_pose.json"
DEFAULT_RATE_HZ = 100.0
POSE_TOPIC = "/nero/localization/vive/controller_pose"
ODOMETRY_TOPIC = "/nero/localization/vive/controller_odometry"
TRACKING_TOPIC = "/nero/localization/vive/tracking"
DIAGNOSTICS_TOPIC = "/nero/localization/vive/diagnostics"


@dataclass(frozen=True, slots=True)
class VivePoseState:
    """A validated snapshot read from the POS atomic pose endpoint."""

    packet: PosePacket
    received_at: float
    age_s: float


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
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Bool, String

        self._rclpy = rclpy
        self._node = rclpy.create_node("nero_vive_pose_bridge")
        self._types = {
            "PoseStamped": PoseStamped,
            "Odometry": Odometry,
            "Bool": Bool,
            "String": String,
        }
        self._latest_file = latest_file
        self._stale_after_s = stale_after_s
        self._frame_id = frame_id
        self._child_frame_id = child_frame_id
        sensor_qos = rclpy.qos.qos_profile_sensor_data
        self._pose_publisher = self._node.create_publisher(PoseStamped, POSE_TOPIC, sensor_qos)
        self._odometry_publisher = self._node.create_publisher(
            Odometry, ODOMETRY_TOPIC, sensor_qos
        )
        self._tracking_publisher = self._node.create_publisher(Bool, TRACKING_TOPIC, 10)
        self._diagnostics_publisher = self._node.create_publisher(String, DIAGNOSTICS_TOPIC, 10)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-file", default=DEFAULT_LATEST_FILE)
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--frame-id", default="lighthouse_world")
    parser.add_argument("--child-frame-id", default="vive_controller")
    args = parser.parse_args(argv)
    if args.stale_after <= 0 or args.rate <= 0:
        parser.error("--stale-after and --rate must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
