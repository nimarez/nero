"""Blind object-approach path following using a calibrated HTC Vive robot pose."""

from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time
from typing import Any, Callable

import numpy as np

from nero.navigation.vive_pursuit import (
    VivePathTracker,
    VivePursuitConfig,
    VivePursuitController,
    object_approach_pose,
    plan_object_approach,
)
from nero.observability import RosObservabilityPublisher
from nero.observability.topics import ObservabilityTopics
from nero.robot import RobotInterface

logger = logging.getLogger(__name__)


class VivePoseSubscriber:
    """Receive a calibrated map-frame body pose and fail-closed tracking state."""

    def __init__(
        self,
        *,
        stale_after_s: float = 0.25,
        pose_topic: str | None = None,
        tracking_topic: str = "/nero/localization/vive/tracking",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool

        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = rclpy.create_node("nero_vive_pursuit_pose")
        self._stale_after_s = stale_after_s
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._pose: np.ndarray | None = None
        self._pose_received_at: float | None = None
        self._tracking = False
        self._tracking_received_at: float | None = None
        self._subscriptions = [
            self._node.create_subscription(
                PoseStamped,
                pose_topic or ObservabilityTopics().reference_pose,
                self._on_pose,
                10,
            ),
            self._node.create_subscription(Bool, tracking_topic, self._on_tracking, 10),
        ]
        self._closed = False
        self._thread = threading.Thread(target=self._spin, name="nero-vive-pose", daemon=True)
        self._thread.start()

    def _on_pose(self, message: Any) -> None:
        orientation = message.pose.orientation
        x, y, z, w = (
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        position = message.pose.position
        values = np.array([position.x, position.y], dtype=float)
        if norm <= 1e-9 or not np.all(np.isfinite(values)):
            return
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        with self._lock:
            self._pose = np.array([values[0], values[1], yaw])
            self._pose_received_at = self._monotonic()

    def _on_tracking(self, message: Any) -> None:
        with self._lock:
            self._tracking = bool(message.data)
            self._tracking_received_at = self._monotonic()

    def current_pose(self) -> np.ndarray | None:
        now = self._monotonic()
        with self._lock:
            if (
                not self._tracking
                or self._pose is None
                or self._pose_received_at is None
                or self._tracking_received_at is None
                or now - self._pose_received_at > self._stale_after_s
                or now - self._tracking_received_at > self._stale_after_s
            ):
                return None
            return self._pose.copy()

    def _spin(self) -> None:
        while not self._closed and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=1.0)
        self._node.destroy_node()


def run_agent(
    robot: Any,
    pose_source: Any,
    args: argparse.Namespace,
    *,
    telemetry: Any | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    controller = VivePursuitController(
        VivePursuitConfig(
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
        )
    )
    object_pose = np.asarray(args.goal, dtype=float)
    goal_pose = object_approach_pose(object_pose, args.stand_off)
    initializer = getattr(robot, "initialize_locomotion_only", robot.initialize)
    initializer()
    robot.stop()
    started = monotonic()
    unavailable_since: float | None = None
    pose_seen = False
    path = None
    tracker = None
    last_log = -float("inf")
    while True:
        now = monotonic()
        if now - started > args.max_runtime:
            raise RuntimeError("blind Vive pursuit exceeded --max-runtime")
        pose = pose_source.current_pose()
        if pose is None:
            robot.stop()
            unavailable_since = now if unavailable_since is None else unavailable_since
            limit = args.loss_timeout if pose_seen else args.startup_timeout
            if now - unavailable_since > limit:
                raise RuntimeError("calibrated Vive pose/tracking is unavailable or stale")
            sleep(1.0 / args.rate)
            continue
        unavailable_since = None
        pose_seen = True
        if controller.has_reached_pose(pose, goal_pose):
            robot.stop()
            logger.info(
                "Arrived at approach pose=(%.3f, %.3f, %.3f)",
                *goal_pose,
            )
            return
        if np.linalg.norm(pose[:2] - goal_pose[:2]) <= controller.config.position_tolerance:
            command = controller.compute_path_command(pose, goal_pose[:2], goal_pose, 0.0)
            robot.set_velocity(command.linear_x, command.linear_y, command.angular_z)
            sleep(1.0 / args.rate)
            continue
        if path is None:
            try:
                path = plan_object_approach(
                    pose,
                    object_pose,
                    args.stand_off,
                    spacing=args.path_spacing,
                )
            except ValueError as error:
                raise RuntimeError(f"Vive approach path planning failed: {error}") from error
            tracker = VivePathTracker(path.points)
            if telemetry is not None:
                telemetry.publish_plan(
                    np.column_stack((path.points, np.zeros(len(path.points)))),
                    time.time(),
                )
            logger.info(
                "Planned %d-point path to approach=(%.3f, %.3f, %.3f) "
                "for object=(%.3f, %.3f, %.3f)",
                len(path.points),
                *path.goal_pose,
                *path.object_pose,
            )
        lookahead = tracker.lookahead(pose[:2], args.lookahead)
        command = controller.compute_path_command(
            pose,
            lookahead,
            path.goal_pose,
            tracker.remaining_distance(pose[:2]),
        )
        robot.set_velocity(command.linear_x, command.linear_y, command.angular_z)
        if now - last_log >= 1.0:
            logger.info(
                "pose=(%.3f, %.3f, %.3f) lookahead=(%.3f, %.3f) command=(%.3f, %.3f)",
                *pose,
                *lookahead,
                command.linear_x,
                command.angular_z,
            )
            last_log = now
        sleep(1.0 / args.rate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "YAW"),
        help="Object pose in the calibrated map frame; yaw is its forward direction",
    )
    parser.add_argument("--stand-off", type=float, default=0.5)
    parser.add_argument("--lookahead", type=float, default=0.30)
    parser.add_argument("--path-spacing", type=float, default=0.05)
    parser.add_argument("--max-velocity", type=float, default=0.10)
    parser.add_argument("--max-angular-velocity", type=float, default=0.35)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--stale-after", type=float, default=0.25)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--loss-timeout", type=float, default=1.0)
    parser.add_argument("--max-runtime", type=float, default=120.0)
    parser.add_argument("--iface", default=os.getenv("BOOSTER_NET_IF", "lo"))
    parser.add_argument(
        "--acknowledge-blind-motion",
        action="store_true",
        help="Required acknowledgement that this agent has no obstacle sensing",
    )
    args = parser.parse_args(argv)
    positive = (
        "stand_off",
        "lookahead",
        "path_spacing",
        "max_velocity",
        "max_angular_velocity",
        "rate",
        "stale_after",
        "startup_timeout",
        "loss_timeout",
        "max_runtime",
    )
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in positive):
        parser.error("motion limits, timeouts, rate, and stand-off must be positive and finite")
    if not args.acknowledge_blind_motion:
        parser.error("--acknowledge-blind-motion is required because obstacles are not sensed")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    robot = None
    pose_source = None
    telemetry = None
    try:
        robot = RobotInterface(network_interface=args.iface)
        pose_source = VivePoseSubscriber(stale_after_s=args.stale_after)
        telemetry = RosObservabilityPublisher.try_create(enabled=True)
        run_agent(robot, pose_source, args, telemetry=telemetry)
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping blind Vive pursuit")
    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    finally:
        if robot is not None:
            try:
                robot.stop()
            finally:
                robot.close()
        if pose_source is not None:
            pose_source.close()
        if telemetry is not None:
            telemetry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
