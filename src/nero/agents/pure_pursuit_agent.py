"""Lightweight object navigation using only camera-frame pure pursuit.

This keeps the human-facing behavior of ``nero-orb-slam`` but intentionally
does not build a map or estimate a global pose.  It detects the requested
object in live RGB-D, curves toward it, and stops at the configured stand-off.

Usage:
    uv run nero-pure-pursuit --no-display
"""

from __future__ import annotations

import argparse
import enum
import logging
import signal
import time
from dataclasses import dataclass, field

import cv2

from nero.interaction import (
    K1VoiceCommandSource,
    NavigationTargetListener,
    TerminalCommandSource,
    UnixSocketCommandSource,
    safe_stand_off_distance,
)
from nero.navigation.controller import VelocityCommand
from nero.navigation.pure_pursuit import PurePursuitConfig, PurePursuitController
from nero.navigation.runtime import SensorFrame, read_sensor_frame, send_velocity
from nero.navigation.safety import SafetyMonitor, SafetyStatus
from nero.observability import RosObservabilityPublisher
from nero.perception.depth_processor import DepthProcessor
from nero.perception.detector_factory import create_object_detector
from nero.perception.object_detector import (
    ObjectDetection,
    ObjectDetector,
    configure_qualcomm_cpu_partition,
)
from nero.robot import RobotInterface
from nero.utils.visualization import Visualization

logger = logging.getLogger(__name__)


class PursuitState(enum.Enum):
    IDLE = "idle"
    WAITING_FOR_OBJECT = "waiting_for_object"
    DETECTING = "detecting"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    LOST = "lost"
    ERROR = "error"


@dataclass
class PursuitStatus:
    state: PursuitState
    message: str
    velocity_command: VelocityCommand = field(default_factory=VelocityCommand)
    detections: list[ObjectDetection] = field(default_factory=list)
    safety_status: SafetyStatus | None = None
    target: str | None = None
    stand_off_distance: float | None = None
    stand_off_tolerance: float = 0.0
    target_position_camera: list[float] | None = None
    obstacle_info: dict | None = None


class DirectPursuitPolicy:
    """Minimal RGB-D object follower with no localization dependency."""

    def __init__(
        self,
        robot,
        *,
        object_detector=None,
        controller=None,
        depth_processor=None,
        safety=None,
        search_angular_velocity: float = 0.08,
        target_timeout: float = 3.0,
        acquisition_timeout: float = 20.0,
    ) -> None:
        if search_angular_velocity < 0:
            raise ValueError("search_angular_velocity must be non-negative")
        if target_timeout <= 0:
            raise ValueError("target_timeout must be positive")
        if acquisition_timeout <= 0:
            raise ValueError("acquisition_timeout must be positive")
        self.robot = robot
        self.detector = object_detector or ObjectDetector()
        self.controller = controller or PurePursuitController()
        self.depth = depth_processor or DepthProcessor()
        self.safety = safety or SafetyMonitor()
        self.search_angular_velocity = search_angular_velocity
        self.target_timeout = target_timeout
        self.acquisition_timeout = acquisition_timeout
        self.state = PursuitState.IDLE
        self.target: str | None = None
        self.stand_off = 0.8
        self.last_sensor: SensorFrame | None = None
        self._last_seen: float | None = None
        self._search_started: float | None = None
        self._last_detection_revision: int | None = None
        self._running = False
        self._last_obstacle_info: dict | None = None

    def start(self) -> PursuitStatus:
        try:
            self.robot.initialize()
            if not self.detector.initialize():
                raise RuntimeError("No live object detector is available")
            self.safety.reset()
        except Exception:
            try:
                self.detector.close()
            except Exception:
                logger.exception("Detector cleanup failed during startup")
            try:
                self.robot.stop()
            except Exception:
                logger.exception("Robot cleanup failed during startup")
            raise
        self._running = True
        self.state = PursuitState.WAITING_FOR_OBJECT
        return self._status("Ready to receive object name")

    def supports_target(self, name: str) -> bool:
        return self.detector.supports_target(name)

    def set_target(self, name: str) -> PursuitStatus:
        resolved = self.detector.resolve_target(name)
        if resolved is None:
            return self._status(f"Object class '{name}' is not supported")
        self.detector.set_target(resolved)
        self.target = resolved
        self.stand_off = safe_stand_off_distance(resolved)
        self._last_seen = None
        self._search_started = time.monotonic()
        self._last_detection_revision = None
        self.state = PursuitState.DETECTING
        return self._status(f"Searching for '{resolved}'")

    def step(self) -> PursuitStatus:
        if not self._running:
            return self._status("Policy not running")
        try:
            sensor = read_sensor_frame(self.robot)
            self.last_sensor = sensor
            depth_m = self.depth.preprocess(sensor.depth)
            obstacles = self.depth.detect_obstacles(depth_m)
            self._last_obstacle_info = obstacles
            safety = self.safety.check_safety(
                imu_rpy=sensor.imu_rpy,
                obstacle_distance=float(obstacles["min_distance"]),
                battery_level=getattr(sensor.raw_state, "battery_level", None),
                depth_sensor_blind=bool(obstacles.get("sensor_blind", False)),
            )
        except Exception as exc:
            logger.exception("Direct pursuit sensor failure")
            self.state = PursuitState.ERROR
            self._stop_robot()
            return self._status(f"Sensor failure: {exc}")

        if not safety.is_safe:
            self._stop_robot()
            return self._status(
                f"Motion blocked: {safety.reason}", safety_status=safety
            )
        if self.target is None:
            self.state = PursuitState.WAITING_FOR_OBJECT
            self._stop_robot()
            return self._status("Waiting for object name", safety_status=safety)

        detections = self.detector.detect(
            sensor.rgb, sensor.depth, sensor.camera_info
        )
        revision = getattr(self.detector, "result_revision", None)
        new_detection_result = (
            revision is None or revision != self._last_detection_revision
        )
        if new_detection_result:
            self._last_detection_revision = revision
        target = self.detector.find_object(detections, self.target)
        if target is None or target.position_3d is None:
            return self._search(detections, safety, obstacles)

        now = time.monotonic()
        if new_detection_result:
            self._last_seen = now
        elif self._last_seen is None or now - self._last_seen > self.target_timeout:
            return self._target_lost(detections, safety)
        self.state = PursuitState.NAVIGATING
        try:
            arrived = self.controller.has_arrived(target.position_3d, self.stand_off)
            command = (
                VelocityCommand()
                if arrived
                else self.controller.compute_command(target.position_3d, self.stand_off)
            )
        except ValueError as exc:
            self._stop_robot()
            return self._status(
                f"Invalid target depth: {exc}",
                detections=detections,
                safety_status=safety,
                target_position=target.position_3d,
            )

        if arrived:
            self.state = PursuitState.ARRIVED
            self._stop_robot()
            return self._status(
                f"Holding stand-off from '{self.target}'",
                detections=detections,
                safety_status=safety,
                target_position=target.position_3d,
            )

        # Never translate into a blocked center corridor. Turning remains
        # allowed so the live target can be reacquired around an obstacle.
        if not obstacles.get("center_clear", False):
            command = VelocityCommand(angular_z=command.angular_z)
        try:
            send_velocity(self.robot, command)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return self._status(
            f"Pursuing '{self.target}'",
            command=command,
            detections=detections,
            safety_status=safety,
            target_position=target.position_3d,
        )

    def _search(self, detections, safety, obstacles) -> PursuitStatus:
        now = time.monotonic()
        acquired = self._last_seen is not None
        reference = self._last_seen if acquired else self._search_started
        reference = now if reference is None else reference
        timeout = self.target_timeout if acquired else self.acquisition_timeout
        if now - reference > timeout:
            return self._target_lost(detections, safety)
        self.state = PursuitState.DETECTING
        angular = (
            self.search_angular_velocity
            if not obstacles.get("has_obstacle", True)
            else 0.0
        )
        command = VelocityCommand(angular_z=angular)
        try:
            send_velocity(self.robot, command)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return self._status(
            f"Searching for '{self.target}'",
            command=command,
            detections=detections,
            safety_status=safety,
        )

    def _target_lost(self, detections, safety) -> PursuitStatus:
        self.state = PursuitState.LOST
        self._stop_robot()
        return self._status(
            f"Could not find '{self.target}'",
            detections=detections,
            safety_status=safety,
        )

    def reset(self) -> PursuitStatus:
        self._stop_robot()
        self.target = None
        self._last_seen = None
        self._search_started = None
        self._last_detection_revision = None
        self.state = PursuitState.WAITING_FOR_OBJECT
        return self._status("Ready for another object command")

    def stop(self) -> None:
        self._running = False
        try:
            self._stop_robot()
        finally:
            try:
                self.detector.close()
            finally:
                self.state = PursuitState.IDLE

    def _stop_robot(self) -> None:
        """Hold position even if Booster already dropped walking control."""
        try:
            send_velocity(self.robot)
        except RuntimeError:
            logger.exception("Robot locomotion controller rejected the stop command")

    def _locomotion_error(self, exc, detections, safety) -> PursuitStatus:
        self.state = PursuitState.ERROR
        self._stop_robot()
        return self._status(
            f"Locomotion command failed: {exc}",
            detections=detections,
            safety_status=safety,
        )

    def _status(
        self,
        message: str,
        *,
        command: VelocityCommand | None = None,
        detections: list[ObjectDetection] | None = None,
        safety_status: SafetyStatus | None = None,
        target_position=None,
    ) -> PursuitStatus:
        return PursuitStatus(
            state=self.state,
            message=message,
            velocity_command=command or VelocityCommand(),
            detections=detections or [],
            safety_status=safety_status,
            target=self.target,
            stand_off_distance=self.stand_off if self.target is not None else None,
            stand_off_tolerance=self.controller.config.position_tolerance,
            target_position_camera=(
                None
                if target_position is None
                else [float(value) for value in target_position]
            ),
            obstacle_info=self._last_obstacle_info,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nero direct RGB-D pure-pursuit object agent"
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--command-source",
        choices=("socket", "terminal", "voice"),
        default="socket",
    )
    parser.add_argument("--command-socket", default="/tmp/nero-navigation.sock")
    parser.add_argument(
        "--no-ros-observability",
        action="store_true",
        help="Disable normalized /nero ROS 2 telemetry topics",
    )
    parser.add_argument(
        "--object-backend",
        help="Detector backend (default: NERO_OBJECT_BACKEND or K1 QNN)",
    )
    parser.add_argument(
        "--aruco-map",
        help="JSON marker-ID to object-name mapping (or set NERO_ARUCO_MAP)",
    )
    parser.add_argument(
        "--aruco-dictionary",
        help="OpenCV ArUco dictionary (default: DICT_4X4_50)",
    )
    parser.add_argument("--max-velocity", type=float, default=0.25)
    parser.add_argument("--max-angular-velocity", type=float, default=0.7)
    parser.add_argument("--target-timeout", type=float, default=3.0)
    parser.add_argument("--acquisition-timeout", type=float, default=20.0)
    parser.add_argument("--search-angular-velocity", type=float, default=0.12)
    return parser.parse_args()


def run_agent(robot, args, *, object_detector=None, command_source=None) -> None:
    controller = PurePursuitController(
        PurePursuitConfig(
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
        )
    )
    policy = DirectPursuitPolicy(
        robot,
        object_detector=object_detector,
        controller=controller,
        target_timeout=args.target_timeout,
        acquisition_timeout=args.acquisition_timeout,
        search_angular_velocity=getattr(args, "search_angular_velocity", 0.12),
    )
    shutdown = False
    policy_started = False
    listener = None
    telemetry = None

    def handle_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    try:
        policy.start()
        policy_started = True
        telemetry = RosObservabilityPublisher.try_create(
            enabled=not getattr(args, "no_ros_observability", False)
        )
        signal.signal(signal.SIGTERM, handle_signal)
        commands = command_source or TerminalCommandSource()
        listener = NavigationTargetListener(
            robot,
            commands,
            cancelled=lambda: shutdown,
            target_validator=policy.supports_target,
        )
        listener.start()
        viz = Visualization()
        target_name = None
        announced_arrival = False
        announced_failure = False

        while not shutdown:
            started = time.monotonic()
            if target_name is None:
                target_name = listener.poll()
                if target_name is not None:
                    policy.set_target(target_name)
                    announced_arrival = False
                    announced_failure = False

            status = policy.step()
            sensor = policy.last_sensor
            if sensor is not None and telemetry is not None:
                if sensor.raw_state is not None:
                    telemetry.publish_robot_state(sensor.raw_state, robot)
                telemetry.publish_policy(status, sensor.timestamp)
            if status.state == PursuitState.ERROR:
                logger.error("Direct pursuit stopped: %s", status.message)
                break
            if sensor is not None and not args.no_display:
                frame = viz.draw_navigation_info(
                    sensor.rgb,
                    state=status.state.value,
                    message=status.message,
                    fps=20.0,
                    velocity=(
                        status.velocity_command.linear_x,
                        status.velocity_command.angular_z,
                    ),
                )
                key = viz.show_stream(frame, "Nero Pure Pursuit Agent", 20.0)
                if key == ord("q"):
                    shutdown = True
                elif key == ord("r"):
                    policy.reset()
                    target_name = None
                    announced_arrival = False
                    listener.start()

            if status.state == PursuitState.ARRIVED and not announced_arrival:
                try:
                    robot.speak(f"Arrived at {target_name}.")
                except RuntimeError as exc:
                    logger.warning("Could not announce arrival: %s", exc)
                announced_arrival = True
                if args.no_display:
                    policy.reset()
                    target_name = None
                    announced_arrival = False
                    listener.start()
            elif status.state == PursuitState.LOST:
                if not announced_failure and target_name is not None:
                    try:
                        robot.speak(f"I could not detect the {target_name}.")
                    except RuntimeError as exc:
                        logger.warning("Could not announce missing object: %s", exc)
                    announced_failure = True
                policy.reset()
                target_name = None
                announced_arrival = False
                listener.start()

            elapsed = time.monotonic() - started
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    except (KeyboardInterrupt, EOFError, InterruptedError):
        logger.info("Stopping direct pursuit agent")
    finally:
        if policy_started:
            try:
                policy.stop()
            except Exception:
                logger.exception("Direct pursuit policy cleanup failed")
        try:
            robot.stop()
        except Exception:
            logger.exception("Robot cleanup failed")
        if listener is not None:
            try:
                listener.close()
            except Exception:
                logger.exception("Command listener cleanup failed")
        if telemetry is not None:
            telemetry.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    configure_qualcomm_cpu_partition(args.object_backend)
    try:
        object_detector = create_object_detector(
            backend=args.object_backend,
            aruco_map=args.aruco_map,
            aruco_dictionary=args.aruco_dictionary,
        )
    except ValueError as exc:
        logger.error("Invalid object detector configuration: %s", exc)
        raise SystemExit(2) from exc
    robot = RobotInterface()
    if args.command_source == "socket":
        command_source = UnixSocketCommandSource(args.command_socket)
    elif args.command_source == "terminal":
        command_source = TerminalCommandSource()
    else:
        command_source = K1VoiceCommandSource()
        command_source.start_listening()
        command_source.stop_listening()
    run_agent(
        robot,
        args,
        object_detector=object_detector,
        command_source=command_source,
    )


if __name__ == "__main__":
    main()
