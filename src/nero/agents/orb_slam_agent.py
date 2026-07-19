"""ORB-SLAM Agent: obey spoken object-navigation directions.

This agent listens for ``go to <object>``, acknowledges the direction aloud,
and uses the K1's built-in RGB-D camera to navigate to that object.

Usage:
    uv run nero-orb-slam
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time

import cv2

from nero.robot import RobotInterface
from nero.interaction import (
    K1VoiceCommandSource,
    NavigationTargetListener,
    TerminalCommandSource,
    UnixSocketCommandSource,
)
from nero.utils.visualization import Visualization
from nero.navigation.policy import NavigationPolicy, PolicyState
from nero.navigation.map_policy import MapNavConfig
from nero.navigation.global_localization import GlobalLocalizationConfig
from nero.observability import RosObservabilityPublisher
from nero.perception.aruco_detector import ArucoObjectDetector
from nero.perception.object_detector import ObjectDetector, configure_qualcomm_cpu_partition

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nero ORB-SLAM Agent - Navigate to a detected object"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visual display",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-ros-observability",
        action="store_true",
        help="Disable normalized /nero ROS 2 telemetry topics",
    )
    parser.add_argument(
        "--command-source",
        choices=("socket", "terminal", "voice"),
        default="socket",
        help="Human command transport (default: robot-local socket for nero-command)",
    )
    parser.add_argument(
        "--command-socket",
        default="/tmp/nero-navigation.sock",
        help="Robot-local Unix socket used by the socket command source",
    )
    parser.add_argument(
        "--object-backend",
        help="Detector backend (default: NERO_OBJECT_BACKEND or K1 QNN; use 'aruco' for markers)",
    )
    parser.add_argument(
        "--aruco-map",
        help="JSON marker-ID to object-name mapping (or set NERO_ARUCO_MAP)",
    )
    parser.add_argument(
        "--aruco-dictionary",
        default=None,
        help="OpenCV dictionary name (default: NERO_ARUCO_DICTIONARY or DICT_4X4_50)",
    )
    parser.add_argument(
        "--map",
        help="Optional occupancy map; enables map alignment and A* for object goals",
    )
    parser.add_argument("--map-yaml", help="YAML metadata for a PNG occupancy map")
    parser.add_argument(
        "--initial-pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Known startup pose in the map; omit for depth-scan localization",
    )
    parser.add_argument("--map-resolution", type=float, default=0.05)
    parser.add_argument(
        "--map-origin", nargs=2, type=float, metavar=("X", "Y"), default=(0.0, 0.0)
    )
    parser.add_argument("--map-inflation", type=float, default=0.3)
    parser.add_argument(
        "--camera-height", type=float, default=GlobalLocalizationConfig.camera_height
    )
    parser.add_argument("--localization-spin-speed", type=float, default=0.3)
    return parser.parse_args()


def build_object_detector(args: argparse.Namespace):
    """Build the selected detector without exposing K1 camera parameters."""
    backend = getattr(args, "object_backend", None) or os.getenv("NERO_OBJECT_BACKEND")
    if backend and backend.strip().lower().replace("_", "-") == "aruco":
        return ArucoObjectDetector(
            mapping_path=getattr(args, "aruco_map", None),
            dictionary=getattr(args, "aruco_dictionary", None),
        )
    if getattr(args, "aruco_map", None):
        raise ValueError("--aruco-map requires --object-backend aruco")
    return ObjectDetector(backend=backend)


def run_agent(
    robot,
    args: argparse.Namespace,
    *,
    slam_options=None,
    object_detector=None,
    command_source=None,
    map_config=None,
) -> None:
    """Run the shared object-following loop with a robot environment adapter."""
    # Initialize navigation policy
    policy = NavigationPolicy(
        robot=robot,
        slam_options=slam_options,
        object_detector=object_detector,
        map_config=map_config,
    )

    try:
        policy.start()
    except Exception:
        try:
            robot.stop()
        except Exception:
            logger.exception("Robot cleanup failed after policy startup error")
        raise
    telemetry = RosObservabilityPublisher.try_create(
        enabled=not getattr(args, "no_ros_observability", False)
    )

    # Signal handler
    shutdown_event = False

    def signal_handler(sig, frame):
        nonlocal shutdown_event
        shutdown_event = True
        logger.info("Shutdown signal received")

    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    logger.info("Starting ORB-SLAM agent loop (press Ctrl+C to stop)")
    # Match the production K1 Geek RGB-D stream; RobotInterface additionally
    # waits for a new synchronized frame so stale images are never reprocessed.
    loop_rate = 20
    loop_interval = 1.0 / loop_rate
    viz = Visualization()
    target_object = None
    announced_arrival = False
    announced_failure = False
    commands = command_source or TerminalCommandSource()
    target_listener = NavigationTargetListener(
        robot,
        commands,
        cancelled=lambda: shutdown_event,
        target_validator=getattr(policy, "supports_target", None),
    )
    target_listener.start()

    try:
        while not shutdown_event:
            loop_start = time.time()

            # Human input is acquired in the background. Sensing, SLAM, safety,
            # and visualization remain live while no target has been requested.
            if target_object is None:
                try:
                    target_object = target_listener.poll()
                except (EOFError, KeyboardInterrupt, InterruptedError):
                    shutdown_event = True
                    break
                if target_object is not None:
                    policy.set_target(target_object)
                    announced_arrival = False
                    announced_failure = False
                    logger.info("Accepted navigation target: %s", target_object)

            # The policy owns the single synchronized K1 sensor read used by
            # localization, safety, detection, display, and observability.
            status = policy.step()
            if status.state == PolicyState.ERROR:
                logger.error("Navigation stopped: %s", status.message)
                break
            sensor = policy.last_sensor
            if sensor is None:
                time.sleep(0.01)
                continue
            frame = sensor.rgb
            sensor_timestamp = sensor.timestamp
            if telemetry is not None:
                if sensor.raw_state is not None:
                    telemetry.publish_robot_state(sensor.raw_state, robot)
                telemetry.publish_policy(status, sensor_timestamp)
                if policy.slam is not None:
                    slam_pose = policy.slam.get_current_pose()
                    if slam_pose is not None:
                        telemetry.publish_tracking(
                            slam_pose.tracking_status, slam_pose.num_map_points
                        )
                    map_points = policy.slam.get_map_points()
                    if len(map_points):
                        if policy.map_navigator is None:
                            telemetry.publish_point_cloud(map_points, sensor_timestamp)
                        elif policy.map_alignment_ready:
                            telemetry.publish_point_cloud(
                                policy.transform_slam_points(map_points),
                                sensor_timestamp,
                            )

            # Draw overlay
            frame = viz.draw_navigation_info(
                frame,
                state=status.state.value,
                message=status.message,
                fps=loop_rate,
                velocity=(
                    (
                        status.velocity_command.linear_x,
                        status.velocity_command.angular_z,
                    )
                    if status.velocity_command
                    else None
                ),
            )

            # Draw detection info if detecting
            if status.state == PolicyState.DETECTING:
                cv2.putText(
                    frame,
                    f"Looking for: {target_object}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Display
            if not args.no_display:
                key = viz.show_stream(frame, "Nero ORB-SLAM Agent", loop_rate)
                if key == ord("q"):
                    shutdown_event = True
                elif key == ord("r") and status.state == PolicyState.ARRIVED:
                    policy.reset()
                    target_object = None
                    announced_arrival = False
                    target_listener.start()
                    logger.info("Reset - ready for new target")

            # Check for completion
            if status.state == PolicyState.ARRIVED and not announced_arrival:
                logger.info(f"Arrived at {target_object}")
                try:
                    robot.speak(f"Arrived at {target_object}.")
                except RuntimeError as e:
                    logger.warning(f"Could not announce arrival: {e}")
                announced_arrival = True
                if args.no_display:
                    policy.reset()
                    target_object = None
                    announced_arrival = False
                    target_listener.start()
                    logger.info("Ready for another object command")
                else:
                    logger.info(
                        "Press 'r' to reset and find another object, 'q' to quit"
                    )

            if status.state == PolicyState.LOST and target_object is not None:
                logger.info("Target lost; ready for another object command")
                if not announced_failure:
                    try:
                        robot.speak(f"I could not detect the {target_object}.")
                    except RuntimeError as e:
                        logger.warning("Could not announce missing object: %s", e)
                    announced_failure = True
                policy.reset()
                target_object = None
                announced_arrival = False
                target_listener.start()

            # Maintain loop rate
            elapsed = time.time() - loop_start
            if elapsed < loop_interval:
                time.sleep(loop_interval - elapsed)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        # Cleanup
        logger.info("Shutting down...")
        policy.stop()
        try:
            robot.stop()
        except RuntimeError as exc:
            logger.warning("Robot locomotion controller rejected final stop: %s", exc)
        target_listener.close()
        if telemetry is not None:
            telemetry.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


def main():
    """Main entry point for ORB-SLAM agent."""
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    configure_qualcomm_cpu_partition(args.object_backend)

    logger.info("Starting Nero ORB-SLAM Agent")
    logger.info("Sensors: K1 built-in RGB-D camera")

    try:
        robot = RobotInterface()
        robot.initialize()
        logger.info("Robot connected and initialized in walk mode")
    except Exception as e:
        logger.error(f"Failed to connect to K1 robot: {e}")
        raise SystemExit(1) from e

    if args.command_source == "socket":
        command_source = UnixSocketCommandSource(args.command_socket)
    elif args.command_source == "terminal":
        command_source = TerminalCommandSource()
    else:
        command_source = K1VoiceCommandSource()
        # Verify that the robot-side LUI service is actually reachable. The SDK
        # client and DDS topic can initialize even when that service is absent.
        command_source.start_listening()
        command_source.stop_listening()

    map_config = None
    if args.map:
        auto_localize = args.initial_pose is None
        map_config = MapNavConfig(
            map_path=args.map,
            yaml_path=args.map_yaml,
            resolution=args.map_resolution,
            origin=tuple(args.map_origin),
            initial_pose=(0.0, 0.0, 0.0) if auto_localize else tuple(args.initial_pose),
            auto_localize=auto_localize,
            localization=GlobalLocalizationConfig(camera_height=args.camera_height),
            localization_spin_speed=args.localization_spin_speed,
            inflation_radius=args.map_inflation,
        )
        logger.info(
            "Fixed-map mode enabled (%s localization)",
            "automatic" if auto_localize else "known-pose",
        )

    try:
        object_detector = build_object_detector(args)
    except ValueError as exc:
        logger.error("Invalid object detector configuration: %s", exc)
        raise SystemExit(2) from exc

    run_agent(
        robot,
        args,
        object_detector=object_detector,
        command_source=command_source,
        map_config=map_config,
    )


if __name__ == "__main__":
    main()
