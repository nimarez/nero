"""ORB-SLAM Agent: obey spoken object-navigation directions.

This agent listens for ``go to <object>``, acknowledges the direction aloud,
and uses the K1's built-in RGB-D camera to navigate to that object.

Usage:
    uv run nero-orb-slam
"""

from __future__ import annotations

import argparse
import logging
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
from nero.observability import RosObservabilityPublisher

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
    return parser.parse_args()


def run_agent(
    robot,
    args: argparse.Namespace,
    *,
    slam_options=None,
    object_detector=None,
    command_source=None,
) -> None:
    """Run the shared object-following loop with a robot environment adapter."""
    # Initialize navigation policy
    policy = NavigationPolicy(
        robot=robot, slam_options=slam_options, object_detector=object_detector
    )

    policy.start()
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
    loop_rate = 30
    loop_interval = 1.0 / loop_rate
    viz = Visualization()
    target_object = None
    announced_arrival = False
    commands = command_source or TerminalCommandSource()
    target_listener = NavigationTargetListener(
        robot, commands, cancelled=lambda: shutdown_event
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
                    logger.info("Accepted navigation target: %s", target_object)

            # Read the K1's built-in RGB-D stream.
            try:
                peek_state = getattr(robot, "peek_state", robot.get_state)
                state = peek_state(include_images=True)
                frame = robot.image_to_array(state.rgb)
                sensor_timestamp = robot.image_timestamp(state.rgb)
                if telemetry is not None:
                    telemetry.publish_robot_state(state, robot)
            except Exception as e:
                logger.warning(f"Failed to read K1 sensors: {e}")
                time.sleep(0.01)
                continue

            # Step policy
            status = policy.step()
            if telemetry is not None:
                telemetry.publish_policy(status, sensor_timestamp)
                if policy.slam is not None:
                    slam_pose = policy.slam.get_current_pose()
                    if slam_pose is not None:
                        telemetry.publish_tracking(
                            slam_pose.tracking_status, slam_pose.num_map_points
                        )
                    map_points = policy.slam.get_map_points()
                    if len(map_points):
                        telemetry.publish_point_cloud(map_points, sensor_timestamp)

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
        robot.stop()
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

    run_agent(robot, args, command_source=command_source)


if __name__ == "__main__":
    main()
