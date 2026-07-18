"""ORB-SLAM Agent: Navigate to a detected object using ORB-SLAM based navigation.

This agent uses the K1's built-in RGB-D camera, announces detected objects,
waits for human confirmation, and navigates to the confirmed object.

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
from nero.interaction import announce_and_confirm, deduce_target_distance
from nero.utils.visualization import Visualization
from nero.navigation.policy import NavigationPolicy, PolicyState

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
    return parser.parse_args()


def main():
    """Main entry point for ORB-SLAM agent."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Nero ORB-SLAM Agent")
    logger.info("Sensors: K1 built-in RGB-D camera")

    # Initialize robot
    try:
        robot = RobotInterface()
        robot.initialize()
        logger.info("Robot connected and initialized in walk mode")
    except Exception as e:
        logger.error(f"Failed to connect to K1 robot: {e}")
        raise SystemExit(1) from e

    # Initialize navigation policy
    policy = NavigationPolicy(robot=robot)

    policy.start()

    # Signal handler
    shutdown_event = False

    def signal_handler(sig, frame):
        nonlocal shutdown_event
        shutdown_event = True
        logger.info("Shutdown signal received")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    logger.info("Starting ORB-SLAM agent loop (press Ctrl+C to stop)")
    loop_rate = 30
    loop_interval = 1.0 / loop_rate
    viz = Visualization()
    target_object = None
    last_announce_time: dict[str, float] = {}
    announce_cooldown = 5.0
    announced_arrival = False

    try:
        while not shutdown_event:
            loop_start = time.time()

            # Read the K1's built-in RGB-D stream.
            try:
                state = robot.get_state(include_images=True)
                frame = robot.image_to_array(state.rgb)
                depth = robot.image_to_array(state.depth)
            except Exception as e:
                logger.warning(f"Failed to read K1 sensors: {e}")
                time.sleep(0.01)
                continue

            # Get current policy state
            status = policy.status

            # Scan continuously until a human confirms one detected object.
            if target_object is None and status.state in (
                PolicyState.SHOWING_CAMERA,
                PolicyState.WAITING_FOR_OBJECT,
            ):
                detections = policy.object_detector.detect(
                    frame, depth, state.camera_info
                )
                now = time.monotonic()
                for detection in detections:
                    object_name = detection.label.lower()
                    if (
                        now - last_announce_time.get(object_name, 0.0)
                        < announce_cooldown
                    ):
                        continue
                    last_announce_time[object_name] = now
                    try:
                        should_follow = announce_and_confirm(robot, object_name)
                    except RuntimeError as e:
                        logger.error(f"Could not announce detection: {e}")
                        should_follow = False
                    if should_follow:
                        target_object = object_name
                        target_distance = deduce_target_distance(
                            object_name, detection.distance
                        )
                        policy.set_target(object_name)
                        policy._goal.target_distance = target_distance
                        logger.info(
                            f"Confirmed target: {object_name}; "
                            f"stopping distance: {target_distance:.2f}m"
                        )
                        break

                frame_with_text = viz.draw_navigation_info(
                    frame,
                    state="idle",
                    message="Scanning for objects...",
                    fps=loop_rate,
                )
                if not args.no_display:
                    key = viz.show_stream(
                        frame_with_text, "Nero ORB-SLAM Agent", loop_rate
                    )
                    if key == ord("q"):
                        shutdown_event = True
                if target_object is None:
                    continue

            # Step policy
            status = policy.step()

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
                    logger.info("Reset - ready for new target")

            # Check for completion
            if status.state == PolicyState.ARRIVED and not announced_arrival:
                logger.info(f"Arrived at {target_object}")
                try:
                    robot.speak(f"Arrived at {target_object}.")
                except RuntimeError as e:
                    logger.warning(f"Could not announce arrival: {e}")
                announced_arrival = True
                if not args.no_display:
                    logger.info(
                        "Press 'r' to reset and find another object, 'q' to quit"
                    )

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
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
