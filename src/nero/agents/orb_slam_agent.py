#!/usr/bin/env python3
"""ORB-SLAM Agent: Navigate to a detected object using ORB-SLAM based navigation.

This agent shows an external camera stream to the user, waits for an object name,
detects the object using GroundingDINO, and navigates the robot to it.

Usage:
    python -m nero.agents.orb_slam_agent --camera usb:0 --object "chair"
    python -m nero.agents.orb_slam_agent --camera rtsp://192.168.1.100/stream --object "bottle"
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2

from nero.robot import RobotInterface
from nero.utils.camera_stream import CameraStream, CameraSource
from nero.utils.visualization import Visualization
from nero.navigation.policy import NavigationPolicy, PolicyState

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nero ORB-SLAM Agent - Navigate to a detected object"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="usb:0",
        help="Camera source (usb:N, rtsp://..., http://..., or file path)",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Target object name (if not provided, will prompt at runtime)",
    )
    parser.add_argument(
        "--robot-serial",
        type=str,
        default=None,
        help="Booster robot serial number",
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=1.0,
        help="Target distance from object in meters",
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


def parse_camera_source(camera_str: str) -> tuple[str, CameraSource]:
    """Parse camera source string."""
    if camera_str.startswith("usb:"):
        return camera_str[4:], CameraSource.USB
    elif camera_str.startswith("rtsp://"):
        return camera_str, CameraSource.RTSP
    elif camera_str.startswith("http://"):
        return camera_str, CameraSource.HTTP
    else:
        return camera_str, CameraSource.FILE


def main():
    """Main entry point for ORB-SLAM agent."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse camera source
    camera_source, camera_type = parse_camera_source(args.camera)

    logger.info("Starting Nero ORB-SLAM Agent")
    logger.info(f"Camera: {camera_type.value} - {camera_source}")
    logger.info(f"Target object: {args.object or '(will prompt)'}")

    # Initialize camera
    camera = CameraStream(
        source=camera_source,
        source_type=camera_type,
        width=640,
        height=480,
        fps=30,
    )

    # Initialize robot
    try:
        robot = RobotInterface(virtual_robot_name=args.robot_serial or "")
        robot.initialize()
        logger.info("Robot connected and initialized in walk mode")
    except Exception as e:
        logger.warning(f"Robot connection failed: {e}, using mock mode")
        robot = None

    # Initialize navigation policy
    policy = NavigationPolicy(robot=robot)

    # Start camera
    if not camera.start():
        logger.error("Failed to start camera")
        sys.exit(1)

    policy.start()
    if args.object:
        policy.set_target(args.object)
        policy._goal.target_distance = args.target_distance

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
    target_object = args.object

    try:
        while not shutdown_event:
            loop_start = time.time()

            # Get camera frame
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Get current policy state
            status = policy.status

            # If no target object set and we're idle, wait for user input
            if target_object is None and status.state in (
                PolicyState.SHOWING_CAMERA,
                PolicyState.WAITING_FOR_OBJECT,
            ):
                frame_with_text = viz.draw_navigation_info(
                    frame,
                    state="idle",
                    message="Press 's' to set target object",
                    fps=camera.get_fps(),
                )
                if not args.no_display:
                    key = viz.show_stream(
                        frame_with_text, "Nero ORB-SLAM Agent", camera.get_fps()
                    )
                    if key == ord("q"):
                        shutdown_event = True
                    elif key == ord("s"):
                        target_object = input("Enter object name: ").strip()
                        if target_object:
                            policy.set_target(target_object)
                            logger.info(f"Target set to: {target_object}")
                continue

            # Step policy
            status = policy.step()

            # Draw overlay
            frame = viz.draw_navigation_info(
                frame,
                state=status.state.value,
                message=status.message,
                fps=camera.get_fps(),
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
                key = viz.show_stream(frame, "Nero ORB-SLAM Agent", camera.get_fps())
                if key == ord("q"):
                    shutdown_event = True
                elif key == ord("r") and status.state == PolicyState.ARRIVED:
                    # Reset and look for new object
                    policy.reset()
                    target_object = None
                    logger.info("Reset - ready for new target")

            # Check for completion
            if status.state == PolicyState.ARRIVED:
                logger.info(f"Arrived at {target_object}")
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
        camera.stop()
        if robot:
            robot.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
