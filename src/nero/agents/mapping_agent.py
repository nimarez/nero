#!/usr/bin/env python3
"""Mapping Agent: Autonomous space exploration and Gaussian splat reconstruction.

This agent explores a space while collecting RGB-D frames with poses,
then trains a 3D Gaussian Splat model for viewing.

Usage:
    python -m nero.agents.mapping_agent --camera usb:0 --pattern spiral --max-frames 500
    python -m nero.agents.mapping_agent --camera rtsp://192.168.1.100/stream --output-dir /path/to/output
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
from nero.mapping.mapping_policy import MappingPolicy, MappingState, MappingConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nero Mapping Agent - Gaussian Splat Reconstruction"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="usb:0",
        help="Camera source (usb:N, rtsp://..., http://..., or file path)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="spiral",
        choices=["spiral", "boustrophedon", "random"],
        help="Exploration pattern",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to collect",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Save every Nth frame",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Exploration speed (m/s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/splats",
        help="Output directory for splat files",
    )
    parser.add_argument(
        "--robot-serial",
        type=str,
        default=None,
        help="Booster robot serial number",
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
    """Main entry point for mapping agent."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse camera source
    camera_source, camera_type = parse_camera_source(args.camera)

    logger.info("Starting Nero Mapping Agent")
    logger.info(f"Camera: {camera_type.value} - {camera_source}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info(f"Max frames: {args.max_frames}")

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
        logger.info("Robot connected")
    except Exception as e:
        logger.warning(f"Robot connection failed: {e}, using mock mode")
        robot = None

    # Initialize mapping config
    mapping_config = MappingConfig(
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        exploration_speed=args.speed,
        coverage_pattern=args.pattern,
        output_dir=args.output_dir,
    )

    # Initialize mapping policy
    policy = MappingPolicy(
        robot=robot,
        slam_config={
            "voc_path": "config/ORBvoc.txt",
            "settings_path": "config/orbslam3_settings.yaml",
        },
        mapping_config=mapping_config,
        safety_config={
            "min_obstacle_distance": 0.5,
        },
    )

    # Start camera
    if not camera.start():
        logger.error("Failed to start camera")
        sys.exit(1)

    # Start mapping
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
    logger.info("Starting mapping loop (press Ctrl+C to stop)")
    loop_rate = 30
    loop_interval = 1.0 / loop_rate
    viz = Visualization()

    try:
        while not shutdown_event:
            loop_start = time.time()

            # Step policy
            status = policy.step()

            # Get camera frame
            frame = camera.get_frame()
            if frame is not None:
                # Draw mapping info overlay
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

                # Draw progress bar
                progress = status.frames_collected / status.max_frames
                h, w = frame.shape[:2]
                bar_width = int(w * 0.8)
                bar_height = 10
                bar_x = int(w * 0.1)
                bar_y = h - 30

                # Background
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (50, 50, 50),
                    -1,
                )
                # Progress
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + int(bar_width * progress), bar_y + bar_height),
                    (0, 255, 0),
                    -1,
                )
                # Text
                cv2.putText(
                    frame,
                    f"Frames: {status.frames_collected}/{status.max_frames} ({progress*100:.0f}%)",
                    (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # Display
                if not args.no_display:
                    key = viz.show_stream(frame, "Nero Mapping Agent", camera.get_fps())
                    if key == ord("q"):
                        shutdown_event = True
                    elif key == ord("t") and status.state == MappingState.EXPLORING:
                        # Trigger training manually
                        logger.info("Manual training trigger")
                        policy.start_training()

            # Check for completion
            if status.state in (MappingState.COMPLETE, MappingState.ERROR):
                logger.info(f"Mapping finished: {status.message}")
                if status.state == MappingState.COMPLETE:
                    logger.info(f"Trajectory length: {status.trajectory_length:.1f}m")
                    logger.info(f"Output: {args.output_dir}")
                break

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
