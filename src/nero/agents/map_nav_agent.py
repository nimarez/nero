"""Map-based navigation agent entry point.

Navigates using a pre-built static map without SLAM.
Uses visual odometry for localization and A* for path planning.

Usage:
    nero-map-nav --map maps/office.png --yaml maps/office.yaml --camera usb:0
    nero-map-nav --map maps/office.npy --goal 3.5 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from nero.navigation import (
    MapNavConfig,
    MapNavState,
    MapNavigationPolicy,
)
from nero.robot import RobotWrapper
from nero.utils.camera_stream import CameraStream

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map-based navigation agent")
    parser.add_argument(
        "--map", required=True, help="Path to occupancy grid map (PNG or .npy)"
    )
    parser.add_argument("--yaml", help="Path to map YAML metadata (for PNG maps)")
    parser.add_argument(
        "--camera", default="usb:0", help="Camera source (usb:0, rtsp://..., file.mp4)"
    )
    parser.add_argument(
        "--depth-camera",
        help="Depth camera source (optional, for scale recovery)",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Goal position in world coordinates (meters)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05, help="Map resolution (m/px)"
    )
    parser.add_argument(
        "--origin",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=(0.0, 0.0),
        help="Map origin in world coords",
    )
    parser.add_argument(
        "--inflation", type=float, default=0.3, help="Obstacle inflation radius (m)"
    )
    parser.add_argument(
        "--max-vel", type=float, default=0.3, help="Max linear velocity (m/s)"
    )
    parser.add_argument("--no-depth", action="store_true", help="Disable depth-assisted VO")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Create config
    config = MapNavConfig(
        map_path=args.map,
        yaml_path=args.yaml,
        resolution=args.resolution,
        origin=tuple(args.origin),
        use_depth=not args.no_depth,
        inflation_radius=args.inflation,
        max_linear_vel=args.max_vel,
    )

    # Create policy
    policy = MapNavigationPolicy(config)

    # Load map
    if not policy.load_map():
        logger.error("Failed to load map. Exiting.")
        sys.exit(1)

    # Open camera
    camera = CameraStream(args.camera)
    if not camera.open():
        logger.error(f"Failed to open camera: {args.camera}")
        sys.exit(1)

    # Open depth camera if provided
    depth_camera: Optional[CameraStream] = None
    if args.depth_camera:
        depth_camera = CameraStream(args.depth_camera)
        if not depth_camera.open():
            logger.error(f"Failed to open depth camera: {args.depth_camera}")
            sys.exit(1)

    # Initialize robot
    try:
        robot = RobotWrapper()
        robot.initialize()
        logger.info("Robot connected and initialized in walk mode")
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        sys.exit(1)

    try:
        # Get first frame and init odometry
        frame = camera.read()
        if frame is None:
            logger.error("Failed to read from camera")
            sys.exit(1)

        depth = None
        if depth_camera is not None:
            depth = depth_camera.read()

        policy.init_odometry(frame)

        # Set goal if provided
        if args.goal:
            policy.set_goal(args.goal[0], args.goal[1])

        # Main loop
        logger.info("Map navigation agent started. Press 'q' to quit, 'c' to click goal on map.")
        running = True
        click_mode = False

        while running:
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame")
                continue

            depth = None
            if depth_camera is not None:
                depth = depth_camera.read()

            # Run policy
            vx, vy, vyaw = policy.update(frame, depth)

            # Send velocity command
            if policy.state not in (MapNavState.IDLE, MapNavState.ARRIVED, MapNavState.LOST):
                robot.set_velocity(vx=vx, vy=vy, vyaw=vyaw)
            else:
                robot.set_velocity(0.0, 0.0, 0.0)

            # Print state
            pose = policy.current_pose
            logger.info(
                f"State: {policy.state.value} | "
                f"Pose: ({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.2f}) | "
                f"Vel: ({vx:.2f}, {vy:.2f}, {vyaw:.2f})"
            )

            # Visualization
            if not args.headless:
                # Show camera view
                cv2.imshow("Camera", frame)

                # Show map view
                map_view = policy.render_map()
                cv2.imshow("Map", map_view)

                # Handle clicks on map
                if click_mode:
                    cv2.setMouseCallback("Map", lambda e, x, y, f, p: None)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                elif key == ord("r"):
                    policy.reset()
                    logger.info("Policy reset")
                elif key == ord("c"):
                    click_mode = not click_mode
                    logger.info(f"Click mode: {'ON' if click_mode else 'OFF'} - click on map to set goal")
                elif key == ord("s") and policy.state == MapNavState.ARRIVED:
                    # Prompt for new goal
                    try:
                        x = float(input("Goal X (m): "))
                        y = float(input("Goal Y (m): "))
                        policy.set_goal(x, y)
                    except ValueError:
                        logger.warning("Invalid input")

            # Check for completion
            if policy.state == MapNavState.ARRIVED:
                logger.info("Arrived at goal!")
                if args.headless:
                    break

        # Stop robot
        robot.set_velocity(0.0, 0.0, 0.0)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        robot.set_velocity(0.0, 0.0, 0.0)
        camera.close()
        if depth_camera is not None:
            depth_camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()