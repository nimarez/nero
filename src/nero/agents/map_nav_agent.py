"""Map-based navigation agent entry point.

Plans through a static occupancy map while the shared IMU-RGBD ORB-SLAM
runtime provides localization, safety checks, and local obstacle avoidance.

Usage:
    uv run nero-map-nav --map maps/office.png --yaml maps/office.yaml \
        --goal 3.5 2.0 1.57

The startup pose in the map frame is localized automatically by matching the
first depth scan against the fixed map; pass --initial-pose X Y YAW to skip
that and use a measured pose instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import cv2

from nero.navigation import (
    GlobalLocalizationConfig,
    MapNavConfig,
)
from nero.navigation.policy import NavigationPolicy, PolicyState
from nero.navigation.controller import VelocityCommand
from nero.robot import RobotInterface
from nero.observability import RosObservabilityPublisher

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map-based navigation agent")
    parser.add_argument("--map", required=True, help="Occupancy map (PNG or .npy)")
    parser.add_argument("--yaml", help="Path to map YAML metadata (for PNG maps)")
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Full map-frame goal pose",
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
    parser.add_argument(
        "--initial-pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Map-frame pose of the robot at startup; omit to localize "
        "automatically by matching the first depth scan against the map",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=GlobalLocalizationConfig.camera_height,
        help="Camera height above the floor (m), used by automatic localization",
    )
    parser.add_argument(
        "--localization-spin-speed",
        type=float,
        default=0.3,
        help="In-place spin speed (rad/s) while auto-localizing; 0 keeps the robot still",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--no-ros-observability",
        action="store_true",
        help="Disable normalized /nero ROS 2 telemetry topics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Create config
    auto_localize = args.initial_pose is None
    if auto_localize:
        logger.info(
            "No --initial-pose given; localizing against the fixed map at startup"
        )
    config = MapNavConfig(
        map_path=args.map,
        yaml_path=args.yaml,
        resolution=args.resolution,
        origin=tuple(args.origin),
        initial_pose=(0.0, 0.0, 0.0) if auto_localize else tuple(args.initial_pose),
        auto_localize=auto_localize,
        localization=GlobalLocalizationConfig(camera_height=args.camera_height),
        localization_spin_speed=args.localization_spin_speed,
        inflation_radius=args.inflation,
        max_linear_vel=args.max_vel,
    )

    robot = None
    policy = None
    try:
        robot = RobotInterface()
        policy = NavigationPolicy(
            robot=robot, map_config=config, enable_object_detection=False
        )
        policy.start()
        telemetry = RosObservabilityPublisher.try_create(
            enabled=not args.no_ros_observability
        )
        logger.info("Robot and shared IMU-RGBD navigation runtime initialized")
    except Exception as e:
        logger.error(f"Failed to connect to K1 robot: {e}")
        if policy is not None and policy.is_running:
            policy.stop()
        if robot is not None:
            robot.stop()
        sys.exit(1)

    try:
        # Set goal if provided
        if args.goal:
            policy.set_pose_goal(*args.goal)

        # Main loop
        logger.info(
            "Map navigation agent started. Press 'q' to quit, 'c' to click goal on map."
        )
        running = True
        click_mode = False
        loop_interval = 1.0 / 20.0

        while running:
            loop_started = time.monotonic()
            status = policy.step()
            if status.state == PolicyState.ERROR:
                logger.error("Navigation stopped: %s", status.message)
                break
            command = status.velocity_command or VelocityCommand()
            if telemetry is not None and policy.last_sensor is not None:
                sensor = policy.last_sensor
                telemetry.publish_robot_state(sensor.raw_state, robot)
                telemetry.publish_policy(status, sensor.timestamp)
                slam_pose = policy.slam.get_current_pose()
                if slam_pose is not None:
                    telemetry.publish_tracking(
                        slam_pose.tracking_status, slam_pose.num_map_points
                    )
                map_points = policy.slam.get_map_points()
                if len(map_points) and policy.map_alignment_ready:
                    telemetry.publish_point_cloud(
                        policy.transform_slam_points(map_points), sensor.timestamp
                    )

            # Print state
            pose = (
                status.current_pose.position_2d
                if status.current_pose is not None
                else config.initial_pose
            )
            logger.debug(
                f"State: {policy.state.value} | "
                f"Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) | "
                f"Vel: ({command.linear_x:.2f}, {command.linear_y:.2f}, "
                f"{command.angular_z:.2f})"
            )

            # Visualization
            if not args.headless:
                # Show camera view
                if policy.last_sensor is not None:
                    cv2.imshow("Camera", policy.last_sensor.rgb)

                # Show map view
                map_view = policy.render_map()
                cv2.imshow("Map", map_view)

                def set_clicked_goal(event, x, y, flags, param):
                    if click_mode and event == cv2.EVENT_LBUTTONUP:
                        goal_x, goal_y = policy.grid.pixel_to_world(x, y)
                        policy.set_pose_goal(goal_x, goal_y, pose[2])

                cv2.setMouseCallback("Map", set_clicked_goal)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                elif key == ord("r"):
                    policy.reset()
                    logger.info("Policy reset")
                elif key == ord("c"):
                    click_mode = not click_mode
                    logger.info(
                        f"Click mode: {'ON' if click_mode else 'OFF'} - click on map to set goal"
                    )
                elif key == ord("s") and policy.state == PolicyState.ARRIVED:
                    # Prompt for new goal
                    try:
                        x = float(input("Goal X (m): "))
                        y = float(input("Goal Y (m): "))
                        yaw = float(input("Goal yaw (rad): "))
                        policy.set_pose_goal(x, y, yaw)
                    except ValueError:
                        logger.warning("Invalid input")

            # Check for completion
            if policy.state == PolicyState.ARRIVED:
                logger.info("Arrived at goal!")
                if args.headless:
                    break

            elapsed = time.monotonic() - loop_started
            if elapsed < loop_interval:
                time.sleep(loop_interval - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        if policy is not None:
            policy.stop()
        elif robot is not None:
            robot.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
