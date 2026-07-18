"""Run Nero against Booster Studio's virtual K1."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from nero.agents.orb_slam_agent import run_agent
from nero.simulation.booster_studio import (
    BoosterStudioRobotInterface,
    BoosterStudioObjectDetector,
    BoosterStudioTopics,
    write_booster_studio_calibration,
)
from nero.slam.k1_calibration import K1Calibration

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nero object-following test on a Booster Studio virtual K1"
    )
    parser.add_argument("--robot-name", default=None)
    parser.add_argument("--rgb-topic", default=BoosterStudioTopics.rgb)
    parser.add_argument("--depth-topic", default=BoosterStudioTopics.depth)
    parser.add_argument("--camera-info-topic", default=BoosterStudioTopics.camera_info)
    parser.add_argument("--imu-topic", default=BoosterStudioTopics.imu)
    parser.add_argument("--pose-topic", default=BoosterStudioTopics.pose)
    parser.add_argument("--clock-topic", default=BoosterStudioTopics.clock)
    parser.add_argument("--odom-topic", default=BoosterStudioTopics.odom)
    parser.add_argument("--joints-topic", default=BoosterStudioTopics.joints)
    parser.add_argument("--detections-topic", default=BoosterStudioTopics.detections)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-ros-observability", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--sensor-calibration",
        type=Path,
        default=Path("config/k1_calibration.json"),
        help="Calibration captured on the real K1 Geek",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ["BOOSTER_NET_IF"] = "127.0.0.1"
    topics = BoosterStudioTopics(
        rgb=args.rgb_topic,
        depth=args.depth_topic,
        camera_info=args.camera_info_topic,
        imu=args.imu_topic,
        pose=args.pose_topic,
        clock=args.clock_topic,
        odom=args.odom_topic,
        joints=args.joints_topic,
        detections=args.detections_topic,
    )
    calibration_source = args.sensor_calibration
    if not calibration_source.is_file():
        calibration_source = Path("config/k1_geek_nominal_calibration.json")
        logger.warning(
            "Real K1 calibration is absent; using the nominal Geek profile at %s",
            calibration_source,
        )
    expected_calibration = K1Calibration.load(calibration_source)
    expected_calibration.validate_geek_profile()
    robot = BoosterStudioRobotInterface(
        topics=topics,
        robot_name=args.robot_name,
        expected_calibration=expected_calibration,
    )
    calibration_path = Path("config/booster_studio_k1_calibration.json")
    settings_path = Path("config/booster_studio_k1_imu_rgbd.yaml")
    try:
        robot.initialize()
        camera_fps, imu_frequency = robot.measure_sensor_rates()
        robot.validate_sensor_profile(camera_fps, imu_frequency)
        calibration = write_booster_studio_calibration(
            robot.get_camera_info(),
            calibration_path,
            camera_fps=camera_fps,
            imu_frequency=imu_frequency,
            reference_calibration=expected_calibration,
        )
        calibration.write_orbslam_settings(settings_path)
        logger.info(
            "Booster Studio RGB-D %.1f Hz and IMU %.1f Hz streams are live",
            camera_fps,
            imu_frequency,
        )
        run_agent(
            robot,
            args,
            slam_options={
                "calibration_path": str(calibration_path),
                "settings_path": str(settings_path),
            },
            object_detector=BoosterStudioObjectDetector(robot),
        )
    finally:
        robot.close()


if __name__ == "__main__":
    main()
