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
    parser.add_argument("--detections-topic", default=BoosterStudioTopics.detections)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
        detections=args.detections_topic,
    )
    robot = BoosterStudioRobotInterface(topics=topics, robot_name=args.robot_name)
    calibration_path = Path("config/booster_studio_k1_calibration.json")
    settings_path = Path("config/booster_studio_k1_imu_rgbd.yaml")
    try:
        robot.initialize()
        calibration = write_booster_studio_calibration(
            robot.get_camera_info(), calibration_path
        )
        calibration.write_orbslam_settings(settings_path)
        logger.info("Booster Studio RGB-D, CameraInfo, IMU, and pose streams are live")
        run_agent(
            robot,
            args,
            slam_options={
                "calibration_path": str(calibration_path),
                "settings_path": str(settings_path),
                "start_imu_source": False,
            },
            object_detector=BoosterStudioObjectDetector(robot),
        )
    finally:
        robot.close()


if __name__ == "__main__":
    main()
