"""Measure native IMU_RGBD SLAM against Booster Studio references."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nero.evaluation.sim_reference import (
    align_se2,
    depth_to_world_points,
    localization_metrics,
    map_metrics,
)
from nero.simulation.booster_studio import (
    BoosterStudioRobotInterface,
    write_booster_studio_calibration,
)
from nero.slam.k1_calibration import K1Calibration
from nero.slam.orb_slam3_node import ORBSLAM3Node
from nero.observability import RosObservabilityPublisher

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native ORB-SLAM3 IMU_RGBD against Booster Studio"
    )
    parser.add_argument("--robot-name", default=None)
    parser.add_argument(
        "--sensor-calibration",
        type=Path,
        default=Path("config/k1_calibration.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/sim_benchmark"))
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--linear-speed", type=float, default=0.12)
    parser.add_argument("--yaw-speed", type=float, default=0.25)
    parser.add_argument("--map-frame-stride", type=int, default=5)
    parser.add_argument("--depth-pixel-stride", type=int, default=8)
    parser.add_argument("--stationary", action="store_true")
    parser.add_argument(
        "--no-ros-observability",
        action="store_true",
        help="Disable normalized /nero ROS 2 telemetry topics",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _body_reference(pose_2d: np.ndarray, imu_rpy: np.ndarray) -> np.ndarray:
    x, y, yaw = pose_2d
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "xyz", [imu_rpy[0], imu_rpy[1], yaw]
    ).as_matrix()
    pose[:2, 3] = [x, y]
    return pose


def _motion_segments(args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    if args.stationary:
        return [(args.segment_seconds, 0.0, 0.0, 0.0)]
    duration = args.segment_seconds
    return [
        (duration, args.linear_speed, 0.0, 0.0),
        (duration, 0.0, 0.0, args.yaw_speed),
        (duration, args.linear_speed, 0.0, 0.0),
        (duration, 0.0, 0.0, args.yaw_speed),
    ]


def _run_trajectory(
    robot: BoosterStudioRobotInterface,
    slam: ORBSLAM3Node,
    calibration: K1Calibration,
    args: argparse.Namespace,
    telemetry: RosObservabilityPublisher | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, int]:
    estimated_body: list[np.ndarray] = []
    reference_body: list[np.ndarray] = []
    estimated_points: list[np.ndarray] = []
    reference_points: list[np.ndarray] = []
    total_frames = 0
    last_timestamp: float | None = None
    schedule = [(args.warmup_seconds, 0.0, 0.0, 0.0), *_motion_segments(args)]

    for duration, vx, vy, vyaw in schedule:
        robot.set_velocity(vx, vy, vyaw)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            state = robot.get_state(include_images=True)
            timestamp = robot.image_timestamp(state.rgb)
            if timestamp == last_timestamp:
                time.sleep(0.002)
                continue
            last_timestamp = timestamp
            total_frames += 1
            if telemetry is not None:
                telemetry.publish_robot_state(state, robot)
            depth = robot.image_to_array(state.depth)
            camera_pose = slam.track_frame(
                robot.image_to_array(state.rgb),
                depth,
                imu_data=state.imu_samples,
                timestamp=timestamp,
            )
            if telemetry is not None:
                telemetry.publish_tracking(
                    camera_pose.tracking_status, camera_pose.num_map_points
                )
            if camera_pose.tracking_status != "OK":
                continue

            body_estimate = slam.body_pose(camera_pose).to_matrix()
            reference_timestamp = robot.image_source_timestamp(state.rgb)
            body_reference = _body_reference(
                robot.get_ground_truth_pose(reference_timestamp),
                state.orientation_rpy,
            )
            estimated_body.append(body_estimate)
            reference_body.append(body_reference)
            if telemetry is not None:
                telemetry.publish_pose(body_estimate, timestamp)
                telemetry.publish_pose(body_reference, timestamp, reference=True)

            if len(estimated_body) % args.map_frame_stride == 0:
                reference_camera = body_reference @ slam.tbc
                intrinsic = np.asarray(state.camera_info.k, dtype=float).reshape(3, 3)
                common = {
                    "camera_matrix": intrinsic,
                    "depth_map_factor": calibration.depth_map_factor,
                    "depth_min_m": calibration.depth_min_m or 0.5,
                    "depth_max_m": calibration.depth_max_m or 6.0,
                    "stride": args.depth_pixel_stride,
                }
                estimated_cloud = depth_to_world_points(
                    depth, camera_pose.to_matrix(), **common
                )
                reference_cloud = depth_to_world_points(
                    depth, reference_camera, **common
                )
                estimated_points.append(estimated_cloud)
                reference_points.append(reference_cloud)
                if telemetry is not None:
                    telemetry.publish_point_cloud(estimated_cloud, timestamp)
                    telemetry.publish_point_cloud(
                        reference_cloud, timestamp, reference=True
                    )
    robot.stop()
    return (
        estimated_body,
        reference_body,
        np.concatenate(estimated_points) if estimated_points else np.empty((0, 3)),
        np.concatenate(reference_points) if reference_points else np.empty((0, 3)),
        total_frames,
    )


def run(args: argparse.Namespace) -> Path:
    if args.warmup_seconds <= 0 or args.segment_seconds <= 0:
        raise ValueError("benchmark durations must be positive")
    if not 0 <= args.linear_speed <= 0.2 or abs(args.yaw_speed) > 0.5:
        raise ValueError("benchmark speeds exceed the conservative K1 test envelope")
    if args.map_frame_stride <= 0 or args.depth_pixel_stride <= 0:
        raise ValueError("benchmark strides must be positive")
    source = args.sensor_calibration
    if not source.is_file():
        source = Path("config/k1_geek_nominal_calibration.json")
        logger.warning("Using nominal K1 Geek calibration: %s", source)
    expected = K1Calibration.load(source)
    expected.validate_geek_profile()
    robot = BoosterStudioRobotInterface(
        robot_name=args.robot_name,
        expected_calibration=expected,
    )
    telemetry = RosObservabilityPublisher.try_create(
        enabled=not args.no_ros_observability
    )
    slam: ORBSLAM3Node | None = None
    try:
        robot.initialize()
        camera_fps, imu_frequency = robot.measure_sensor_rates()
        robot.validate_sensor_profile(camera_fps, imu_frequency)
        calibration_path = Path("config/booster_studio_k1_calibration.json")
        settings_path = Path("config/booster_studio_k1_imu_rgbd.yaml")
        calibration = write_booster_studio_calibration(
            robot.get_camera_info(),
            calibration_path,
            camera_fps=camera_fps,
            imu_frequency=imu_frequency,
            reference_calibration=expected,
        )
        calibration.write_orbslam_settings(settings_path)
        slam = ORBSLAM3Node(
            calibration_path=str(calibration_path),
            settings_path=str(settings_path),
            allow_fallback=False,
        )
        slam.initialize(camera_info=robot.get_camera_info())
        estimated, reference, estimated_map, reference_map, total = _run_trajectory(
            robot, slam, calibration, args, telemetry
        )
        if len(estimated) < 2:
            raise RuntimeError(
                f"ORB-SLAM3 produced only {len(estimated)} valid poses from {total} frames"
            )
        _, alignment = align_se2(estimated, reference)
        result: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": slam.backend_name,
            "sensor_profile": {
                "width": calibration.width,
                "height": calibration.height,
                "camera_fps": camera_fps,
                "imu_frequency": imu_frequency,
                "depth_min_m": calibration.depth_min_m,
                "depth_max_m": calibration.depth_max_m,
            },
            "localization": localization_metrics(
                estimated, reference, total_frames=total
            ),
        }
        if len(estimated_map) and len(reference_map):
            result["map"] = map_metrics(
                estimated_map, reference_map, alignment=alignment
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = args.output_dir / f"benchmark-{stem}.json"
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(
            args.output_dir / f"trajectory-{stem}.npz",
            estimated=np.asarray(estimated),
            reference=np.asarray(reference),
            alignment=alignment,
        )
        logger.info("Benchmark report: %s", report_path)
        print(json.dumps(result, indent=2))
        return report_path
    finally:
        try:
            try:
                robot.stop()
            except Exception as exc:
                logger.warning("Could not send benchmark stop command: %s", exc)
        finally:
            if slam is not None:
                slam.shutdown()
            if telemetry is not None:
                telemetry.close()
            robot.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
