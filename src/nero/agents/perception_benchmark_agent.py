"""Non-moving K1 benchmark for the detector under live IMU-RGBD SLAM load."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import numpy as np

from nero.perception.object_detector import (
    ObjectDetector,
    configure_qualcomm_cpu_partition,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark open-vocabulary detection without commanding motion"
    )
    parser.add_argument("--target", default="green can")
    parser.add_argument("--results", type=int, default=3)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the K1 RGB-D stream instead of a zero-valued test frame",
    )
    parser.add_argument(
        "--with-slam",
        action="store_true",
        help="Run native IMU_RGBD SLAM concurrently (requires --live)",
    )
    return parser.parse_args()


def _wait_for_state(robot, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return robot.get_state(include_images=True)
        except RuntimeError:
            time.sleep(0.1)
    raise RuntimeError("K1 sensor snapshot did not become ready")


def main() -> None:
    args = parse_args()
    if args.results < 1:
        raise SystemExit("--results must be positive")
    if args.with_slam and not args.live:
        raise SystemExit("--with-slam requires --live")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    robot = None
    slam = None
    configure_qualcomm_cpu_partition()
    if args.live:
        from nero.robot import RobotInterface

        robot = RobotInterface()
        state = _wait_for_state(robot)
        if args.with_slam:
            from nero.slam.orb_slam3_node import ORBSLAM3Node

            slam = ORBSLAM3Node()
            slam.initialize(state.camera_info)

    detector = ObjectDetector()
    if not detector.initialize():
        raise SystemExit("detector initialization failed")
    detector.set_target(args.target)
    revision = detector.result_revision or 0
    submitted = None
    elapsed_results: list[float] = []
    slam_results: list[float] = []

    try:
        while len(elapsed_results) < args.results:
            if robot is None:
                rgb = np.zeros((448, 544, 3), dtype=np.uint8)
                depth = np.full((448, 544), 1000, dtype=np.uint16)
                camera_info = None
                time.sleep(0.01)
            else:
                state = robot.get_state(include_images=True)
                rgb = robot.image_to_array(state.rgb)
                depth = robot.image_to_array(state.depth)
                camera_info = state.camera_info
                if slam is not None:
                    slam_started = time.perf_counter()
                    slam.track_frame(
                        rgb,
                        depth,
                        imu_data=state.imu_samples,
                        timestamp=robot.image_timestamp(state.rgb),
                    )
                    slam_results.append(time.perf_counter() - slam_started)
            if submitted is None:
                submitted = time.perf_counter()
            detections = detector.detect(rgb, depth, camera_info)
            current_revision = detector.result_revision or 0
            if current_revision == revision:
                continue
            elapsed = time.perf_counter() - submitted
            elapsed_results.append(elapsed)
            revision = current_revision
            submitted = time.perf_counter()
            print(
                f"result {revision}: {elapsed:.3f}s, "
                f"detections={len(detections)}, backend={detector.backend}",
                flush=True,
            )
    finally:
        detector.close()
        if slam is not None:
            slam.shutdown()
        if robot is not None:
            robot.close()

    median = statistics.median(elapsed_results)
    print(
        f"median: {median:.3f}s ({1.0 / median:.2f} FPS), "
        f"samples={len(elapsed_results)}",
        flush=True,
    )
    if slam_results:
        slam_median = statistics.median(slam_results)
        print(
            f"SLAM median: {slam_median:.3f}s ({1.0 / slam_median:.2f} FPS), "
            f"samples={len(slam_results)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
