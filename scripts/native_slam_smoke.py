"""Exercise the real Linux ORB-SLAM3 IMU_RGBD binding in a container."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from nero.slam.k1_calibration import K1Calibration


def main() -> None:
    import booster_robotics_sdk_python as booster
    import orbslam3

    vocabulary = Path(sys.argv[1])
    if not vocabulary.is_file():
        raise SystemExit(f"vocabulary is missing: {vocabulary}")
    if not hasattr(booster, "B1RosImuSubscriber"):
        raise SystemExit("official Booster SDK has no B1RosImuSubscriber")
    if not hasattr(orbslam3.Sensor, "IMU_RGBD"):
        raise SystemExit("ORB binding has no IMU_RGBD sensor mode")

    calibration = K1Calibration(
        camera_frame="camera",
        imu_frame="imu",
        width=320,
        height=240,
        camera_fps=30,
        camera_matrix=[216.5, 0, 160, 0, 216.5, 120, 0, 0, 1],
        distortion=[0, 0, 0, 0, 0],
        depth_map_factor=1000,
        camera_rgb=True,
        tbc=np.eye(4).tolist(),
        imu_frequency=200,
        imu_noise_gyro=0.001,
        imu_noise_acc=0.01,
        imu_gyro_walk=0.0001,
        imu_acc_walk=0.001,
        source="Docker native smoke fixture",
    )

    with tempfile.TemporaryDirectory() as directory:
        settings = Path(directory) / "imu_rgbd.yaml"
        calibration.write_orbslam_settings(settings)
        system = orbslam3.System(
            str(vocabulary), str(settings), orbslam3.Sensor.IMU_RGBD
        )
        if hasattr(system, "set_use_viewer"):
            system.set_use_viewer(False)
        system.initialize()
        try:
            rows, columns = np.indices((240, 320))
            checkerboard = (((rows // 12) + (columns // 12)) % 2 * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.repeat(checkerboard[..., None], 3, axis=2))
            depth = np.full((240, 320), 1000, dtype=np.uint16)
            imu = [
                (0.0, 0.0, 9.81, 0.0, 0.0, 0.0, timestamp)
                for timestamp in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
            ]
            result = system.process_rgbd_inertial_enhanced(rgb, depth, 0.03, imu)
            if not all(
                hasattr(result, field)
                for field in ("success", "is_valid", "state", "num_map_points")
            ):
                raise RuntimeError("ORB enhanced tracking result contract changed")
        finally:
            system.shutdown()

    print("Native Booster + ORB-SLAM3 IMU_RGBD smoke test passed")


if __name__ == "__main__":
    main()
