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
    if not hasattr(booster, "B1LowStateSubscriber"):
        raise SystemExit("official Booster SDK has no B1LowStateSubscriber")
    if not hasattr(orbslam3.Sensor, "IMU_RGBD"):
        raise SystemExit("ORB binding has no IMU_RGBD sensor mode")

    calibration_path = Path(__file__).resolve().parents[1] / (
        "config/k1_geek_nominal_calibration.json"
    )
    calibration = K1Calibration.load(calibration_path)
    calibration.validate_geek_profile()

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
            rows, columns = np.indices((calibration.height, calibration.width))
            checkerboard = (((rows // 12) + (columns // 12)) % 2 * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.repeat(checkerboard[..., None], 3, axis=2))
            depth = np.full(
                (calibration.height, calibration.width), 1000, dtype=np.uint16
            )
            imu_dt = 1.0 / calibration.imu_frequency
            frame_timestamp = 1.0 / calibration.camera_fps
            imu = [
                (0.0, 0.0, 9.81, 0.0, 0.0, 0.0, timestamp)
                for timestamp in np.arange(imu_dt, frame_timestamp + imu_dt / 2, imu_dt)
            ]
            result = system.process_rgbd_inertial_enhanced(
                rgb, depth, frame_timestamp, imu
            )
            if not all(
                hasattr(result, field)
                for field in ("success", "is_valid", "state", "num_map_points")
            ):
                raise RuntimeError("ORB enhanced tracking result contract changed")
        finally:
            system.shutdown()

    print(
        "Native Booster + ORB-SLAM3 IMU_RGBD smoke test passed "
        f"at {calibration.width}x{calibration.height}@{calibration.camera_fps:g}fps "
        f"with {calibration.imu_frequency:g}Hz IMU"
    )


if __name__ == "__main__":
    main()
