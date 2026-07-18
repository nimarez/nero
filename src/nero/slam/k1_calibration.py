"""Probe a connected K1 and generate calibrated ORB-SLAM3 IMU_RGBD settings."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_INFO_TOPIC = "rt/booster/camera/color/camera_info"


def _yaml_real(value: float) -> str:
    text = f"{float(value):.12g}"
    return text if "." in text or "e" in text.lower() else text + ".0"


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _vector3(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([value[axis] for axis in ("x", "y", "z")], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 3:
            raise ValueError("3-vector must contain exactly three values")
        return np.asarray(value, dtype=float)
    return np.array([value.x, value.y, value.z], dtype=float)


def _transform(xyz: Any, rpy: Any) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", _vector3(rpy)).as_matrix()
    result[:3, 3] = _vector3(xyz)
    return result


def _sdk_transform(value: Any) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        [
            value.orientation.x,
            value.orientation.y,
            value.orientation.z,
            value.orientation.w,
        ]
    ).as_matrix()
    result[:3, 3] = _vector3(value.position)
    return result


def _device_body(value: Any) -> Any:
    """Unwrap the DeviceInfo JSON emitted by the official Python binding."""
    if isinstance(value, dict) and "body" in value:
        value = value["body"]
    if isinstance(value, str):
        value = json.loads(value)
    return value


@dataclass
class K1Calibration:
    camera_frame: str
    imu_frame: str
    width: int
    height: int
    camera_fps: float
    camera_matrix: list[float]
    distortion: list[float]
    depth_map_factor: float
    camera_rgb: bool
    tbc: list[list[float]]
    imu_frequency: float
    imu_noise_gyro: float
    imu_noise_acc: float
    imu_gyro_walk: float
    imu_acc_walk: float
    source: str = "K1 runtime + stationary IMU measurement"

    def validate(self) -> None:
        if len(self.camera_matrix) != 9:
            raise ValueError("camera_matrix must contain nine values")
        if np.asarray(self.tbc).shape != (4, 4):
            raise ValueError("tbc must be a 4x4 camera-to-IMU transform")
        positive = {
            "width": self.width,
            "height": self.height,
            "camera_fps": self.camera_fps,
            "depth_map_factor": self.depth_map_factor,
            "imu_frequency": self.imu_frequency,
            "imu_noise_gyro": self.imu_noise_gyro,
            "imu_noise_acc": self.imu_noise_acc,
            "imu_gyro_walk": self.imu_gyro_walk,
            "imu_acc_walk": self.imu_acc_walk,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0]
        if invalid:
            raise ValueError(f"invalid IMU_RGBD calibration fields: {', '.join(invalid)}")
        if not np.all(np.isfinite(np.asarray(self.tbc, dtype=float))):
            raise ValueError("tbc contains non-finite values")

    def save(self, path: Path) -> Path:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "K1Calibration":
        calibration = cls(**json.loads(path.read_text()))
        calibration.validate()
        return calibration

    def write_orbslam_settings(self, path: Path) -> Path:
        """Write an OpenCV YAML file accepted by ORB-SLAM3 IMU_RGBD."""
        self.validate()
        k = self.camera_matrix
        d = (self.distortion + [0.0] * 5)[:5]
        matrix = ",\n          ".join(
            ", ".join(_yaml_real(value) for value in row) for row in self.tbc
        )
        text = f"""%YAML:1.0
---
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {_yaml_real(k[0])}
Camera1.fy: {_yaml_real(k[4])}
Camera1.cx: {_yaml_real(k[2])}
Camera1.cy: {_yaml_real(k[5])}
Camera1.k1: {_yaml_real(d[0])}
Camera1.k2: {_yaml_real(d[1])}
Camera1.p1: {_yaml_real(d[2])}
Camera1.p2: {_yaml_real(d[3])}
Camera1.k3: {_yaml_real(d[4])}
Camera.width: {self.width}
Camera.height: {self.height}
Camera.fps: {_yaml_real(self.camera_fps)}
Camera.RGB: {1 if self.camera_rgb else 0}
Stereo.ThDepth: 40.0
Stereo.b: 0.07732
RGBD.DepthMapFactor: {_yaml_real(self.depth_map_factor)}
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{matrix}]
IMU.NoiseGyro: {_yaml_real(self.imu_noise_gyro)}
IMU.NoiseAcc: {_yaml_real(self.imu_noise_acc)}
IMU.GyroWalk: {_yaml_real(self.imu_gyro_walk)}
IMU.AccWalk: {_yaml_real(self.imu_acc_walk)}
IMU.Frequency: {_yaml_real(self.imu_frequency)}
IMU.InsertKFsWhenLost: 0
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
loopClosing: 1
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


def estimate_imu_noise(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Estimate noise terms from a stationary, uniformly sampled IMU sequence."""
    if len(samples) < 100:
        raise ValueError("at least 100 stationary IMU samples are required")
    timestamps = np.asarray([sample[0] for sample in samples])
    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("IMU timestamps are not strictly increasing")
    gyro = np.asarray([sample[1] for sample in samples])
    accel = np.asarray([sample[2] for sample in samples])
    frequency = 1.0 / dt
    noise_gyro = float(np.max(np.std(gyro, axis=0, ddof=1)) * np.sqrt(dt))
    noise_acc = float(np.max(np.std(accel, axis=0, ddof=1)) * np.sqrt(dt))

    block_size = max(10, int(round(frequency)))
    block_count = len(samples) // block_size
    if block_count < 2:
        raise ValueError("stationary capture is too short to estimate IMU bias walk")
    gyro_means = gyro[: block_count * block_size].reshape(block_count, block_size, 3).mean(1)
    acc_means = accel[: block_count * block_size].reshape(block_count, block_size, 3).mean(1)
    block_duration = block_size * dt
    gyro_walk = float(np.max(np.std(gyro_means, axis=0, ddof=1)) / np.sqrt(block_duration))
    acc_walk = float(np.max(np.std(acc_means, axis=0, ddof=1)) / np.sqrt(block_duration))
    epsilon = np.finfo(float).eps
    return {
        "imu_frequency": frequency,
        "imu_noise_gyro": max(noise_gyro, epsilon),
        "imu_noise_acc": max(noise_acc, epsilon),
        "imu_gyro_walk": max(gyro_walk, epsilon),
        "imu_acc_walk": max(acc_walk, epsilon),
    }


def estimate_frequency(timestamps: list[float], sensor: str) -> float:
    """Estimate a live sensor rate from monotonically increasing timestamps."""
    if len(timestamps) < 10:
        raise ValueError(f"at least 10 {sensor} timestamps are required")
    values = np.asarray(timestamps, dtype=float)
    intervals = np.diff(values)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) < 9:
        raise ValueError(f"{sensor} timestamps are not strictly increasing")
    period = float(np.median(intervals))
    if period <= 0 or not np.isfinite(period):
        raise ValueError(f"cannot estimate {sensor} frequency")
    return 1.0 / period


def probe_k1(
    *, iface: str, robot_name: str | None, duration: float, camera_info_topic: str
) -> K1Calibration:
    """Collect factory geometry and stationary noise data from a connected K1."""
    try:
        import booster_robotics_sdk_python as br
    except ImportError as exc:
        raise RuntimeError("the official Booster Python SDK is only available on Linux") from exc

    br.ChannelFactory.Instance().Init(0, iface)
    camera_client = br.CameraClient()
    loco_client = br.B1LocoClient()
    camera_client.InitWithName(robot_name) if robot_name else camera_client.Init()
    loco_client.InitWithName(robot_name) if robot_name else loco_client.Init()

    camera_catalog = _device_body(json.loads(camera_client.GetCameras().to_json_str()))
    sensor_catalog = _device_body(json.loads(loco_client.GetSensors().to_json_str()))
    cameras = (
        camera_catalog if isinstance(camera_catalog, list) else camera_catalog.get("cameras", [])
    )
    if not cameras:
        raise RuntimeError(f"K1 returned no cameras: {camera_catalog}")
    camera = next(
        (
            item
            for item in cameras
            if "head" in str(item.get("pos", item.get("position", ""))).lower()
        ),
        cameras[0],
    )
    imus = (
        sensor_catalog
        if isinstance(sensor_catalog, list)
        else sensor_catalog.get("imus", sensor_catalog.get("imu", []))
    )
    if isinstance(imus, dict):
        imus = [imus]
    if not imus:
        raise RuntimeError(f"K1 returned no IMU metadata: {sensor_catalog}")
    imu = imus[0]

    camera_info: dict[str, Any] = {}
    camera_timestamps: list[float] = []
    imu_samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    info_event = threading.Event()

    def on_camera_info(message: Any) -> None:
        # CameraInfo can be latched or use a different clock domain. Receipt
        # intervals measure the delivered rate that ORB-SLAM actually sees.
        timestamp = time.monotonic()
        if not camera_timestamps or timestamp > camera_timestamps[-1]:
            camera_timestamps.append(timestamp)
        camera_info.update(
            frame=str(message.header.frame_id),
            width=int(message.width),
            height=int(message.height),
            k=list(message.k),
            d=list(message.d),
        )
        info_event.set()

    def on_imu(message: Any) -> None:
        timestamp = _stamp_seconds(message.header.stamp)
        gyro = _vector3(message.angular_velocity)
        accel = _vector3(message.linear_acceleration)
        if not imu_samples or timestamp > imu_samples[-1][0]:
            imu_samples.append((timestamp, gyro, accel))

    camera_subscriber = br.CameraInfoSubscriber(on_camera_info, camera_info_topic)
    imu_subscriber = br.B1RosImuSubscriber(on_imu)
    camera_subscriber.InitChannel()
    imu_subscriber.InitChannel()
    try:
        if not info_event.wait(timeout=10.0):
            raise RuntimeError(f"no CameraInfo received on {camera_info_topic}")
        logger.info("Keep the K1 completely stationary for %.1f seconds", duration)
        time.sleep(duration)
    finally:
        camera_subscriber.CloseChannel()
        imu_subscriber.CloseChannel()

    noise = estimate_imu_noise(imu_samples)
    camera_fps = estimate_frequency(camera_timestamps, "camera")
    extrinsics = camera.get("extrinsics", {})
    if not all(key in extrinsics for key in ("xyz", "rpy", "parent_frame")):
        raise RuntimeError(f"camera catalog has no usable factory extrinsics: {camera}")
    parent = str(extrinsics["parent_frame"]).lower()
    imu_mount = str(imu.get("mount_position", "body")).lower()
    parent_to_camera = _transform(extrinsics["xyz"], extrinsics["rpy"])
    if ("body" in parent and "body" in imu_mount) or ("head" in parent and "head" in imu_mount):
        tbc = parent_to_camera
    elif "head" in parent and "body" in imu_mount:
        body_to_head = _sdk_transform(loco_client.GetFrameTransform(br.Frame.kBody, br.Frame.kHead))
        tbc = body_to_head @ parent_to_camera
    else:
        raise RuntimeError(f"cannot compose camera parent {parent!r} with IMU mount {imu_mount!r}")

    depth_scale = float(camera.get("depth_scale", 0.001))
    color = camera.get("color_encoding", {})
    color_format = str(color.get("format", "RGB8") if isinstance(color, dict) else color).upper()
    return K1Calibration(
        camera_frame=camera_info["frame"],
        imu_frame=str(imu.get("name", imu_mount)),
        width=camera_info["width"],
        height=camera_info["height"],
        camera_fps=camera_fps,
        camera_matrix=[float(value) for value in camera_info["k"]],
        distortion=[float(value) for value in camera_info["d"]],
        depth_map_factor=1.0 / depth_scale,
        camera_rgb="BGR" not in color_format,
        tbc=tbc.tolist(),
        **noise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe K1 calibration for ORB-SLAM3 IMU_RGBD")
    parser.add_argument("--iface", default="lo", help="DDS interface; use lo when running on K1")
    parser.add_argument("--robot-name", default=None)
    parser.add_argument(
        "--duration", type=float, default=60.0, help="stationary IMU capture seconds"
    )
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument("--output", type=Path, default=Path("config/k1_calibration.json"))
    parser.add_argument("--settings", type=Path, default=Path("config/k1_orbslam3_imu_rgbd.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    calibration = probe_k1(
        iface=args.iface,
        robot_name=args.robot_name,
        duration=args.duration,
        camera_info_topic=args.camera_info_topic,
    )
    calibration.save(args.output)
    calibration.write_orbslam_settings(args.settings)
    print(f"K1 calibration saved: {args.output}")
    print(f"ORB-SLAM3 IMU_RGBD settings saved: {args.settings}")


if __name__ == "__main__":
    main()
