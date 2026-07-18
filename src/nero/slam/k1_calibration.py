"""Probe a connected K1 and generate calibrated ORB-SLAM3 IMU_RGBD settings."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_INFO_TOPIC = "/boostercamera/head/rgb/camera_info"
DEFAULT_RGB_TOPIC = "/boostercamera/head/rgb"
DEFAULT_DEPTH_TOPIC = "/boostercamera/head/depth"
K1_GEEK_RESOLUTION = (544, 448)
K1_GEEK_FPS = 20.0
K1_GEEK_FOV_DEGREES = (105.0, 94.0)
K1_GEEK_DEPTH_RANGE_M = (0.5, 6.0)


def _yaml_real(value: float) -> str:
    text = f"{float(value):.12g}"
    return text if "." in text or "e" in text.lower() else text + ".0"


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
    shutter_type: str = "global"
    rgb_fov_degrees: list[float] | None = None
    depth_width: int | None = None
    depth_height: int | None = None
    depth_fps: float | None = None
    depth_fov_degrees: list[float] | None = None
    depth_accuracy_at_1m: float | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None

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
        matrix = np.asarray(self.camera_matrix, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera_matrix must contain finite positive focal lengths")
        rotation = np.asarray(self.tbc, dtype=float)[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError("tbc rotation must be a proper orthonormal matrix")
        if self.shutter_type.lower() != "global":
            raise ValueError("K1 Geek calibration must use its global shutter")
        if self.depth_width is not None and self.depth_width <= 0:
            raise ValueError("depth_width must be positive")
        if self.depth_height is not None and self.depth_height <= 0:
            raise ValueError("depth_height must be positive")
        if self.depth_fps is not None and self.depth_fps <= 0:
            raise ValueError("depth_fps must be positive")
        if self.depth_width is not None and self.depth_width != self.width:
            raise ValueError("K1 Geek RGB and depth widths must match")
        if self.depth_height is not None and self.depth_height != self.height:
            raise ValueError("K1 Geek RGB and depth heights must match")
        if self.depth_fps is not None and not np.isclose(self.depth_fps, self.camera_fps):
            raise ValueError("K1 Geek RGB and depth frame rates must match")
        for name, fov in (
            ("rgb_fov_degrees", self.rgb_fov_degrees),
            ("depth_fov_degrees", self.depth_fov_degrees),
        ):
            if fov is not None and (
                len(fov) != 2 or any(not 0 < float(value) < 180 for value in fov)
            ):
                raise ValueError(f"{name} must contain horizontal and vertical angles")
        if self.depth_accuracy_at_1m is not None and not 0 < self.depth_accuracy_at_1m < 1:
            raise ValueError("depth_accuracy_at_1m must be a fractional error")
        if self.depth_min_m is not None and self.depth_max_m is not None:
            if not 0 < self.depth_min_m < self.depth_max_m:
                raise ValueError("invalid K1 depth operating range")

    def validate_geek_profile(self) -> None:
        """Require the delivered sensor contract published for the K1 Geek."""
        self.validate()
        if (self.width, self.height) != K1_GEEK_RESOLUTION:
            raise ValueError(
                f"K1 Geek RGB must be {K1_GEEK_RESOLUTION[0]}x{K1_GEEK_RESOLUTION[1]}"
            )
        if not np.isclose(self.camera_fps, K1_GEEK_FPS, rtol=0.05):
            raise ValueError(f"K1 Geek RGB must deliver {K1_GEEK_FPS:g} fps")
        if self.depth_width != self.width or self.depth_height != self.height:
            raise ValueError("K1 Geek depth resolution metadata is missing or mismatched")
        if self.depth_fps is None or not np.isclose(
            self.depth_fps, K1_GEEK_FPS, rtol=0.05
        ):
            raise ValueError("K1 Geek depth must deliver 20 fps")
        if self.rgb_fov_degrees != list(K1_GEEK_FOV_DEGREES):
            raise ValueError("K1 Geek RGB FOV must be 105x94 degrees")
        if self.depth_fov_degrees != list(K1_GEEK_FOV_DEGREES):
            raise ValueError("K1 Geek depth FOV must be 105x94 degrees")
        if (self.depth_min_m, self.depth_max_m) != K1_GEEK_DEPTH_RANGE_M:
            raise ValueError("K1 Geek depth range must be 0.5-6 m")

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
Camera.fps: {int(round(self.camera_fps))}
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
    # Some K1 camera pipelines alternate 33 ms and 67 ms delivery intervals.
    # Count over the complete span measures their real average rate; a median
    # would incorrectly report 30 Hz for that 20 Hz stream.
    period = float((values[-1] - values[0]) / (len(values) - 1))
    if period <= 0 or not np.isfinite(period):
        raise ValueError(f"cannot estimate {sensor} frequency")
    return 1.0 / period


def probe_k1_ros(
    *,
    duration: float,
    iface: str = "lo",
    camera_info_topic: str = DEFAULT_CAMERA_INFO_TOPIC,
    rgb_topic: str = DEFAULT_RGB_TOPIC,
    depth_topic: str = DEFAULT_DEPTH_TOPIC,
) -> K1Calibration:
    """Calibrate from the ROS streams shipped on production K1 firmware.

    Production firmware 1.5 exposes calibrated intrinsics and synchronized
    RGB-D frames via ROS. The mount transform comes from Booster's nominal K1
    Geek model; the output source field records that distinction explicitly.
    """
    try:
        import rclpy
        import booster_robotics_sdk_python as booster
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError("K1 ROS calibration requires ROS 2 sensor messages") from exc

    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=None)
    booster.ChannelFactory.Instance().Init(0, iface)
    node = rclpy.create_node("nero_k1_calibration")
    qos = rclpy.qos.QoSProfile(
        depth=200,
        reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    )
    camera_info: dict[str, Any] = {}
    rgb_timestamps: list[float] = []
    depth_timestamps: list[float] = []
    imu_samples: list[tuple[float, np.ndarray, np.ndarray]] = []

    def on_info(message: Any) -> None:
        camera_info.update(
            frame=str(message.header.frame_id), width=int(message.width),
            height=int(message.height), k=list(message.k), d=list(message.d),
        )

    def on_image(values: list[float]):
        def callback(message: Any) -> None:
            timestamp = _stamp_seconds(message.header.stamp)
            if not values or timestamp > values[-1]:
                values.append(timestamp)
        return callback

    def on_low_state(message: Any) -> None:
        timestamp = time.time()
        message = message.imu_state
        if not imu_samples or timestamp > imu_samples[-1][0]:
            imu_samples.append(
                (
                    timestamp,
                    np.asarray(message.gyro, dtype=float),
                    np.asarray(message.acc, dtype=float),
                )
            )

    subscriptions = [
        node.create_subscription(CameraInfo, camera_info_topic, on_info, qos),
        node.create_subscription(Image, rgb_topic, on_image(rgb_timestamps), qos),
        node.create_subscription(Image, depth_topic, on_image(depth_timestamps), qos),
    ]
    low_state_subscriber = booster.B1LowStateSubscriber(on_low_state)
    low_state_subscriber.InitChannel()
    try:
        logger.info("Keep the K1 completely stationary for %.1f seconds", duration)
        deadline = time.monotonic() + max(10.0, duration + 10.0)
        capture_started: float | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if camera_info and rgb_timestamps and depth_timestamps and imu_samples:
                if capture_started is None:
                    capture_started = time.monotonic()
                if time.monotonic() - capture_started >= duration:
                    break
        missing = [
            name for name, values in (
                (camera_info_topic, camera_info), (rgb_topic, rgb_timestamps),
                (depth_topic, depth_timestamps), ("K1 low-state IMU", imu_samples),
            ) if not values
        ]
        if missing:
            raise RuntimeError("no live K1 samples received from: " + ", ".join(missing))
    finally:
        low_state_subscriber.CloseChannel()
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()

    noise = estimate_imu_noise(imu_samples)
    rgb_fps = estimate_frequency(rgb_timestamps, "RGB camera")
    depth_fps = estimate_frequency(depth_timestamps, "depth camera")
    nominal_path = Path(__file__).resolve().parents[3] / "config/k1_geek_nominal_calibration.json"
    nominal = K1Calibration.load(nominal_path)
    return K1Calibration(
        camera_frame=camera_info["frame"],
        imu_frame="body_imu",
        width=camera_info["width"], height=camera_info["height"],
        camera_fps=rgb_fps,
        camera_matrix=[float(value) for value in camera_info["k"]],
        distortion=[float(value) for value in camera_info["d"]],
        depth_map_factor=1000.0,
        camera_rgb=False,
        tbc=nominal.tbc,
        source=(
            "K1 production ROS intrinsics/rates + stationary IMU measurement; "
            "nominal K1 Geek camera-to-body mount transform"
        ),
        shutter_type="global", rgb_fov_degrees=list(K1_GEEK_FOV_DEGREES),
        depth_width=camera_info["width"], depth_height=camera_info["height"],
        depth_fps=depth_fps, depth_fov_degrees=list(K1_GEEK_FOV_DEGREES),
        depth_accuracy_at_1m=0.03,
        depth_min_m=K1_GEEK_DEPTH_RANGE_M[0], depth_max_m=K1_GEEK_DEPTH_RANGE_M[1],
        **noise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe K1 calibration for ORB-SLAM3 IMU_RGBD")
    parser.add_argument("--iface", default="lo", help="DDS interface; use lo when running on K1")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="stationary IMU capture seconds"
    )
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument("--output", type=Path, default=Path("config/k1_calibration.json"))
    parser.add_argument("--settings", type=Path, default=Path("config/k1_orbslam3_imu_rgbd.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    calibration = probe_k1_ros(
        duration=args.duration,
        iface=args.iface,
        camera_info_topic=args.camera_info_topic,
    )
    calibration.validate_geek_profile()
    calibration.save(args.output)
    calibration.write_orbslam_settings(args.settings)
    print(f"K1 calibration saved: {args.output}")
    print(f"ORB-SLAM3 IMU_RGBD settings saved: {args.settings}")


if __name__ == "__main__":
    main()
