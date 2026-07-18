"""ORB-SLAM3 IMU_RGBD wrapper with a development-only RGB-D VO fallback."""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from nero.slam.imu_buffer import IMUMeasurement, K1IMUSource
from nero.slam.k1_calibration import K1Calibration

logger = logging.getLogger(__name__)


@dataclass
class SLAMPose:
    position: np.ndarray
    orientation: np.ndarray
    timestamp: float = 0.0
    tracking_status: str = "OK"
    num_map_points: int = 0

    @property
    def position_2d(self) -> np.ndarray:
        return self.position[:2]

    @property
    def yaw(self) -> float:
        x, y, z, w = self.orientation
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def to_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        matrix[:3, 3] = self.position
        return matrix


@dataclass
class SLAMConfig:
    fx: float = 216.5
    fy: float = 216.5
    cx: float = 160.0
    cy: float = 120.0
    n_features: int = 1000
    scale_factor: float = 1.2
    n_levels: int = 8
    ini_th_fast: int = 20
    min_th_fast: int = 7
    depth_threshold: float = 5.0
    keyframe_distance: float = 0.1
    keyframe_angle: float = 0.1
    enable_loop_closure: bool = True
    loop_closure_score_threshold: float = 0.7
    use_imu: bool = True
    imu_noise_sigma: float = 1e-2
    imu_accel_noise_sigma: float = 1e-2
    imu_gyro_bias_rw: float = 1e-5
    imu_accel_bias_rw: float = 1e-5
    enable_atlas: bool = True
    max_map_points: int = 100000


def _config_from(value: SLAMConfig | dict[str, Any] | None) -> SLAMConfig:
    if isinstance(value, SLAMConfig):
        return value
    allowed = {field.name for field in fields(SLAMConfig)}
    return SLAMConfig(
        **{key: item for key, item in (value or {}).items() if key in allowed}
    )


class ORBSLAM3Node:
    """Run native ORB-SLAM3 in IMU_RGBD mode whenever its Linux wheel is present."""

    def __init__(
        self,
        config: SLAMConfig | dict[str, Any] | None = None,
        vocab_path: str | None = None,
        settings_path: str | None = None,
        calibration_path: str | None = None,
        allow_fallback: bool | None = None,
        start_imu_source: bool = True,
        **aliases: Any,
    ):
        config_dict = config if isinstance(config, dict) else {}
        self.config = _config_from(config)
        self.vocab_path = Path(
            vocab_path
            or aliases.pop("voc_path", None)
            or config_dict.get("voc_path", "config/ORBvoc.txt")
        )
        self.settings_path = Path(
            settings_path
            or config_dict.get("settings_path", "config/k1_orbslam3_imu_rgbd.yaml")
        )
        self.calibration_path = Path(
            calibration_path
            or config_dict.get("calibration_path", "config/k1_calibration.json")
        )
        if aliases:
            raise TypeError(f"unexpected ORBSLAM3Node arguments: {', '.join(aliases)}")
        self.allow_fallback = (
            sys.platform != "linux" if allow_fallback is None else allow_fallback
        )
        self.start_imu_source = start_imu_source
        self._lock = threading.Lock()
        self._slam_system: Any = None
        self._imu_source: K1IMUSource | None = None
        self._calibration: K1Calibration | None = None
        self._use_fallback = True
        self._is_initialized = False
        self._current_pose: SLAMPose | None = None
        self._frame_count = 0
        self._keyframe_count = 0
        self._map_points_count = 0
        self._last_frame_timestamp: float | None = None
        self._prev_keypoints: Any = None
        self._prev_descriptors: Any = None
        self._prev_depth: np.ndarray | None = None

    @property
    def backend_name(self) -> str:
        return "rgbd-fallback" if self._use_fallback else "orbslam3-imu-rgbd"

    @property
    def using_native(self) -> bool:
        return not self._use_fallback

    def initialize(self, camera_info: Any = None) -> bool:
        if camera_info is not None:
            k = np.asarray(camera_info.k).reshape(3, 3)
            self.config.fx, self.config.fy = float(k[0, 0]), float(k[1, 1])
            self.config.cx, self.config.cy = float(k[0, 2]), float(k[1, 2])
        try:
            self._initialize_native()
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"ORB-SLAM3 IMU_RGBD initialization failed: {exc}"
                ) from exc
            logger.warning(
                "Native ORB-SLAM3 IMU_RGBD unavailable; using RGB-D VO: %s", exc
            )
            self._use_fallback = True
        self._is_initialized = True
        return True

    def _initialize_native(self) -> None:
        import orbslam3

        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"ORB vocabulary missing: {self.vocab_path}; "
                "run `uv run nero-setup-orbslam`"
            )
        self._calibration = K1Calibration.load(self.calibration_path)
        self._calibration.write_orbslam_settings(self.settings_path)
        system = orbslam3.System(
            str(self.vocab_path), str(self.settings_path), orbslam3.Sensor.IMU_RGBD
        )
        if hasattr(system, "set_use_viewer"):
            system.set_use_viewer(False)
        system.initialize()
        self._slam_system = system
        self._use_fallback = False
        if self.start_imu_source:
            self._imu_source = K1IMUSource()
            try:
                self._imu_source.start()
            except Exception as exc:
                self._slam_system.shutdown()
                self._slam_system = None
                raise RuntimeError("K1 IMU subscription failed") from exc
        logger.info("ORB-SLAM3 initialized with Sensor.IMU_RGBD")

    @staticmethod
    def _coerce_imu(data: Any) -> list[tuple[float, ...]]:
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]
        result = []
        for sample in data:
            if isinstance(sample, IMUMeasurement):
                result.append(sample.as_orbslam_tuple())
            elif isinstance(sample, dict):
                accel, gyro = sample["accel"], sample["gyro"]
                result.append((*accel, *gyro, float(sample["timestamp"])))
            else:
                values = tuple(float(value) for value in sample)
                if len(values) != 7:
                    raise ValueError(
                        "each IMU sample must contain ax ay az gx gy gz timestamp"
                    )
                result.append(values)
        return result

    def track_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        imu_data: Any = None,
        timestamp: float | None = None,
    ) -> SLAMPose:
        if not self._is_initialized:
            raise RuntimeError("initialize() must be called before track_frame()")
        ts = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            if (
                self._last_frame_timestamp is not None
                and ts <= self._last_frame_timestamp
            ):
                return self._current_pose or self._lost_pose(ts)
            self._frame_count += 1
            if self._use_fallback:
                pose = self._track_fallback(rgb, depth, ts)
            else:
                if imu_data is None and self._imu_source is not None:
                    imu_data = self._imu_source.buffer.between(
                        self._last_frame_timestamp, ts
                    )
                pose = self._track_native(rgb, depth, self._coerce_imu(imu_data), ts)
            self._last_frame_timestamp = ts
            self._current_pose = pose
            return pose

    def _lost_pose(self, timestamp: float) -> SLAMPose:
        return SLAMPose(
            position=(
                self._current_pose.position.copy()
                if self._current_pose
                else np.zeros(3)
            ),
            orientation=(
                self._current_pose.orientation.copy()
                if self._current_pose
                else np.array([0.0, 0.0, 0.0, 1.0])
            ),
            timestamp=timestamp,
            tracking_status="LOST",
            num_map_points=self._map_points_count,
        )

    def _track_native(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        imu: list[tuple[float, ...]],
        timestamp: float,
    ) -> SLAMPose:
        if not imu:
            logger.warning(
                "Skipping IMU_RGBD frame %.6f: no synchronized IMU samples", timestamp
            )
            return self._lost_pose(timestamp)
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_image = np.ascontiguousarray(depth)
        if np.issubdtype(depth_image.dtype, np.floating):
            factor = self._calibration.depth_map_factor if self._calibration else 1000.0
            depth_image = (
                np.nan_to_num(depth_image * factor).clip(0, 65535).astype(np.uint16)
            )
        process = getattr(self._slam_system, "process_rgbd_inertial_enhanced", None)
        if process is None:
            process = self._slam_system.process_image_rgbd_inertial
        result = process(image, depth_image, timestamp, imu)
        if result is not None and hasattr(result, "success"):
            if not bool(result.success) or not bool(result.is_valid):
                logger.warning("ORB-SLAM3 rejected IMU_RGBD frame %.6f", timestamp)
                return self._lost_pose(timestamp)
            state_value = result.state
            self._map_points_count = int(result.num_map_points)
        else:
            state_value = self._slam_system.get_tracking_state()
            points = self._slam_system.get_tracked_mappoints()
            self._map_points_count = len(points) if points is not None else 0
        tcw = np.asarray(self._slam_system.get_frame_pose(), dtype=float)
        if tcw.shape != (4, 4) or not np.all(np.isfinite(tcw)):
            return self._lost_pose(timestamp)
        twc = np.linalg.inv(tcw)
        state = str(state_value).upper()
        if "RECENTLY_LOST" in state:
            status = "RECENTLY_LOST"
        else:
            status = (
                "OK" if state in {"2", "OK", "TRACKING_OK"} or "OK" in state else "LOST"
            )
        return SLAMPose(
            position=twc[:3, 3].copy(),
            orientation=Rotation.from_matrix(twc[:3, :3]).as_quat(),
            timestamp=timestamp,
            tracking_status=status,
            num_map_points=self._map_points_count,
        )

    def _track_fallback(
        self, rgb: np.ndarray, depth: np.ndarray, timestamp: float
    ) -> SLAMPose:
        import cv2

        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=self.config.n_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 8:
            return self._lost_pose(timestamp)
        if self._prev_descriptors is None or self._prev_depth is None:
            self._prev_keypoints, self._prev_descriptors = keypoints, descriptors
            self._prev_depth = np.asarray(depth).copy()
            return SLAMPose(
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp,
                "OK",
                len(keypoints),
            )

        matches = sorted(
            cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
                self._prev_descriptors, descriptors
            ),
            key=lambda match: match.distance,
        )
        object_points, image_points = [], []
        previous_depth = self._prev_depth
        for match in matches:
            u, v = self._prev_keypoints[match.queryIdx].pt
            x, y = int(round(u)), int(round(v))
            if not (
                0 <= x < previous_depth.shape[1] and 0 <= y < previous_depth.shape[0]
            ):
                continue
            z = float(previous_depth[y, x])
            if previous_depth.dtype == np.uint16:
                z /= 1000.0
            if not np.isfinite(z) or not (0.1 < z <= self.config.depth_threshold):
                continue
            object_points.append(
                [
                    (u - self.config.cx) * z / self.config.fx,
                    (v - self.config.cy) * z / self.config.fy,
                    z,
                ]
            )
            image_points.append(keypoints[match.trainIdx].pt)
        self._prev_keypoints, self._prev_descriptors = keypoints, descriptors
        self._prev_depth = np.asarray(depth).copy()
        if len(object_points) < 6:
            return self._lost_pose(timestamp)
        camera = np.array(
            [
                [self.config.fx, 0, self.config.cx],
                [0, self.config.fy, self.config.cy],
                [0, 0, 1],
            ],
            dtype=float,
        )
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, np.float32),
            np.asarray(image_points, np.float32),
            camera,
            None,
        )
        if not success or inliers is None or len(inliers) < 6:
            return self._lost_pose(timestamp)
        current_from_previous = np.eye(4)
        current_from_previous[:3, :3] = cv2.Rodrigues(rvec)[0]
        current_from_previous[:3, 3] = tvec[:, 0]
        previous_world = (
            self._current_pose.to_matrix() if self._current_pose else np.eye(4)
        )
        current_world = previous_world @ np.linalg.inv(current_from_previous)
        return SLAMPose(
            current_world[:3, 3],
            Rotation.from_matrix(current_world[:3, :3]).as_quat(),
            timestamp,
            "OK",
            len(inliers),
        )

    def insert_keyframe(self, rgb: np.ndarray, depth: np.ndarray) -> bool:
        if self._current_pose is None:
            return False
        self._keyframe_count += 1
        return True

    def perform_loop_closure(self) -> bool:
        return bool(not self._use_fallback and self.config.enable_loop_closure)

    def get_current_pose(self) -> SLAMPose | None:
        return self._current_pose

    def get_map_points_count(self) -> int:
        return self._map_points_count

    def get_keyframe_count(self) -> int:
        return self._keyframe_count

    def is_tracking(self) -> bool:
        return (
            self._current_pose is not None
            and self._current_pose.tracking_status == "OK"
        )

    def reset(self) -> None:
        with self._lock:
            if self._slam_system is not None and hasattr(self._slam_system, "reset"):
                self._slam_system.reset()
            self._current_pose = None
            self._frame_count = self._keyframe_count = self._map_points_count = 0
            self._last_frame_timestamp = None
            self._prev_keypoints = self._prev_descriptors = self._prev_depth = None

    def save_map(self, path: str) -> bool:
        logger.warning("The Python ORB-SLAM3 binding does not expose atlas persistence")
        return False

    def load_map(self, path: str) -> bool:
        logger.warning("The Python ORB-SLAM3 binding does not expose atlas persistence")
        return False

    def shutdown(self) -> None:
        if self._imu_source is not None:
            self._imu_source.close()
        if self._slam_system is not None and hasattr(self._slam_system, "shutdown"):
            self._slam_system.shutdown()
        self._is_initialized = False
