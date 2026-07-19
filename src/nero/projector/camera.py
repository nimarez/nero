"""RealSense capture with a latest-frame ArUco overlay."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarkerDetection:
    marker_id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    distance_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.marker_id,
            "corners": [list(point) for point in self.corners],
            "center": list(self.center),
            "distance_m": self.distance_m,
        }


@dataclass(frozen=True, slots=True)
class CameraFrame:
    sequence: int
    captured_at: float
    image: np.ndarray
    jpeg: bytes
    detections: tuple[MarkerDetection, ...]


class RealSenseArucoCamera:
    """Own one D435i color stream and always expose only the newest frame."""

    def __init__(
        self,
        *,
        marker_ids: tuple[int, ...] = (1, 2, 3, 4),
        marker_size_m: float = 0.130,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 72,
        web_width: int = 960,
    ) -> None:
        self.marker_ids = marker_ids
        self.marker_size_m = marker_size_m
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.web_width = web_width
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def start(self) -> "RealSenseArucoCamera":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="realsense-aruco", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _set_error(self, message: str | None) -> None:
        with self._lock:
            self._error = message

    def _publish(self, frame: CameraFrame) -> None:
        with self._lock:
            self._latest = frame
            self._error = None

    def _run(self) -> None:
        pipeline = None
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            profile = pipeline.start(config)
            stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = stream_profile.get_intrinsics()
            camera_matrix = np.asarray(
                ((intrinsics.fx, 0, intrinsics.ppx), (0, intrinsics.fy, intrinsics.ppy), (0, 0, 1)),
                dtype=np.float64,
            )
            distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            sequence = 0
            logger.info("RealSense color stream started at %dx%d@%d", self.width, self.height, self.fps)

            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if not color:
                    continue
                image = np.asanyarray(color.get_data()).copy()
                annotated, detections = annotate_aruco(
                    image,
                    detector=detector,
                    expected_ids=self.marker_ids,
                    marker_size_m=self.marker_size_m,
                    camera_matrix=camera_matrix,
                    distortion=distortion,
                )
                web_height = round(annotated.shape[0] * self.web_width / annotated.shape[1])
                web_image = cv2.resize(
                    annotated, (self.web_width, web_height), interpolation=cv2.INTER_AREA
                )
                ok, encoded = cv2.imencode(
                    ".jpg", web_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue
                sequence += 1
                self._publish(
                    CameraFrame(
                        sequence=sequence,
                        captured_at=time.time(),
                        image=annotated,
                        jpeg=encoded.tobytes(),
                        detections=detections,
                    )
                )
        except Exception as error:
            logger.exception("RealSense capture stopped")
            self._set_error(str(error))
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


def _marker_distance(
    corners: np.ndarray,
    *,
    marker_size_m: float,
    camera_matrix: np.ndarray | None,
    distortion: np.ndarray | None,
) -> float | None:
    if camera_matrix is None:
        return None
    half = marker_size_m / 2.0
    object_points = np.asarray(
        ((-half, half, 0), (half, half, 0), (half, -half, 0), (-half, -half, 0)),
        dtype=np.float32,
    )
    try:
        ok, _, translation = cv2.solvePnP(
            object_points,
            corners.astype(np.float32),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    distance = float(np.linalg.norm(translation))
    return distance if math.isfinite(distance) else None


def annotate_aruco(
    image: np.ndarray,
    *,
    detector: Any | None = None,
    expected_ids: tuple[int, ...] = (1, 2, 3, 4),
    marker_size_m: float = 0.130,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[MarkerDetection, ...]]:
    """Detect IDs 1-4 and draw an uncluttered high-contrast camera overlay."""

    if detector is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grayscale)
    annotated = image.copy()
    found: list[MarkerDetection] = []
    expected = set(expected_ids)

    if ids is not None:
        for raw_id, raw_corners in zip(ids.flatten(), corners):
            marker_id = int(raw_id)
            if marker_id not in expected:
                continue
            points = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
            center = tuple(float(value) for value in points.mean(axis=0))
            distance = _marker_distance(
                points,
                marker_size_m=marker_size_m,
                camera_matrix=camera_matrix,
                distortion=distortion,
            )
            polyline = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [polyline], True, (105, 255, 145), 4, cv2.LINE_AA)
            center_px = tuple(np.rint(center).astype(int))
            cv2.circle(annotated, center_px, 8, (0, 145, 255), -1, cv2.LINE_AA)
            label = f"ID {marker_id}"
            origin = tuple(np.rint(points[0] + np.array((0, -14))).astype(int))
            cv2.putText(
                annotated,
                label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            found.append(
                MarkerDetection(
                    marker_id=marker_id,
                    corners=tuple(tuple(float(value) for value in point) for point in points),
                    center=center,
                    distance_m=distance,
                )
            )

    found.sort(key=lambda item: item.marker_id)
    status = f"DICT_4X4_50 · 130 mm · visible {len(found)}/4"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        status,
        (20, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (118, 255, 148) if len(found) == 4 else (0, 160, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated, tuple(found)
