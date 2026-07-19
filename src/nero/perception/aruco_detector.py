"""OpenCV ArUco marker detector implementing Nero's object-detector contract."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .object_detector import ObjectDetection, project_bbox_to_3d

logger = logging.getLogger(__name__)


def load_marker_map(path: str | Path) -> dict[int, str]:
    """Load a JSON object mapping numeric ArUco IDs to human object names."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load ArUco marker map {source}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("ArUco marker map must be a non-empty JSON object")
    try:
        marker_map = {int(marker_id): _normalize_name(name) for marker_id, name in payload.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("ArUco marker IDs must be integers and names must be strings") from exc
    if any(marker_id < 0 for marker_id in marker_map):
        raise ValueError("ArUco marker IDs must be non-negative")
    if len(set(marker_map.values())) != len(marker_map):
        raise ValueError("ArUco object names must be unique")
    return marker_map


def _normalize_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError("object name must be a string")
    normalized = " ".join(name.lower().split())
    if not normalized:
        raise ValueError("object name must not be empty")
    return normalized


class ArucoObjectDetector:
    """Detect mapped fiducials in the K1 RGB stream and localize them with depth."""

    backend = "aruco"

    def __init__(
        self,
        marker_map: Mapping[int, str] | None = None,
        *,
        mapping_path: str | Path | None = None,
        dictionary: str | None = None,
        depth_threshold_min: float = 0.2,
        depth_threshold_max: float = 5.0,
    ) -> None:
        configured_path = mapping_path or os.getenv("NERO_ARUCO_MAP")
        if marker_map is not None and configured_path is not None:
            raise ValueError("provide either marker_map or mapping_path, not both")
        self._marker_map = (
            {int(marker_id): _normalize_name(name) for marker_id, name in marker_map.items()}
            if marker_map is not None
            else (load_marker_map(configured_path) if configured_path else {})
        )
        if len(set(self._marker_map.values())) != len(self._marker_map):
            raise ValueError("ArUco object names must be unique")
        self.dictionary_name = dictionary or os.getenv("NERO_ARUCO_DICTIONARY", "DICT_4X4_50")
        self.depth_threshold_min = depth_threshold_min
        self.depth_threshold_max = depth_threshold_max
        self._target_name: str | None = None
        self._detector = None

    def initialize(self) -> bool:
        if not self._marker_map:
            logger.error("ArUco requires --aruco-map or NERO_ARUCO_MAP")
            return False
        dictionary_id = getattr(cv2.aruco, self.dictionary_name, None)
        if not isinstance(dictionary_id, int):
            logger.error("Unknown OpenCV ArUco dictionary: %s", self.dictionary_name)
            return False
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )
        logger.info(
            "ArUco detector initialized (%s, %d mapped markers)",
            self.dictionary_name,
            len(self._marker_map),
        )
        return True

    @property
    def supported_targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._marker_map.values()))

    @property
    def result_revision(self) -> None:
        return None

    def resolve_target(self, object_name: str) -> str | None:
        normalized = _normalize_name(object_name)
        return normalized if normalized in self._marker_map.values() else None

    def supports_target(self, object_name: str) -> bool:
        return self.resolve_target(object_name) is not None

    def set_target(self, object_name: str) -> None:
        resolved = self.resolve_target(object_name)
        if resolved is None:
            choices = ", ".join(self.supported_targets)
            raise ValueError(f"unmapped ArUco target {object_name!r}; choose one of: {choices}")
        self._target_name = resolved

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        camera_info=None,
    ) -> list[ObjectDetection]:
        if self._detector is None:
            raise RuntimeError("ArUco detector is not initialized")
        image = np.asarray(rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        height, width = gray.shape[:2]
        detections = []
        for marker_corners, marker_id_array in zip(corners, ids):
            marker_id = int(np.asarray(marker_id_array).reshape(-1)[0])
            label = self._marker_map.get(marker_id)
            if label is None:
                continue
            points = np.asarray(marker_corners).reshape(-1, 2)
            x_min, y_min = np.floor(points.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(points.max(axis=0)).astype(int)
            bbox = (
                max(0, x_min),
                max(0, y_min),
                min(width - 1, x_max),
                min(height - 1, y_max),
            )
            position = (
                project_bbox_to_3d(
                    bbox,
                    depth,
                    camera_info,
                    depth_threshold_min=self.depth_threshold_min,
                    depth_threshold_max=self.depth_threshold_max,
                )
                if depth is not None
                else None
            )
            detections.append(
                ObjectDetection(
                    label=label,
                    confidence=1.0,
                    bbox=bbox,
                    position_3d=position,
                    distance=float(np.linalg.norm(position)) if position is not None else 0.0,
                )
            )
        return detections

    def find_object(
        self, detections: list[ObjectDetection], target_name: str
    ) -> ObjectDetection | None:
        resolved = self.resolve_target(target_name)
        matches = [detection for detection in detections if detection.label == resolved]
        return min(matches, key=lambda detection: detection.distance) if matches else None

    def close(self) -> None:
        self._detector = None
