"""Object detection and 3D localization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectDetection:
    """A detected object with 2D bounding box and optional 3D position."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in camera frame
    distance: float = 0.0  # Euclidean distance to object

    @property
    def center(self) -> tuple[float, float]:
        """Center of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def size(self) -> tuple[int, int]:
        """Size of bounding box."""
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


class ObjectDetector:
    """Detects objects in RGB images and computes their 3D positions.

    Uses the boosteros Detection API when available, with a fallback
    to simple color/shape-based detection for development.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        depth_threshold_min: float = 0.2,
        depth_threshold_max: float = 5.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.depth_threshold_min = depth_threshold_min
        self.depth_threshold_max = depth_threshold_max
        self._detection_api = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize detection API.

        Returns True if the boosteros Detection API is available.
        """
        try:
            from boosteros.brain import Detection
            self._detection_api = Detection()
            self._initialized = True
            logger.info("Detection API initialized")
            return True
        except ImportError:
            logger.warning("boosteros[brain] not available, using fallback detection")
            self._initialized = False
            return False

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Detect objects in RGB image and compute 3D positions.

        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) uint16 or float32
            camera_info: CameraInfo for 3D projection

        Returns:
            List of ObjectDetection
        """
        if self._initialized and self._detection_api is not None:
            return self._detect_with_api(rgb, depth, camera_info)
        else:
            return self._detect_fallback(rgb, depth, camera_info)

    def _detect_with_api(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Detect using boosteros Detection API."""
        try:
            results = self._detection_api.detect(rgb)
            detections = []
            for result in results:
                if result.confidence < self.confidence_threshold:
                    continue

                bbox = (
                    int(result.bbox.x_min),
                    int(result.bbox.y_min),
                    int(result.bbox.x_max),
                    int(result.bbox.y_max),
                )

                # Compute 3D position from depth
                pos_3d = self._compute_3d_position(bbox, depth, camera_info)
                distance = np.linalg.norm(pos_3d) if pos_3d is not None else 0.0

                detections.append(ObjectDetection(
                    label=result.label,
                    confidence=result.confidence,
                    bbox=bbox,
                    position_3d=pos_3d,
                    distance=distance,
                ))

            return detections
        except Exception as e:
            logger.error(f"Detection API error: {e}")
            return self._detect_fallback(rgb, depth, camera_info)

    def _detect_fallback(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Fallback detection using simple heuristics.

        This is a placeholder for development when the Detection API
        is not available. In production, always use boosteros[brain].
        """
        # Return empty list - no detection without proper API
        logger.debug("Fallback detection: no objects detected")
        return []

    def find_object(
        self,
        detections: list[ObjectDetection],
        target_name: str,
    ) -> Optional[ObjectDetection]:
        """Find the closest detection matching the target name.

        Args:
            detections: List of detections
            target_name: Object name to find

        Returns:
            Closest matching detection or None
        """
        target_lower = target_name.lower()
        matches = [
            d for d in detections
            if target_lower in d.label.lower()
        ]

        if not matches:
            return None

        # Return closest match
        return min(matches, key=lambda d: d.distance)

    def _compute_3d_position(
        self,
        bbox: tuple[int, int, int, int],
        depth: np.ndarray,
        camera_info=None,
    ) -> Optional[np.ndarray]:
        """Compute 3D position from bounding box and depth.

        Uses the center of the bounding box and median depth within
        a small region around the center.

        Args:
            bbox: (x_min, y_min, x_max, y_max)
            depth: Depth image
            camera_info: CameraInfo with intrinsics

        Returns:
            [x, y, z] in camera frame, or None if invalid
        """
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Get depth in a small region around center
        region_size = 5
        y_start = max(0, cy - region_size)
        y_end = min(depth.shape[0], cy + region_size)
        x_start = max(0, cx - region_size)
        x_end = min(depth.shape[1], cx + region_size)

        region_depth = depth[y_start:y_end, x_start:x_end]

        # Filter invalid depths
        if region_depth.dtype == np.uint16:
            region_depth = region_depth.astype(np.float32) / 1000.0  # mm to m

        valid = region_depth[
            (region_depth >= self.depth_threshold_min) &
            (region_depth <= self.depth_threshold_max)
        ]

        if len(valid) == 0:
            return None

        z = float(np.median(valid))

        # Get camera intrinsics
        if camera_info is not None:
            fx = camera_info.k[0, 0]
            fy = camera_info.k[1, 1]
            cx_cam = camera_info.k[0, 2]
            cy_cam = camera_info.k[1, 2]
        else:
            # Default intrinsics for K1
            fx = fy = 216.5
            cx_cam = 160.0
            cy_cam = 120.0

        # Back-project to 3D
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        return np.array([x, y, z])