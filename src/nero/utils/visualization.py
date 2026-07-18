"""Visualization utilities for camera streams and navigation."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from nero.perception.object_detector import ObjectDetection

logger = logging.getLogger(__name__)


class Visualization:
    """Handles rendering and visualization of camera streams."""

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: list[ObjectDetection],
        target_name: Optional[str] = None,
    ) -> np.ndarray:
        """Draw detection bounding boxes on image.

        Args:
            image: BGR image
            detections: List of object detections
            target_name: Name of target object (highlighted differently)

        Returns:
            Image with detections drawn
        """
        img = image.copy()

        for det in detections:
            bbox = det.bbox
            is_target = target_name and det.class_name.lower() == target_name.lower()

            # Choose color
            color = (0, 255, 0) if is_target else (255, 255, 0)
            thickness = 3 if is_target else 2

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.distance is not None:
                label += f" {det.distance:.1f}m"

            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

        return img

    @staticmethod
    def draw_navigation_info(
        image: np.ndarray,
        state: str,
        message: str,
        fps: float = 0.0,
        velocity: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Draw navigation status overlay.

        Args:
            image: BGR image
            state: Current policy state
            message: Status message
            fps: Current FPS
            velocity: (linear, angular) velocity

        Returns:
            Image with overlay
        """
        img = image.copy()
        h, w = img.shape[:2]

        # Semi-transparent overlay at top
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # State
        cv2.putText(
            img, f"State: {state}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Message
        cv2.putText(
            img, f"Msg: {message}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # FPS
        cv2.putText(
            img, f"FPS: {fps:.1f}", (w - 100, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Velocity
        if velocity:
            linear, angular = velocity
            cv2.putText(
                img, f"v={linear:.2f} w={angular:.2f}", (w - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return img

    @staticmethod
    def draw_crosshair(
        image: np.ndarray,
        center: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 255),
        size: int = 20,
    ) -> np.ndarray:
        """Draw crosshair at center point.

        Args:
            image: BGR image
            center: (x, y) center point
            color: BGR color
            size: Crosshair size

        Returns:
            Image with crosshair
        """
        img = image.copy()
        x, y = center
        cv2.line(img, (x - size, y), (x + size, y), color, 2)
        cv2.line(img, (x, y - size), (x, y + size), color, 2)
        cv2.circle(img, center, 5, color, 2)
        return img

    @staticmethod
    def show_stream(
        image: np.ndarray,
        window_name: str = "Camera Stream",
        fps: float = 0.0,
    ) -> int:
        """Display image in window.

        Args:
            image: BGR image
            window_name: Window name
            fps: Current FPS to display

        Returns:
            Key pressed (ord('q') to quit)
        """
        cv2.imshow(window_name, image)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def save_frame(image: np.ndarray, path: str) -> None:
        """Save frame to file.

        Args:
            image: BGR image
            path: Output path
        """
        cv2.imwrite(path, image)
        logger.info(f"Saved frame to {path}")