"""External camera stream handler.

Supports multiple input sources:
- USB cameras (OpenCV)
- RTSP streams
- HTTP MJPEG streams
- Video files (for testing)
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraSource(Enum):
    """Supported camera source types."""
    USB = "usb"
    RTSP = "rtsp"
    HTTP = "http"
    FILE = "file"


class CameraStream:
    """Handles external camera streaming.

    Supports USB, RTSP, HTTP MJPEG, and video file sources.
    Runs capture in a background thread for non-blocking access.
    """

    def __init__(
        self,
        source: str,
        source_type: CameraSource = CameraSource.USB,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
    ):
        """Initialize camera stream.

        Args:
            source: Camera source (device index, URL, or file path)
            source_type: Type of camera source
            width: Frame width
            height: Frame height
            fps: Target frames per second
            buffer_size: Frame buffer size (1 for latest frame only)
        """
        self.source = source
        self.source_type = source_type
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0

    def start(self) -> bool:
        """Start the camera stream.

        Returns:
            True if stream started successfully
        """
        logger.info(f"Starting camera stream: {self.source_type.value} - {self.source}")

        # Open capture based on source type
        if self.source_type == CameraSource.USB:
            self._cap = cv2.VideoCapture(int(self.source))
        elif self.source_type in (CameraSource.RTSP, CameraSource.HTTP):
            self._cap = cv2.VideoCapture(self.source)
        elif self.source_type == CameraSource.FILE:
            self._cap = cv2.VideoCapture(self.source)
        else:
            logger.error(f"Unsupported source type: {self.source_type}")
            return False

        if not self._cap or not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self.source}")
            return False

        # Set properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Camera stream started: {self.width}x{self.height}@{self.fps}")
        return True

    def stop(self) -> None:
        """Stop the camera stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stream stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame.

        Returns:
            BGR image or None if not available
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_fps(self) -> float:
        """Get current capture FPS."""
        return self._current_fps

    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running and self._cap is not None and self._cap.isOpened()

    def _capture_loop(self) -> None:
        """Background thread for capturing frames."""
        while self._running:
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame

                # Update FPS counter
                self._fps_counter += 1
                now = time.time()
                if now - self._fps_time >= 1.0:
                    self._current_fps = self._fps_counter / (now - self._fps_time)
                    self._fps_counter = 0
                    self._fps_time = now
            else:
                # Try to reconnect on failure
                logger.warning("Frame capture failed, attempting reconnect...")
                time.sleep(1.0)
                if not self._reconnect():
                    logger.error("Reconnect failed")
                    break

    def _reconnect(self) -> bool:
        """Attempt to reconnect to camera."""
        if self._cap:
            self._cap.release()

        if self.source_type == CameraSource.USB:
            self._cap = cv2.VideoCapture(int(self.source))
        elif self.source_type in (CameraSource.RTSP, CameraSource.HTTP):
            self._cap = cv2.VideoCapture(self.source)
        elif self.source_type == CameraSource.FILE:
            self._cap = cv2.VideoCapture(self.source)

        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()