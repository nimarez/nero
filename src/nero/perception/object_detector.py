"""Object detection and 3D localization."""

from __future__ import annotations

import logging
import math
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO80 = tuple(
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,"
    "teddy bear,hair drier,toothbrush".split(",")
)


@dataclass
class ObjectDetection:
    """A detected object with 2D bounding box and optional 3D position."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in camera frame
    distance: float = 0.0  # Euclidean distance to object
    coordinate_frame: str = "camera"  # "camera" or "body"

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

    @property
    def angle(self) -> float:
        """Horizontal bearing to the object in radians."""
        if self.position_3d is None:
            return 0.0
        return math.atan2(float(self.position_3d[0]), float(self.position_3d[2]))


class ObjectDetector:
    """Detects objects in RGB images and computes their 3D positions.

    Uses OpenCV DNN with a local YOLOv8 COCO model. No network access occurs in
    the policy loop.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        depth_threshold_min: float = 0.2,
        depth_threshold_max: float = 5.0,
        model_path: str | Path | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.depth_threshold_min = depth_threshold_min
        self.depth_threshold_max = depth_threshold_max
        self._net = None
        self._world_model = None
        self._target_name: str | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[list[ObjectDetection]] | None = None
        self._latest_detections: list[ObjectDetection] = []
        self._result_revision = 0
        self._result_lock = threading.Lock()
        self.model_path = Path(
            model_path
            or os.getenv("NERO_OBJECT_MODEL", "config/yolov8s-worldv2.pt")
        )
        self._initialized = False

    def initialize(self) -> bool:
        """Load the repository's supported object-detection backend."""
        if not self.model_path.is_file():
            logger.error(
                "Object model missing: %s (run scripts/setup_object_detector.sh)",
                self.model_path,
            )
            return False
        if self.model_path.suffix.lower() == ".pt":
            try:
                import torch
                from ultralytics import YOLOWorld

                torch.set_num_threads(max(1, int(os.getenv("NERO_YOLO_THREADS", "4"))))
                self._world_model = YOLOWorld(str(self.model_path))
                # Load/cache the CLIP text tower before a human gives a command.
                self._world_model.set_classes(["object"])
            except (ImportError, RuntimeError, OSError, ValueError) as exc:
                logger.error("Could not load YOLO-World model %s: %s", self.model_path, exc)
                return False
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nero-yolo-world"
            )
            self._initialized = True
            logger.info("Ultralytics YOLO-World initialized from %s", self.model_path)
            return True
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            logger.error("Could not load object model %s: %s", self.model_path, exc)
            return False
        self._initialized = True
        logger.info("OpenCV YOLO object detector initialized from %s", self.model_path)
        return True

    @property
    def result_revision(self) -> int | None:
        """Increment when an asynchronous open-vocabulary result completes."""
        return self._result_revision if self._world_model is not None else None

    def set_target(self, object_name: str) -> None:
        """Condition the open-vocabulary detector on an arbitrary text prompt."""
        normalized = " ".join(object_name.split()).strip()
        if not normalized:
            raise ValueError("object target must not be empty")
        if self._world_model is None:
            self._target_name = normalized
            return
        if self._future is not None:
            self._future.result()
            self._future = None
        self._world_model.set_classes([normalized])
        with self._result_lock:
            self._target_name = normalized
            self._latest_detections = []
            self._result_revision = 0
        logger.info("YOLO-World prompt set to %r", normalized)

    def detect(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
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
        if self._world_model is not None:
            return self._detect_world_async(rgb, depth, camera_info)
        if self._net is not None:
            return self._detect_yolo(rgb, depth, camera_info)
        return []

    def _detect_world_async(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Submit only the newest frame and return the latest completed result."""
        if self._target_name is None or self._executor is None:
            return []
        if self._future is not None and self._future.done():
            try:
                completed = self._future.result()
            except Exception:
                logger.exception("YOLO-World inference failed")
                completed = []
            with self._result_lock:
                self._latest_detections = completed
                self._result_revision += 1
            self._future = None
        if self._future is None:
            rgb_copy = np.ascontiguousarray(rgb).copy()
            depth_copy = None if depth is None else np.ascontiguousarray(depth).copy()
            self._future = self._executor.submit(
                self._detect_world, rgb_copy, depth_copy, camera_info
            )
        with self._result_lock:
            return list(self._latest_detections)

    def _detect_world(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Run one target-conditioned YOLO-World inference."""
        results = self._world_model.predict(
            np.asarray(rgb),
            imgsz=448,
            conf=self.confidence_threshold,
            device="cpu",
            verbose=False,
        )
        detections: list[ObjectDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            names = getattr(result, "names", {})
            for bounds, score, class_id in zip(xyxy, scores, class_ids):
                bbox = tuple(int(round(float(value))) for value in bounds)
                position = (
                    self._compute_3d_position(bbox, depth, camera_info)
                    if depth is not None
                    else None
                )
                label = (
                    names.get(class_id, self._target_name or "object")
                    if hasattr(names, "get")
                    else names[class_id]
                )
                detections.append(
                    ObjectDetection(
                        label=str(label),
                        confidence=float(score),
                        bbox=bbox,
                        position_3d=position,
                        distance=(
                            float(np.linalg.norm(position)) if position is not None else 0.0
                        ),
                    )
                )
        return detections

    def close(self) -> None:
        """Stop the optional asynchronous inference worker."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _detect_yolo(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Run an Ultralytics YOLOv8 ONNX export with OpenCV DNN."""
        if self._net is None:
            return []
        image = np.asarray(rgb)
        height, width = image.shape[:2]
        size = 640
        scale = min(size / width, size / height)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (resized_w, resized_h))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - resized_w) // 2, (size - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 255.0, (size, size), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        output = np.asarray(self._net.forward()).squeeze()
        if output.ndim == 1 and output.size == 4 + len(COCO80):
            output = output.reshape(4 + len(COCO80), 1)
        if output.ndim != 2:
            raise RuntimeError(f"unexpected YOLO output shape: {output.shape}")
        if output.shape[0] == 4 + len(COCO80):
            output = output.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in output:
            class_scores = row[4 : 4 + len(COCO80)]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence_threshold:
                continue
            cx, cy, box_w, box_h = (float(value) for value in row[:4])
            x = int(round((cx - box_w / 2 - pad_x) / scale))
            y = int(round((cy - box_h / 2 - pad_y) / scale))
            w = int(round(box_w / scale))
            h = int(round(box_h / scale))
            x, y = max(0, x), max(0, y)
            w, h = min(width - x, w), min(height - y, h)
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, w, h])
            scores.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, 0.45
        )
        detections = []
        for index in np.asarray(indices).reshape(-1) if len(indices) else []:
            x, y, w, h = boxes[int(index)]
            bbox = (x, y, x + w, y + h)
            position = (
                self._compute_3d_position(bbox, depth, camera_info)
                if depth is not None
                else None
            )
            detections.append(
                ObjectDetection(
                    label=COCO80[class_ids[int(index)]],
                    confidence=scores[int(index)],
                    bbox=bbox,
                    position_3d=position,
                    distance=float(np.linalg.norm(position)) if position is not None else 0.0,
                )
            )
        return detections

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
        matches = [d for d in detections if target_lower in d.label.lower()]

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
            (region_depth >= self.depth_threshold_min)
            & (region_depth <= self.depth_threshold_max)
        ]

        if len(valid) == 0:
            return None

        z = float(np.median(valid))

        # Get camera intrinsics
        if camera_info is not None:
            matrix = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
            fx = matrix[0, 0]
            fy = matrix[1, 1]
            cx_cam = matrix[0, 2]
            cy_cam = matrix[1, 2]
        else:
            # Default intrinsics for K1
            fx = fy = 216.5
            cx_cam = 160.0
            cy_cam = 120.0

        # Back-project to 3D
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        return np.array([x, y, z])
