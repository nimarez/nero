"""Object detection and 3D localization."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import platform
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
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

QNN_BACKEND = "yolo-world-qnn"
MODAL_BACKEND = "yolo-world-modal"
OPEN_VOCAB_BACKENDS = ("yolo-world", QNN_BACKEND, MODAL_BACKEND, "yoloe")
_BACKEND_ALIASES = {
    "world": "yolo-world",
    "yolo-world": "yolo-world",
    "yoloworld": "yolo-world",
    "qnn": QNN_BACKEND,
    "yolo-world-qnn": QNN_BACKEND,
    "yoloworld-qnn": QNN_BACKEND,
    "modal": MODAL_BACKEND,
    "yolo-world-modal": MODAL_BACKEND,
    "yoloworld-modal": MODAL_BACKEND,
    "yoloe": "yoloe",
    "opencv": "opencv",
    "onnx": "opencv",
}
_DEFAULT_MODELS = {
    "yolo-world": "config/yolov8s-worldv2.pt",
    QNN_BACKEND: "config/yolov8s-worldv2-open-vocab-256-qnn/model.onnx",
    MODAL_BACKEND: "yolov8s-worldv2.pt",
    "yoloe": "config/yoloe-26n-seg.pt",
    "opencv": "config/yolov8n.onnx",
}
_DEFAULT_INFERENCE_SIZES = {
    "yolo-world": 256,
    QNN_BACKEND: 256,
    MODAL_BACKEND: 256,
    "yoloe": 320,
    "opencv": 640,
}

_PROMPT_WORKER_MODEL = None
_PROMPT_WORKER_BACKEND = ""


def _parse_cpu_list(value: str) -> set[int]:
    """Parse a Linux CPU list such as ``6,7`` for detector isolation."""
    cpus = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(cpu < 0 for cpu in cpus):
        raise ValueError("detector CPU indices must be non-negative")
    return cpus


def configure_qualcomm_cpu_partition() -> tuple[set[int], set[int]] | None:
    """Reserve the fastest two K1 CPUs for the isolated detector process."""
    if os.getenv("NERO_CPU_PARTITION", "1") == "0":
        return None
    if not (
        platform.system() == "Linux"
        and platform.machine().lower() in {"aarch64", "arm64"}
        and hasattr(os, "sched_getaffinity")
        and hasattr(os, "sched_setaffinity")
    ):
        return None
    try:
        model = Path("/proc/device-tree/model").read_text(errors="replace").lower()
    except OSError:
        return None
    if "qualcomm" not in model and "qcs8550" not in model and "kalamap" not in model:
        return None
    allowed = set(os.sched_getaffinity(0))
    if len(allowed) < 6:
        return None

    def max_frequency(cpu: int) -> int:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return 0

    ranked = sorted(allowed, key=lambda cpu: (max_frequency(cpu), cpu), reverse=True)
    detector = set(ranked[:2])
    navigation = allowed - detector
    if len(navigation) < 2:
        return None
    os.environ.setdefault(
        "NERO_DETECTOR_CPUS", ",".join(str(cpu) for cpu in sorted(detector))
    )
    os.sched_setaffinity(0, navigation)
    logger.info(
        "QCS8550 CPU partition: navigation=%s detector=%s",
        sorted(navigation),
        sorted(detector),
    )
    return navigation, detector


def _prompt_worker_initialize(
    backend: str,
    model_path: str,
    text_model_path: str | None,
    threads: int,
    inference_size: int,
    confidence_threshold: float,
    max_detections: int,
    warmup: bool,
    cpu_list: str,
) -> None:
    """Load the open-vocabulary model entirely inside its worker process."""
    global _PROMPT_WORKER_BACKEND, _PROMPT_WORKER_MODEL
    if cpu_list and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, _parse_cpu_list(cpu_list))

    import torch

    torch.set_num_threads(max(1, threads))
    set_interop_threads = getattr(torch, "set_num_interop_threads", None)
    if set_interop_threads is not None:
        try:
            set_interop_threads(1)
        except RuntimeError:
            pass
    if backend == "yoloe":
        from ultralytics import SETTINGS, YOLOE

        if text_model_path is not None:
            SETTINGS["weights_dir"] = str(Path(text_model_path).resolve().parent)
        model_class = YOLOE
    else:
        from ultralytics import YOLOWorld

        model_class = YOLOWorld

    model = model_class(model_path)
    model.set_classes(["object"])
    if warmup:
        model.predict(
            np.zeros((448, 544, 3), dtype=np.uint8),
            imgsz=inference_size,
            conf=confidence_threshold,
            device="cpu",
            max_det=max_detections,
            rect=True,
            verbose=False,
        )
    _PROMPT_WORKER_BACKEND = backend
    _PROMPT_WORKER_MODEL = model


def _prompt_worker_ready() -> bool:
    return _PROMPT_WORKER_MODEL is not None


def _prompt_worker_set_target(target: str) -> str:
    if _PROMPT_WORKER_MODEL is None:
        raise RuntimeError("open-vocabulary worker is not initialized")
    _PROMPT_WORKER_MODEL.set_classes([target])
    return target


def _prompt_worker_detect(
    rgb: np.ndarray,
    inference_size: int,
    confidence_threshold: float,
    max_detections: int,
    target_name: str,
) -> tuple[list[tuple[str, float, tuple[int, int, int, int]]], float]:
    """Run one image-only inference; depth projection stays in the parent."""
    if _PROMPT_WORKER_MODEL is None:
        raise RuntimeError("open-vocabulary worker is not initialized")
    started = time.perf_counter()
    results = _PROMPT_WORKER_MODEL.predict(
        np.asarray(rgb),
        imgsz=inference_size,
        conf=confidence_threshold,
        device="cpu",
        max_det=max_detections,
        rect=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        names = getattr(result, "names", {})
        for bounds, score, class_id in zip(xyxy, scores, class_ids):
            label = (
                names.get(class_id, target_name)
                if hasattr(names, "get")
                else names[class_id]
            )
            detections.append(
                (
                    str(label),
                    float(score),
                    tuple(int(round(float(value))) for value in bounds),
                )
            )
    return detections, elapsed

_FIXED_VOCAB_ALIASES = {
    "cellphone": "cell phone",
    "dining room table": "dining table",
    "flower pot": "potted plant",
    "mobile phone": "cell phone",
    "motorbike": "motorcycle",
    "plant": "potted plant",
    "sofa": "couch",
    "table": "dining table",
    "television": "tv",
}


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

    Uses QNN-accelerated or CPU prompt-conditioned YOLO-World, or YOLOE, for
    arbitrary text targets, with an explicit OpenCV YOLOv8 ONNX fixed-vocabulary
    fallback. The K1 runtime defaults to fail-closed QNN HTP execution.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        depth_threshold_min: float = 0.2,
        depth_threshold_max: float = 5.0,
        model_path: str | Path | None = None,
        backend: str | None = None,
        text_model_path: str | Path | None = None,
        inference_size: int | None = None,
        max_detections: int | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.depth_threshold_min = depth_threshold_min
        self.depth_threshold_max = depth_threshold_max
        self._net = None
        self._class_names = COCO80
        self._prompt_model = None
        self._qnn_runtime = None
        self._modal_client = None
        self._prompt_encoder = None
        self._target_embedding: np.ndarray | None = None
        self._prompt_process = False
        self._target_name: str | None = None
        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        self._future: Future | None = None
        self._future_depth: np.ndarray | None = None
        self._future_camera_matrix: np.ndarray | None = None
        self._latest_detections: list[ObjectDetection] = []
        self._result_revision = 0
        self._result_lock = threading.Lock()
        configured_model = model_path or os.getenv("NERO_OBJECT_MODEL")
        configured_backend = backend or os.getenv("NERO_OBJECT_BACKEND")
        if configured_backend is None:
            configured_backend = self._infer_backend(configured_model)
        self.backend = self._normalize_backend(configured_backend)
        self.model_path = Path(configured_model or _DEFAULT_MODELS[self.backend])
        self.text_model_path = (
            Path(
                text_model_path
                or os.getenv("NERO_YOLOE_TEXT_MODEL", "config/mobileclip2_b.ts")
            )
            if self.backend == "yoloe"
            else None
        )
        if (
            self.text_model_path is not None
            and self.text_model_path.name != "mobileclip2_b.ts"
        ):
            raise ValueError("YOLOE text model must be named mobileclip2_b.ts")
        self.inference_size = int(
            inference_size
            if inference_size is not None
            else os.getenv(
                "NERO_OBJECT_IMGSZ",
                os.getenv(
                    "NERO_YOLO_IMGSZ", str(_DEFAULT_INFERENCE_SIZES[self.backend])
                ),
            )
        )
        if self.inference_size < 256 or self.inference_size % 32:
            raise ValueError("YOLO inference size must be >= 256 and divisible by 32")
        if self.backend == QNN_BACKEND and self.inference_size != 256:
            raise ValueError("the pinned QNN YOLO-World graph requires inference size 256")
        self.max_detections = int(
            max_detections
            if max_detections is not None
            else os.getenv("NERO_YOLO_MAX_DETECTIONS", "10")
        )
        if self.max_detections < 1:
            raise ValueError("YOLO max detections must be positive")
        self._inference_count = 0
        self._inference_seconds_ema: float | None = None
        process_default = (
            os.name == "posix"
            and platform.system() == "Linux"
            and platform.machine().lower() in {"aarch64", "arm64"}
        )
        process_setting = os.getenv(
            "NERO_DETECTOR_PROCESS", "1" if process_default else "0"
        )
        if process_setting not in {"0", "1"}:
            raise ValueError("NERO_DETECTOR_PROCESS must be 0 or 1")
        self._use_prompt_process = process_setting == "1" and self.backend not in {
            QNN_BACKEND,
            MODAL_BACKEND,
        }
        self._initialized = False

    @staticmethod
    def _infer_backend(model_path: str | Path | None) -> str:
        """Preserve suffix-based compatibility when no backend is configured."""
        if model_path is None:
            if platform.system() == "Linux" and platform.machine().lower() in {
                "aarch64",
                "arm64",
            }:
                return QNN_BACKEND
            return "yolo-world"
        path = Path(model_path)
        if "open-vocab" in path.name.lower() or "qnn" in str(path).lower():
            return QNN_BACKEND
        if "yoloe" in path.name.lower():
            return "yoloe"
        return "yolo-world" if path.suffix.lower() == ".pt" else "opencv"

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = backend.strip().lower().replace("_", "-")
        try:
            return _BACKEND_ALIASES[normalized]
        except KeyError as exc:
            choices = ", ".join(OPEN_VOCAB_BACKENDS + ("opencv",))
            raise ValueError(
                f"unsupported object detector backend {backend!r}; choose {choices}"
            ) from exc

    def initialize(self) -> bool:
        """Load the repository's supported object-detection backend."""
        if self.backend == MODAL_BACKEND:
            return self._initialize_modal()
        if not self.model_path.is_file():
            logger.error(
                "Object model missing: %s (run scripts/setup_object_detector.sh)",
                self.model_path,
            )
            return False
        if self.backend == "yoloe" and not self.text_model_path.is_file():
            logger.error(
                "YOLOE text model missing: %s (run "
                "NERO_OBJECT_BACKEND=yoloe scripts/setup_object_detector.sh)",
                self.text_model_path,
            )
            return False
        if self.backend == QNN_BACKEND:
            return self._initialize_qnn()
        if self.backend in OPEN_VOCAB_BACKENDS:
            if self._use_prompt_process:
                return self._initialize_prompt_process()
            try:
                import torch

                if self.backend == "yoloe":
                    from ultralytics import SETTINGS, YOLOE

                    # Ultralytics resolves the fixed text-tower filename through
                    # its weights directory. Setup puts the verified artifact here.
                    SETTINGS["weights_dir"] = str(self.text_model_path.resolve().parent)
                    model_class = YOLOE
                else:
                    from ultralytics import YOLOWorld

                    model_class = YOLOWorld

                torch.set_num_threads(max(1, int(os.getenv("NERO_YOLO_THREADS", "4"))))
                set_interop_threads = getattr(torch, "set_num_interop_threads", None)
                if set_interop_threads is not None:
                    try:
                        set_interop_threads(1)
                    except RuntimeError:
                        # PyTorch allows this setting only before inter-op work starts.
                        logger.debug("PyTorch inter-op thread count was already fixed")
                self._prompt_model = model_class(str(self.model_path))
                # Resolve the text encoder and its weights before a human gives a command.
                self._prompt_model.set_classes(["object"])
                if os.getenv("NERO_YOLO_WARMUP", "1") != "0":
                    started = time.perf_counter()
                    self._prompt_model.predict(
                        np.zeros((448, 544, 3), dtype=np.uint8),
                        imgsz=self.inference_size,
                        conf=self.confidence_threshold,
                        device="cpu",
                        max_det=self.max_detections,
                        rect=True,
                        verbose=False,
                    )
                    logger.info(
                        "%s warmup completed in %.2fs at imgsz=%d",
                        self.backend,
                        time.perf_counter() - started,
                        self.inference_size,
                    )
            except (
                AttributeError,
                ImportError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                logger.error(
                    "Could not load %s model %s: %s",
                    self.backend,
                    self.model_path,
                    exc,
                )
                return False
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"nero-{self.backend}"
            )
            self._initialized = True
            logger.info(
                "Ultralytics %s initialized from %s", self.backend, self.model_path
            )
            return True
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            logger.error("Could not load object model %s: %s", self.model_path, exc)
            return False
        self._initialized = True
        logger.info("OpenCV YOLO object detector initialized from %s", self.model_path)
        return True

    def _initialize_modal(self) -> bool:
        """Validate and warm the authenticated Modal inference endpoint."""
        try:
            from nero.perception.modal_yolo_world import ModalYoloWorldClient

            self._modal_client = ModalYoloWorldClient()
            if os.getenv("NERO_MODAL_WARMUP", "1") != "0":
                started = time.perf_counter()
                self._modal_client.detect(
                    np.zeros((256, 256, 3), dtype=np.uint8),
                    "object",
                    self.confidence_threshold,
                    self.inference_size,
                    self.max_detections,
                )
                logger.info(
                    "Modal YOLO-World endpoint warmed in %.2fs", time.perf_counter() - started
                )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error("Could not initialize Modal detector: %s", exc)
            self._modal_client = None
            return False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nero-yolo-world-modal"
        )
        self._initialized = True
        logger.info("Modal YOLO-World initialized at %s", self._modal_client.url)
        return True

    def _initialize_qnn(self) -> bool:
        """Load the pinned graph on QNN HTP and preload the prompt text tower."""
        try:
            from nero.perception.qnn_artifact import verify_qnn_artifact
            from nero.perception.qnn_yolo_world import (
                QNNYoloWorldRuntime,
                YoloWorldPromptEncoder,
            )

            verified_model = verify_qnn_artifact(self.model_path.parent)
            if verified_model.resolve() != self.model_path.resolve():
                raise RuntimeError(
                    f"manifest selected {verified_model}, not {self.model_path}"
                )
            self._qnn_runtime = QNNYoloWorldRuntime(
                self.model_path, inference_size=self.inference_size
            )
            self._prompt_encoder = YoloWorldPromptEncoder()
            # Resolve CLIP weights at startup, never after the robot accepts a command.
            self._target_embedding = self._prompt_encoder.encode("object")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error("Could not initialize fail-closed QNN detector: %s", exc)
            self._qnn_runtime = None
            self._prompt_encoder = None
            self._target_embedding = None
            return False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nero-yolo-world-qnn"
        )
        self._initialized = True
        logger.info(
            "QNN YOLO-World initialized from %s (providers=%s)",
            self.model_path,
            self._qnn_runtime.providers,
        )
        return True

    def _initialize_prompt_process(self) -> bool:
        """Start an isolated inference process so SLAM cannot starve the detector."""
        try:
            context = multiprocessing.get_context("spawn")
            threads = max(1, int(os.getenv("NERO_YOLO_THREADS", "2")))
            warmup = os.getenv("NERO_YOLO_WARMUP", "1") != "0"
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_prompt_worker_initialize,
                initargs=(
                    self.backend,
                    str(self.model_path.resolve()),
                    (
                        str(self.text_model_path.resolve())
                        if self.text_model_path is not None
                        else None
                    ),
                    threads,
                    self.inference_size,
                    self.confidence_threshold,
                    self.max_detections,
                    warmup,
                    os.getenv("NERO_DETECTOR_CPUS", ""),
                ),
            )
            if not self._executor.submit(_prompt_worker_ready).result(timeout=180.0):
                raise RuntimeError("open-vocabulary worker did not initialize")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "Could not start isolated %s worker for %s: %s",
                self.backend,
                self.model_path,
                exc,
            )
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            return False
        self._prompt_process = True
        self._initialized = True
        logger.info(
            "Ultralytics %s initialized in an isolated process from %s "
            "(threads=%d, cpus=%s)",
            self.backend,
            self.model_path,
            threads,
            os.getenv("NERO_DETECTOR_CPUS", "scheduler-default"),
        )
        return True

    def resolve_target(self, object_name: str) -> str | None:
        """Return the backend's canonical class name, or ``None`` if unsupported."""
        normalized = " ".join(object_name.lower().split()).strip()
        if not normalized:
            return None
        if self.backend in OPEN_VOCAB_BACKENDS:
            return normalized
        canonical = _FIXED_VOCAB_ALIASES.get(normalized, normalized)
        classes = {name.lower(): name for name in self._class_names}
        return classes.get(canonical)

    def supports_target(self, object_name: str) -> bool:
        """Whether this backend can detect the requested object class."""
        return self.resolve_target(object_name) is not None

    @property
    def supported_targets(self) -> tuple[str, ...] | None:
        """Fixed class vocabulary, or ``None`` for an open-vocabulary backend."""
        return None if self.backend in OPEN_VOCAB_BACKENDS else self._class_names

    @property
    def result_revision(self) -> int | None:
        """Increment when an asynchronous detector result completes."""
        if (
            self._prompt_model is not None
            or self._prompt_process
            or self._qnn_runtime
            or self._modal_client
        ):
            return self._result_revision
        return None

    def set_target(self, object_name: str) -> None:
        """Set an open-vocabulary prompt or select a fixed detector class."""
        normalized = " ".join(object_name.split()).strip()
        if not normalized:
            raise ValueError("object target must not be empty")
        resolved = self.resolve_target(normalized)
        if resolved is None:
            raise ValueError(
                f"{object_name!r} is not supported by the {self.backend} detector"
            )
        if self._prompt_process:
            if self._future is not None:
                self._future.result()
                self._future = None
            if self._executor is None:
                raise RuntimeError("open-vocabulary worker is not initialized")
            acknowledged = self._executor.submit(
                _prompt_worker_set_target, resolved
            ).result(timeout=60.0)
            if acknowledged != resolved:
                raise RuntimeError("open-vocabulary worker rejected the target")
            with self._result_lock:
                self._target_name = resolved
                self._latest_detections = []
                self._result_revision = 0
            logger.info("%s process prompt set to %r", self.backend, resolved)
            return
        if self._qnn_runtime is not None:
            if self._future is not None:
                self._future.result()
                self._future = None
            if self._prompt_encoder is None:
                raise RuntimeError("QNN prompt encoder is not initialized")
            embedding = self._prompt_encoder.encode(resolved)
            with self._result_lock:
                self._target_embedding = embedding
                self._target_name = resolved
                self._latest_detections = []
                self._result_revision = 0
            logger.info("QNN prompt set to %r", resolved)
            return
        if self._prompt_model is None:
            if self._future is not None:
                self._future.result()
                self._future = None
            with self._result_lock:
                self._target_name = resolved
                self._latest_detections = []
                self._result_revision = 0
            return
        if self._future is not None:
            self._future.result()
            self._future = None
        self._prompt_model.set_classes([resolved])
        with self._result_lock:
            self._target_name = resolved
            self._latest_detections = []
            self._result_revision = 0
        logger.info("%s prompt set to %r", self.backend, resolved)

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
        if (
            self._prompt_model is not None
            or self._prompt_process
            or self._qnn_runtime
            or self._modal_client
        ):
            return self._detect_prompt_async(rgb, depth, camera_info)
        if self._net is not None:
            return self._detect_yolo(rgb, depth, camera_info)
        return []

    def _detect_prompt_async(
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
                if self._prompt_process:
                    raw_detections, elapsed = completed
                    self._record_prompt_inference(elapsed)
                    completed = self._project_worker_detections(
                        raw_detections,
                        self._future_depth,
                        self._future_camera_matrix,
                    )
            except Exception:
                logger.exception("%s inference failed", self.backend)
                completed = []
            with self._result_lock:
                self._latest_detections = completed
                self._result_revision += 1
            self._future = None
            self._future_depth = None
            self._future_camera_matrix = None
        if self._future is None:
            # The SDK may reuse camera buffers after this call. Take exactly one
            # contiguous snapshot for the asynchronous worker.
            rgb_copy = np.array(rgb, copy=True, order="C")
            if self._prompt_process:
                self._future_depth = (
                    None if depth is None else np.array(depth, copy=True, order="C")
                )
                self._future_camera_matrix = (
                    None
                    if camera_info is None
                    else np.asarray(camera_info.k, dtype=float).reshape(3, 3).copy()
                )
                self._future = self._executor.submit(
                    _prompt_worker_detect,
                    rgb_copy,
                    self.inference_size,
                    self.confidence_threshold,
                    self.max_detections,
                    self._target_name,
                )
            else:
                depth_copy = (
                    None if depth is None else np.array(depth, copy=True, order="C")
                )
                if self._qnn_runtime:
                    detector = self._detect_qnn
                elif self._modal_client:
                    detector = self._detect_modal
                else:
                    detector = self._detect_prompt
                self._future = self._executor.submit(
                    detector, rgb_copy, depth_copy, camera_info
                )
        with self._result_lock:
            return list(self._latest_detections)

    def _record_prompt_inference(self, elapsed: float) -> None:
        self._inference_count += 1
        smoothing = 0.2
        self._inference_seconds_ema = (
            elapsed
            if self._inference_seconds_ema is None
            else smoothing * elapsed
            + (1.0 - smoothing) * self._inference_seconds_ema
        )
        if self._inference_count == 1 or self._inference_count % 20 == 0:
            logger.info(
                "%s inference %.0fms (EMA %.0fms, %.2f FPS, imgsz=%d)",
                self.backend,
                elapsed * 1000.0,
                self._inference_seconds_ema * 1000.0,
                1.0 / self._inference_seconds_ema,
                self.inference_size,
            )

    def _project_worker_detections(
        self,
        detections: list[tuple[str, float, tuple[int, int, int, int]]],
        depth: np.ndarray | None,
        camera_matrix: np.ndarray | None,
    ) -> list[ObjectDetection]:
        projected = []
        for label, score, bbox in detections:
            position = (
                self._compute_3d_position(bbox, depth, camera_matrix)
                if depth is not None
                else None
            )
            projected.append(
                ObjectDetection(
                    label=label,
                    confidence=score,
                    bbox=bbox,
                    position_3d=position,
                    distance=(
                        float(np.linalg.norm(position)) if position is not None else 0.0
                    ),
                )
            )
        return projected

    def _detect_qnn(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Run the visual graph on QNN and project its boxes through live depth."""
        from nero.perception.qnn_yolo_world import (
            decode_yolo_world,
            preprocess_yolo_world,
        )

        if self._qnn_runtime is None or self._target_embedding is None:
            raise RuntimeError("QNN detector has no active runtime prompt")
        image, geometry = preprocess_yolo_world(rgb, self.inference_size)
        output, elapsed = self._qnn_runtime.infer(image, self._target_embedding)
        self._record_prompt_inference(elapsed)
        raw = [
            (self._target_name or "object", score, bbox)
            for score, bbox in decode_yolo_world(
                output,
                geometry,
                self.confidence_threshold,
                self.max_detections,
            )
        ]
        camera_matrix = (
            None
            if camera_info is None
            else np.asarray(camera_info.k, dtype=float).reshape(3, 3)
        )
        return self._project_worker_detections(raw, depth, camera_matrix)

    def _detect_modal(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Run remote 2D inference, then keep depth projection on the robot."""
        if self._modal_client is None or self._target_name is None:
            raise RuntimeError("Modal detector has no active endpoint target")
        started = time.perf_counter()
        detections, server_elapsed = self._modal_client.detect(
            rgb,
            self._target_name,
            self.confidence_threshold,
            self.inference_size,
            self.max_detections,
        )
        elapsed = time.perf_counter() - started
        self._record_prompt_inference(elapsed)
        logger.debug(
            "Modal detector round trip %.0fms (server inference %.0fms)",
            elapsed * 1000.0,
            server_elapsed * 1000.0,
        )
        raw = [
            (detection.label, detection.confidence, detection.bbox)
            for detection in detections
        ]
        camera_matrix = (
            None
            if camera_info is None
            else np.asarray(camera_info.k, dtype=float).reshape(3, 3)
        )
        return self._project_worker_detections(raw, depth, camera_matrix)

    def _detect_prompt(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        camera_info=None,
    ) -> list[ObjectDetection]:
        """Run one target-conditioned open-vocabulary inference."""
        started = time.perf_counter()
        results = self._prompt_model.predict(
            np.asarray(rgb),
            imgsz=self.inference_size,
            conf=self.confidence_threshold,
            device="cpu",
            max_det=self.max_detections,
            rect=True,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        self._inference_count += 1
        smoothing = 0.2
        self._inference_seconds_ema = (
            elapsed
            if self._inference_seconds_ema is None
            else smoothing * elapsed + (1.0 - smoothing) * self._inference_seconds_ema
        )
        if self._inference_count == 1 or self._inference_count % 20 == 0:
            logger.info(
                "%s inference %.0fms (EMA %.0fms, %.2f FPS, imgsz=%d)",
                self.backend,
                elapsed * 1000.0,
                self._inference_seconds_ema * 1000.0,
                1.0 / self._inference_seconds_ema,
                self.inference_size,
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
                            float(np.linalg.norm(position))
                            if position is not None
                            else 0.0
                        ),
                    )
                )
        return detections

    def close(self) -> None:
        """Stop the optional asynchronous inference worker."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._modal_client = None

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

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, 0.45)
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
                    distance=(
                        float(np.linalg.norm(position)) if position is not None else 0.0
                    ),
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
        target_lower = (self.resolve_target(target_name) or target_name).lower()
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
            matrix = np.asarray(
                getattr(camera_info, "k", camera_info), dtype=float
            ).reshape(3, 3)
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
