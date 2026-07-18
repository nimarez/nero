"""Qualcomm QNN runtime for Nero's one-target YOLO-World graph."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


QNN_PROVIDER = "QNNExecutionProvider"


def _discover_qnn_backend() -> str | None:
    configured = os.getenv("NERO_QNN_BACKEND_PATH")
    if configured:
        return configured
    for root in (Path("/opt/qcom/qirp-sdk"), Path("/opt/qcom/aistack")):
        if not root.is_dir():
            continue
        candidates = sorted(root.glob("**/libQnnHtp.so"))
        native = [
            path
            for path in candidates
            if "aarch64" in str(path).lower() or "arm64" in str(path).lower()
        ]
        if native:
            return str(native[0])
        if len(candidates) == 1:
            return str(candidates[0])
    return None


@dataclass(frozen=True)
class LetterboxGeometry:
    """Geometry needed to map model boxes back into the camera image."""

    scale: float
    left: int
    top: int
    image_width: int
    image_height: int


def preprocess_yolo_world(
    image: np.ndarray, inference_size: int
) -> tuple[np.ndarray, LetterboxGeometry]:
    """Match Ultralytics' centered letterbox and BGR-to-RGB preprocessing."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got {source.shape}")
    height, width = source.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("camera image must not be empty")

    scale = min(inference_size / height, inference_size / width)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_width = inference_size - resized_width
    pad_height = inference_size - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)

    resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if canvas.shape[:2] != (inference_size, inference_size):
        raise RuntimeError(f"letterbox produced unexpected shape {canvas.shape}")
    tensor = canvas[..., ::-1].transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / np.float32(255.0)
    return tensor, LetterboxGeometry(scale, left, top, width, height)


def decode_yolo_world(
    output: np.ndarray,
    geometry: LetterboxGeometry,
    confidence_threshold: float,
    max_detections: int,
    iou_threshold: float = 0.45,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Decode one-class YOLO output and apply class-agnostic NMS."""
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise RuntimeError(f"expected batch size one, got {predictions.shape}")
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise RuntimeError(f"unexpected QNN YOLO output shape: {predictions.shape}")
    if predictions.shape[0] == 5:
        predictions = predictions.T
    elif predictions.shape[1] != 5:
        raise RuntimeError(f"unexpected QNN YOLO output shape: {predictions.shape}")

    candidates: list[tuple[float, list[int], tuple[int, int, int, int]]] = []
    for row in predictions:
        score = float(row[4])
        if not np.isfinite(score) or score < confidence_threshold:
            continue
        cx, cy, box_width, box_height = (float(value) for value in row[:4])
        model_box = (
            cx - box_width / 2,
            cy - box_height / 2,
            cx + box_width / 2,
            cy + box_height / 2,
        )
        x1 = int(round((model_box[0] - geometry.left) / geometry.scale))
        y1 = int(round((model_box[1] - geometry.top) / geometry.scale))
        x2 = int(round((model_box[2] - geometry.left) / geometry.scale))
        y2 = int(round((model_box[3] - geometry.top) / geometry.scale))
        x1 = min(max(x1, 0), geometry.image_width)
        x2 = min(max(x2, 0), geometry.image_width)
        y1 = min(max(y1, 0), geometry.image_height)
        y2 = min(max(y2, 0), geometry.image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((score, [x1, y1, x2 - x1, y2 - y1], (x1, y1, x2, y2)))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if not candidates:
        return []
    boxes = [candidate[1] for candidate in candidates]
    scores = [candidate[0] for candidate in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    selected = np.asarray(indices).reshape(-1) if len(indices) else np.empty(0, dtype=int)
    return [
        (candidates[int(index)][0], candidates[int(index)][2])
        for index in selected[:max_detections]
    ]


class YoloWorldPromptEncoder:
    """Generate exactly the normalized CLIP features used by YOLO-World."""

    def __init__(self) -> None:
        import torch
        from ultralytics.nn.text_model import build_text_model

        torch.set_num_threads(max(1, int(os.getenv("NERO_CLIP_THREADS", "2"))))
        self._model = build_text_model("clip:ViT-B/32", device=torch.device("cpu"))

    def encode(self, target: str) -> np.ndarray:
        tokens = self._model.tokenize([target])
        features = self._model.encode_text(tokens).detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32).reshape(1, 1, -1)
        if features.shape != (1, 1, 512):
            raise RuntimeError(f"unexpected YOLO-World text feature shape {features.shape}")
        if not np.isfinite(features).all():
            raise RuntimeError("YOLO-World text encoder returned non-finite values")
        return np.ascontiguousarray(features)


class QNNYoloWorldRuntime:
    """Fail-closed ONNX Runtime session whose entire graph must run on QNN HTP."""

    def __init__(self, model_path: str | Path, inference_size: int = 256) -> None:
        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        if QNN_PROVIDER not in available:
            raise RuntimeError(
                "ONNX Runtime does not expose QNNExecutionProvider "
                f"(available: {', '.join(available) or 'none'})"
            )

        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options: dict[str, str] = {
            "htp_performance_mode": os.getenv("NERO_QNN_PERFORMANCE_MODE", "burst"),
            "htp_graph_finalization_optimization_mode": os.getenv(
                "NERO_QNN_GRAPH_OPTIMIZATION", "3"
            ),
            "enable_htp_fp16_precision": "1",
            # Keep graph I/O conversion in QNN so the strict no-CPU contract is literal.
            "offload_graph_io_quantization": "0",
        }
        backend_path = _discover_qnn_backend()
        if backend_path:
            provider_options["backend_path"] = backend_path
        else:
            provider_options["backend_type"] = "htp"

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[QNN_PROVIDER],
            provider_options=[provider_options],
        )
        disable_fallback = getattr(self._session, "disable_fallback", None)
        if disable_fallback is not None:
            disable_fallback()
        active = tuple(self._session.get_providers())
        if not active or active[0] != QNN_PROVIDER:
            raise RuntimeError(f"QNN provider was not activated (active: {active})")

        inputs = {item.name: tuple(item.shape) for item in self._session.get_inputs()}
        expected = {
            "images": (1, 3, inference_size, inference_size),
            "text_features": (1, 1, 512),
        }
        if inputs != expected:
            raise RuntimeError(f"unexpected QNN model inputs {inputs}; expected {expected}")
        outputs = self._session.get_outputs()
        if len(outputs) != 1 or tuple(outputs[0].shape) != (1, 5, 1344):
            raise RuntimeError(
                "unexpected QNN model output "
                f"{[(item.name, tuple(item.shape)) for item in outputs]}"
            )

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self._session.get_providers())

    def infer(self, images: np.ndarray, text_features: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        output = self._session.run(
            None,
            {"images": images, "text_features": text_features},
        )[0]
        return np.asarray(output), time.perf_counter() - started
