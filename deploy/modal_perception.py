"""Modal deployment for Nero's GPU-backed open-vocabulary perception path."""

from __future__ import annotations

import base64
import binascii
import time

import modal

APP_NAME = "nero-perception"
MODEL_PATH = "/models/yolov8s-worldv2.pt"
MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    "yolov8s-worldv2.pt"
)
MODEL_SHA256 = "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .uv_pip_install("fastapi[standard]", "ultralytics==8.3.228")
    .run_commands(
        "mkdir -p /models",
        f"curl --fail --location --retry 3 --output {MODEL_PATH} {MODEL_URL}",
        f"echo '{MODEL_SHA256}  {MODEL_PATH}' | sha256sum --check",
    )
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="L4",
    max_containers=1,
    scaledown_window=300,
    timeout=150,
)
class YoloWorldEndpoint:
    """Keep one YOLO-World model warm and serve authenticated JPEG requests."""

    @modal.enter()
    def load_model(self) -> None:
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(MODEL_PATH)
        self.target = None

    @modal.fastapi_endpoint(
        method="POST",
        docs=False,
        requires_proxy_auth=True,
    )
    def detect(self, request: dict) -> dict:
        import cv2
        import numpy as np
        from fastapi import HTTPException

        try:
            target = " ".join(str(request["target"]).split()).strip()
            confidence = float(request.get("confidence_threshold", 0.5))
            inference_size = int(request.get("inference_size", 256))
            max_detections = int(request.get("max_detections", 10))
            encoded = base64.b64decode(request["image_b64"], validate=True)
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail="invalid detector request") from exc
        if not target or frame is None:
            raise HTTPException(status_code=422, detail="target and JPEG are required")
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=422, detail="confidence must be in [0, 1]")
        if inference_size < 256 or inference_size % 32:
            raise HTTPException(
                status_code=422,
                detail="inference_size must be >= 256 and divisible by 32",
            )
        if max_detections < 1:
            raise HTTPException(status_code=422, detail="max_detections must be positive")

        if target != self.target:
            self.model.set_classes([target])
            self.target = target
        started = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=inference_size,
            conf=confidence,
            device=0,
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
            for bounds, score in zip(xyxy, scores):
                detections.append(
                    {
                        "label": target,
                        "confidence": float(score),
                        "bbox": [int(round(float(value))) for value in bounds],
                    }
                )
        return {"detections": detections, "inference_seconds": elapsed}
