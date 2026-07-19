"""HTTP client for Nero's Modal-hosted YOLO-World detector."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ModalDetection:
    """One image-space detection returned by the Modal endpoint."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class ModalYoloWorldClient:
    """Send compressed camera frames to a proxy-authenticated Modal endpoint."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        secret: str | None = None,
        timeout: float | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        self.url = (url or os.getenv("NERO_MODAL_URL", "")).strip()
        self.key = (key or os.getenv("NERO_MODAL_KEY", "")).strip()
        self.secret = (secret or os.getenv("NERO_MODAL_SECRET", "")).strip()
        self.timeout = float(
            timeout if timeout is not None else os.getenv("NERO_MODAL_TIMEOUT", "120")
        )
        self.jpeg_quality = int(
            jpeg_quality
            if jpeg_quality is not None
            else os.getenv("NERO_MODAL_JPEG_QUALITY", "85")
        )
        if not self.url:
            raise ValueError("NERO_MODAL_URL is required for the Modal detector")
        if not self.url.startswith("https://") and os.getenv(
            "NERO_MODAL_ALLOW_HTTP", "0"
        ) != "1":
            raise ValueError("NERO_MODAL_URL must use HTTPS")
        if not self.key or not self.secret:
            raise ValueError(
                "NERO_MODAL_KEY and NERO_MODAL_SECRET are required for proxy auth"
            )
        if self.timeout <= 0:
            raise ValueError("NERO_MODAL_TIMEOUT must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("NERO_MODAL_JPEG_QUALITY must be between 1 and 100")

    def detect(
        self,
        image: np.ndarray,
        target: str,
        confidence_threshold: float,
        inference_size: int,
        max_detections: int,
    ) -> tuple[list[ModalDetection], float]:
        """Run one remote inference and return detections plus server model time."""
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Modal detector image must have shape (H, W, 3)")
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("could not JPEG-encode detector frame")
        body = json.dumps(
            {
                "image_b64": base64.b64encode(encoded).decode("ascii"),
                "target": target,
                "confidence_threshold": float(confidence_threshold),
                "inference_size": int(inference_size),
                "max_detections": int(max_detections),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Modal-Key": self.key,
                "Modal-Secret": self.secret,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Modal detector returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Modal detector request failed: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("detections"), list
        ):
            raise RuntimeError("Modal detector returned an invalid response")

        detections = []
        for item in payload["detections"]:
            try:
                bbox_values = item["bbox"]
                if len(bbox_values) != 4:
                    raise ValueError
                detection = ModalDetection(
                    label=str(item["label"]),
                    confidence=float(item["confidence"]),
                    bbox=tuple(int(value) for value in bbox_values),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Modal detector returned an invalid detection") from exc
            detections.append(detection)
        return detections, float(payload.get("inference_seconds", 0.0))
