"""Shared construction for navigation object-detector backends."""

from __future__ import annotations

import os
from pathlib import Path

from .aruco_detector import ArucoObjectDetector
from .object_detector import ObjectDetector


def create_object_detector(
    *,
    backend: str | None = None,
    aruco_map: str | Path | None = None,
    aruco_dictionary: str | None = None,
):
    """Build a neural or ArUco detector from shared CLI/environment options."""
    resolved_backend = backend or os.getenv("NERO_OBJECT_BACKEND")
    normalized = (
        resolved_backend.strip().lower().replace("_", "-")
        if resolved_backend
        else None
    )
    if normalized == "aruco":
        return ArucoObjectDetector(
            mapping_path=aruco_map,
            dictionary=aruco_dictionary,
        )
    if aruco_map is not None:
        raise ValueError("--aruco-map requires --object-backend aruco")
    return ObjectDetector(backend=resolved_backend)
