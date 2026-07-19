"""Pure OpenCV rendering for the projector calibration surface."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import ProjectorCalibration


@dataclass(frozen=True, slots=True)
class GridStyle:
    """High-contrast palette tuned for projection onto a black floor surface."""

    minor_green: tuple[int, int, int] = (88, 178, 104)
    major_green: tuple[int, int, int] = (118, 255, 148)
    boundary_green: tuple[int, int, int] = (92, 255, 126)
    white: tuple[int, int, int] = (255, 255, 255)
    orange: tuple[int, int, int] = (0, 142, 255)
    corner_ring: tuple[int, int, int] = (88, 255, 150)


def _line_points(
    calibration: ProjectorCalibration,
    *,
    vertical: bool,
    coordinate: float,
    samples: int = 96,
) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    if vertical:
        source = np.column_stack((np.full_like(axis, coordinate), axis))
    else:
        source = np.column_stack((axis, np.full_like(axis, coordinate)))
    return np.rint(calibration.transform(source)).astype(np.int32).reshape(-1, 1, 2)


def render_projector_grid(
    calibration: ProjectorCalibration,
    style: GridStyle | None = None,
) -> np.ndarray:
    """Render a keystone-corrected grid into projector pixel space."""

    style = style or GridStyle()
    canvas = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
    divisions = calibration.grid_divisions

    for index in range(divisions + 1):
        coordinate = index / divisions
        boundary = index in (0, divisions)
        major = boundary or index % max(1, divisions // 4) == 0
        color = (
            style.boundary_green
            if boundary
            else style.major_green
            if major
            else style.minor_green
        )
        thickness = calibration.line_thickness + (2 if boundary else 1 if major else 0)
        cv2.polylines(
            canvas,
            [_line_points(calibration, vertical=True, coordinate=coordinate)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.polylines(
            canvas,
            [_line_points(calibration, vertical=False, coordinate=coordinate)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )

    for marker_id, point in zip(calibration.marker_ids, calibration.handles):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, center, 27, style.white, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 39, style.corner_ring, 7, cv2.LINE_AA)
        label_origin = (center[0] + 44, center[1] - 18)
        cv2.putText(
            canvas,
            f"ID {marker_id}",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            style.white,
            2,
            cv2.LINE_AA,
        )

    center = calibration.transform(((0.5, 0.5),))[0]
    center_px = tuple(np.rint(center).astype(int))
    cv2.circle(canvas, center_px, 22, style.orange, 5, cv2.LINE_AA)
    cv2.drawMarker(
        canvas,
        center_px,
        style.orange,
        cv2.MARKER_CROSS,
        86,
        7,
        cv2.LINE_AA,
    )
    return canvas


def render_motion_circle(
    base_frame: np.ndarray,
    calibration: ProjectorCalibration,
    floor_uv: tuple[float, float] | list[float],
    *,
    label: str = "CONTROLLER",
    radius: int = 118,
) -> np.ndarray:
    """Overlay a large, legible controller target on a cached grid frame."""

    canvas = base_frame.copy()
    point = calibration.transform(((float(floor_uv[0]), float(floor_uv[1])),))[0]
    center = tuple(np.rint(point).astype(int))
    overlay = canvas.copy()
    cv2.circle(overlay, center, radius, (0, 92, 170), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0.0, canvas)
    cv2.circle(canvas, center, radius, (0, 164, 255), 16, cv2.LINE_AA)
    cv2.circle(canvas, center, radius - 24, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawMarker(canvas, center, (118, 255, 148), cv2.MARKER_CROSS, 72, 8, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (center[0] - radius, center[1] - radius - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return canvas
