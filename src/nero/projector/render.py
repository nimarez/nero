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
    ring_uv: list[list[float]] | list[tuple[float, float]] | None = None,
    label: str = "CONTROLLER",
) -> np.ndarray:
    """Warp a real floor-space circle through the surface homography."""

    canvas = base_frame.copy()
    center_uv = (float(floor_uv[0]), float(floor_uv[1]))
    if ring_uv is None:
        angles = np.linspace(0.0, np.pi * 2.0, 96, endpoint=False)
        ring_uv = [
            (center_uv[0] + 0.093 * np.cos(angle), center_uv[1] + 0.127 * np.sin(angle))
            for angle in angles
        ]
    ring_floor = np.asarray(ring_uv, dtype=np.float32)
    ring = np.rint(calibration.transform(ring_floor)).astype(np.int32).reshape(-1, 1, 2)
    inner_floor = np.asarray(
        [
            (
                center_uv[0] + (point[0] - center_uv[0]) * 0.77,
                center_uv[1] + (point[1] - center_uv[1]) * 0.77,
            )
            for point in ring_floor
        ],
        dtype=np.float32,
    )
    inner = np.rint(calibration.transform(inner_floor)).astype(np.int32).reshape(-1, 1, 2)
    center = tuple(np.rint(calibration.transform((center_uv,))[0]).astype(int))
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [ring], (0, 92, 170), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0.0, canvas)
    cv2.polylines(canvas, [ring], True, (0, 164, 255), 16, cv2.LINE_AA)
    cv2.polylines(canvas, [inner], True, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawMarker(canvas, center, (118, 255, 148), cv2.MARKER_CROSS, 72, 8, cv2.LINE_AA)
    top = int(ring[:, 0, 1].min())
    left = int(ring[:, 0, 0].min())
    cv2.putText(
        canvas,
        label,
        (left, top - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return canvas


def render_navigation_overlay(
    base_frame: np.ndarray,
    calibration: ProjectorCalibration,
    *,
    robot_frame_uv: dict | None,
    trajectory_uv: list[list[float]] | None,
    goal_uv: list[float] | None,
    goal_heading_uv: list[float] | None,
    animation_phase: float = 0.0,
) -> np.ndarray:
    """Render robot-local axes and a Waymo-style floor trajectory preview."""

    canvas = base_frame.copy()
    if trajectory_uv and len(trajectory_uv) >= 2:
        floor_path = np.asarray(trajectory_uv, dtype=np.float32)
        path = np.rint(calibration.transform(floor_path)).astype(np.int32).reshape(-1, 1, 2)
        overlay = canvas.copy()
        cv2.polylines(overlay, [path], False, (120, 54, 0), 54, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0.0, canvas)
        cv2.polylines(canvas, [path], False, (255, 205, 82), 13, cv2.LINE_AA)
        cv2.polylines(canvas, [path], False, (255, 255, 255), 3, cv2.LINE_AA)

        pixels = path[:, 0, :].astype(np.float64)
        for start, end in zip(pixels, pixels[1:]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1.0:
                continue
            direction = delta / length
            normal = np.asarray((-direction[1], direction[0]))
            spacing = 92.0
            distance = (animation_phase % 1.0) * spacing
            while distance < length:
                tip = start + direction * distance
                back = tip - direction * 26.0
                chevron = np.rint(
                    np.asarray((back + normal * 15.0, tip, back - normal * 15.0))
                ).astype(np.int32)
                cv2.polylines(
                    canvas, [chevron.reshape(-1, 1, 2)], False, (255, 255, 255), 5, cv2.LINE_AA
                )
                distance += spacing

    if goal_uv:
        center = tuple(np.rint(calibration.transform((goal_uv,))[0]).astype(int))
        cv2.circle(canvas, center, 42, (255, 205, 82), 8, cv2.LINE_AA)
        cv2.circle(canvas, center, 15, (255, 255, 255), -1, cv2.LINE_AA)
        if goal_heading_uv:
            tip = tuple(np.rint(calibration.transform((goal_heading_uv,))[0]).astype(int))
            cv2.arrowedLine(canvas, center, tip, (255, 205, 82), 9, cv2.LINE_AA, tipLength=0.28)
        cv2.putText(
            canvas,
            "GOAL",
            (center[0] + 52, center[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    if robot_frame_uv:
        for line in robot_frame_uv.get("grid_lines", []):
            points = np.rint(calibration.transform(line)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], False, (72, 116, 84), 3, cv2.LINE_AA)
        footprint = robot_frame_uv.get("footprint")
        if footprint:
            points = np.rint(calibration.transform(footprint)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, (255, 255, 255), 6, cv2.LINE_AA)
        x_axis = robot_frame_uv.get("x_axis")
        if x_axis:
            start, end = np.rint(calibration.transform(x_axis)).astype(np.int32)
            cv2.arrowedLine(
                canvas, tuple(start), tuple(end), (0, 142, 255), 11, cv2.LINE_AA, tipLength=0.24
            )
            cv2.putText(
                canvas,
                "+X / FORWARD",
                tuple(end + np.asarray((18, -14))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        y_axis = robot_frame_uv.get("y_axis")
        if y_axis:
            start, end = np.rint(calibration.transform(y_axis)).astype(np.int32)
            cv2.arrowedLine(
                canvas, tuple(start), tuple(end), (255, 220, 90), 8, cv2.LINE_AA, tipLength=0.24
            )
            cv2.putText(
                canvas,
                "+Y",
                tuple(end + np.asarray((14, -10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def render_floor_calibration_target(
    base_frame: np.ndarray,
    calibration: ProjectorCalibration,
    floor_uv: tuple[float, float] | list[float],
    *,
    index: int,
    total: int,
) -> np.ndarray:
    """Draw the fixed target where the operator should place the controller."""

    canvas = base_frame.copy()
    u, v = float(floor_uv[0]), float(floor_uv[1])
    angles = np.linspace(0.0, np.pi * 2.0, 80, endpoint=False)
    floor_ring = np.column_stack((u + 0.07 * np.cos(angles), v + 0.095 * np.sin(angles)))
    ring = np.rint(calibration.transform(floor_ring)).astype(np.int32).reshape(-1, 1, 2)
    center = tuple(np.rint(calibration.transform(((u, v),))[0]).astype(int))
    cv2.polylines(canvas, [ring], True, (255, 255, 0), 13, cv2.LINE_AA)
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 100, 12, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"PLACE CONTROLLER HERE  {index}/{total}",
        (center[0] - 230, center[1] - 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return canvas
