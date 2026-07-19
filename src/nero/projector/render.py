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


def _scaled_color(color: tuple[int, int, int], intensity: float) -> tuple[int, int, int]:
    return tuple(int(round(component * intensity)) for component in color)


def _hex_bgr(color: str) -> tuple[int, int, int]:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return blue, green, red


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
    mission_mode = calibration.visualization_mode == "mission"
    intensity = calibration.mission_grid_intensity if mission_mode else 1.0

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
        color = _scaled_color(color, intensity)
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

    if not mission_mode:
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


def _resample_path(points: np.ndarray, count: int = 96) -> np.ndarray:
    if len(points) < 2:
        return points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] < 1e-6:
        return points[:1]
    distances = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        (
            np.interp(distances, cumulative, points[:, 0]),
            np.interp(distances, cumulative, points[:, 1]),
        )
    )


def _trim_path_before_goal(points: np.ndarray, gap: float) -> np.ndarray:
    if len(points) < 2:
        return points
    kept = [point.copy() for point in points]
    remaining = gap
    while len(kept) >= 2 and remaining > 0:
        segment = kept[-1] - kept[-2]
        length = float(np.linalg.norm(segment))
        if length <= remaining:
            remaining -= length
            kept.pop()
            continue
        kept[-1] = kept[-1] - segment / length * remaining
        remaining = 0.0
    return np.asarray(kept, dtype=np.float64)


def _trim_path_after_start(points: np.ndarray, gap: float) -> np.ndarray:
    if len(points) < 2:
        return points
    return _trim_path_before_goal(points[::-1], gap)[::-1]


def _path_normals(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(samples, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(length, 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return tangent, normal


def _path_polygon(samples: np.ndarray, widths: np.ndarray) -> np.ndarray:
    _, normal = _path_normals(samples)
    left = samples + normal * widths[:, None]
    right = samples - normal * widths[:, None]
    return np.vstack((left, right[::-1]))


def _floor_pixels(calibration: ProjectorCalibration, points: np.ndarray) -> np.ndarray:
    return np.rint(calibration.transform(points)).astype(np.int32).reshape(-1, 1, 2)


def _blend_floor_polygon(
    canvas: np.ndarray,
    calibration: ProjectorCalibration,
    points: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if len(points) < 3 or alpha <= 0:
        return
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [_floor_pixels(calibration, points)], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, min(1.0, alpha), canvas, 1.0 - min(1.0, alpha), 0.0, canvas)


def _floor_ring(center: np.ndarray, radius: float, samples: int = 96) -> np.ndarray:
    angle = np.linspace(0.0, np.pi * 2.0, samples, endpoint=False)
    return np.column_stack((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)))


def _signal_drop(
    center: np.ndarray, tangent: np.ndarray, normal: np.ndarray, scale: float
) -> np.ndarray:
    return np.asarray(
        (
            center + tangent * scale * 1.05,
            center + tangent * scale * 0.10 + normal * scale * 0.48,
            center - tangent * scale * 0.92,
            center + tangent * scale * 0.10 - normal * scale * 0.48,
        )
    )


def _render_target_zone(
    canvas: np.ndarray,
    calibration: ProjectorCalibration,
    goal_uv: list[float],
    *,
    color: tuple[int, int, int],
    radius: float,
    glow: float,
    animation_phase: float,
    complete: bool = False,
) -> None:
    center_floor = np.asarray(goal_uv, dtype=np.float64)
    edge_space = min(
        center_floor[0], 1.0 - center_floor[0], center_floor[1], 1.0 - center_floor[1]
    )
    radius = min(radius, max(0.025, float(edge_space) / 1.62))
    pulse = 1.0 + 0.055 * np.sin(animation_phase * np.pi * 2.0)
    outer = _floor_ring(center_floor, radius * (1.72 if complete else 1.52) * pulse)
    middle = _floor_ring(center_floor, radius * 1.10)
    inner = _floor_ring(center_floor, radius * 0.48)
    _blend_floor_polygon(canvas, calibration, outer, _scaled_color(color, 0.42), 0.16 * glow)
    _blend_floor_polygon(canvas, calibration, middle, color, 0.18 * glow)
    cv2.polylines(canvas, [_floor_pixels(calibration, outer)], True, color, 8 if complete else 5, cv2.LINE_AA)
    cv2.polylines(canvas, [_floor_pixels(calibration, middle)], True, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.polylines(canvas, [_floor_pixels(calibration, inner)], True, color, 4, cv2.LINE_AA)
    center = tuple(np.rint(calibration.transform((center_floor,))[0]).astype(int))
    cv2.circle(canvas, center, 10, (255, 255, 255), -1, cv2.LINE_AA)
    if complete:
        check = np.asarray(
            (
                center_floor + (-radius * 0.42, 0.0),
                center_floor + (-radius * 0.10, radius * 0.32),
                center_floor + (radius * 0.48, -radius * 0.38),
            ),
            dtype=np.float64,
        )
        cv2.polylines(
            canvas,
            [_floor_pixels(calibration, check)],
            False,
            (255, 255, 255),
            13,
            cv2.LINE_AA,
        )
        ray_angle = np.linspace(0.0, np.pi * 2.0, 12, endpoint=False)
        for angle in ray_angle:
            ray = np.asarray(
                (
                    center_floor
                    + radius * 1.82 * np.asarray((np.cos(angle), np.sin(angle))),
                    center_floor
                    + radius * 2.18 * np.asarray((np.cos(angle), np.sin(angle))),
                )
            )
            cv2.polylines(
                canvas, [_floor_pixels(calibration, ray)], False, (255, 255, 255), 5, cv2.LINE_AA
            )
    label = "PICKUP READY" if complete else "TARGET"
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.92, 3
    )
    label_x = center[0] + 58
    if label_x + label_width > calibration.width - 18:
        label_x = center[0] - label_width - 58
    label_y = min(calibration.height - 18, max(label_height + 18, center[1] - 42))
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def _render_robot_work_zone(
    canvas: np.ndarray,
    calibration: ProjectorCalibration,
    start: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    route_width: float,
) -> None:
    red = (28, 42, 255)
    bar_center = start - tangent * route_width * 0.36
    half_width = route_width * 1.32
    bar_floor = np.asarray((bar_center - normal * half_width, bar_center + normal * half_width))
    bar = _floor_pixels(calibration, bar_floor)
    cv2.polylines(canvas, [bar], False, (8, 12, 92), 30, cv2.LINE_AA)
    cv2.polylines(canvas, [bar], False, red, 15, cv2.LINE_AA)
    cv2.polylines(canvas, [bar], False, (255, 255, 255), 3, cv2.LINE_AA)
    center = tuple(np.rint(calibration.transform((bar_center,))[0]).astype(int))
    label = "ROBOT WORKING"
    font_scale = 1.08
    thickness = 4
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_x = min(calibration.width - label_width - 18, max(18, center[0] - label_width // 2))
    label_y = center[1] + 74
    if label_y > calibration.height - 18:
        label_y = center[1] - 58
    label_y = min(calibration.height - 18, max(label_height + 18, label_y))
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        red,
        11,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _render_mission_path(
    canvas: np.ndarray,
    calibration: ProjectorCalibration,
    trajectory_uv: list[list[float]] | None,
    goal_uv: list[float] | None,
    *,
    animation_phase: float,
    distance_to_goal_m: float | None,
) -> None:
    route_color = _hex_bgr(calibration.mission_route_color)
    target_color = _hex_bgr(calibration.mission_target_color)
    glow = calibration.mission_glow_strength
    samples = np.empty((0, 2), dtype=np.float64)
    robot_start = None
    robot_tangent = None
    robot_normal = None
    if trajectory_uv and len(trajectory_uv) >= 2:
        path = np.asarray(trajectory_uv, dtype=np.float64)
        robot_start = path[0].copy()
        initial_direction = path[1] - path[0]
        initial_direction /= max(float(np.linalg.norm(initial_direction)), 1e-6)
        robot_tangent = initial_direction
        robot_normal = np.asarray((-initial_direction[1], initial_direction[0]))
        path = _trim_path_before_goal(path, calibration.mission_goal_gap)
        path = _trim_path_after_start(path, calibration.mission_route_width * 0.72)
        samples = _resample_path(path)
    if len(samples) >= 2:
        progress = np.linspace(0.0, 1.0, len(samples))
        tangent, normal = _path_normals(samples)
        preset = calibration.mission_preset
        if preset == "intent_field":
            widths = calibration.mission_route_width * (0.96 - 0.82 * progress**1.35)
            outer = _path_polygon(samples, widths * 1.58)
            field = _path_polygon(samples, widths)
            core = _path_polygon(samples, widths * 0.43)
            _blend_floor_polygon(
                canvas, calibration, outer, _scaled_color(route_color, 0.38), 0.18 * glow
            )
            _blend_floor_polygon(canvas, calibration, field, route_color, 0.34 * glow)
            _blend_floor_polygon(canvas, calibration, core, (255, 255, 255), 0.30 * glow)
            cv2.polylines(
                canvas,
                [_floor_pixels(calibration, samples)],
                False,
                route_color,
                6,
                cv2.LINE_AA,
            )
        elif preset == "safety_corridor":
            widths = np.full(len(samples), calibration.mission_route_width)
            polygon = _path_polygon(samples, widths * 1.18)
            _blend_floor_polygon(
                canvas, calibration, polygon, _scaled_color(route_color, 0.45), 0.16 * glow
            )
            left = samples + normal * widths[:, None]
            right = samples - normal * widths[:, None]
            for edge in (left, right):
                cv2.polylines(
                    canvas,
                    [_floor_pixels(calibration, edge)],
                    False,
                    route_color,
                    10,
                    cv2.LINE_AA,
                )
                cv2.polylines(
                    canvas,
                    [_floor_pixels(calibration, edge)],
                    False,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        signal_count = 7 if preset == "beacon_trail" else 4
        for index in range(signal_count):
            position = (animation_phase * 0.16 + index / signal_count) % 1.0
            sample_index = min(len(samples) - 1, int(position * (len(samples) - 1)))
            scale = calibration.mission_route_width * (0.62 if preset == "beacon_trail" else 0.34)
            drop = _signal_drop(samples[sample_index], tangent[sample_index], normal[sample_index], scale)
            if preset == "beacon_trail":
                halo = _signal_drop(
                    samples[sample_index], tangent[sample_index], normal[sample_index], scale * 1.65
                )
                _blend_floor_polygon(
                    canvas, calibration, halo, _scaled_color(route_color, 0.42), 0.22 * glow
                )
            _blend_floor_polygon(canvas, calibration, drop, route_color, 0.92)
            inner = samples[sample_index] + tangent[sample_index] * scale * 0.18
            inner_ring = _floor_ring(inner, scale * 0.17, 24)
            _blend_floor_polygon(canvas, calibration, inner_ring, (255, 255, 255), 0.95)

        if robot_start is not None and robot_tangent is not None and robot_normal is not None:
            _render_robot_work_zone(
                canvas,
                calibration,
                robot_start,
                robot_tangent,
                robot_normal,
                calibration.mission_route_width,
            )

    if goal_uv:
        _render_target_zone(
            canvas,
            calibration,
            goal_uv,
            color=target_color,
            radius=calibration.mission_target_radius,
            glow=glow,
            animation_phase=animation_phase,
            complete=distance_to_goal_m is not None and distance_to_goal_m <= 0.32,
        )


def _render_engineering_overlay(
    canvas: np.ndarray,
    calibration: ProjectorCalibration,
    *,
    robot_frame_uv: dict | None,
    trajectory_uv: list[list[float]] | None,
    goal_uv: list[float] | None,
    goal_heading_uv: list[float] | None,
    animation_phase: float,
) -> None:
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


def render_navigation_overlay(
    base_frame: np.ndarray,
    calibration: ProjectorCalibration,
    *,
    robot_frame_uv: dict | None,
    trajectory_uv: list[list[float]] | None,
    goal_uv: list[float] | None,
    goal_heading_uv: list[float] | None,
    animation_phase: float = 0.0,
    distance_to_goal_m: float | None = None,
) -> np.ndarray:
    """Render either the preserved engineering view or the factory mission view."""

    canvas = base_frame.copy()
    if calibration.visualization_mode == "mission":
        _render_mission_path(
            canvas,
            calibration,
            trajectory_uv,
            goal_uv,
            animation_phase=animation_phase,
            distance_to_goal_m=distance_to_goal_m,
        )
    else:
        _render_engineering_overlay(
            canvas,
            calibration,
            robot_frame_uv=robot_frame_uv,
            trajectory_uv=trajectory_uv,
            goal_uv=goal_uv,
            goal_heading_uv=goal_heading_uv,
            animation_phase=animation_phase,
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
