"""Durable projector calibration state and homography helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import cv2
import numpy as np

DEFAULT_MARKER_IDS = (1, 2, 3, 4)
DEFAULT_MARKER_SIZE_M = 0.130
DEFAULT_PROJECTOR_SIZE = (1920, 1080)


def _default_handles() -> tuple[tuple[float, float], ...]:
    return ((360.0, 220.0), (1560.0, 220.0), (1560.0, 860.0), (360.0, 860.0))


def _finite_point(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be an [x, y] pair")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{field_name} must contain finite numbers")
    return point


def _hex_color(value: Any, field_name: str) -> str:
    color = str(value).upper()
    if len(color) != 7 or color[0] != "#" or any(
        character not in "0123456789ABCDEF" for character in color[1:]
    ):
        raise ValueError(f"{field_name} must be a #RRGGBB color")
    return color


@dataclass(frozen=True, slots=True)
class ProjectorCalibration:
    """Four projector pixels corresponding to marker centers 1, 2, 3, 4.

    Marker order is clockwise around the floor patch. The browser allows the
    operator to move each handle; the rendered grid uses the same normalized
    floor frame for every update.
    """

    width: int = DEFAULT_PROJECTOR_SIZE[0]
    height: int = DEFAULT_PROJECTOR_SIZE[1]
    marker_dictionary: str = "DICT_4X4_50"
    marker_ids: tuple[int, int, int, int] = DEFAULT_MARKER_IDS
    marker_size_m: float = DEFAULT_MARKER_SIZE_M
    handles: tuple[tuple[float, float], ...] = field(default_factory=_default_handles)
    grid_divisions: int = 12
    line_thickness: int = 2
    visualization_mode: str = "engineering"
    mission_preset: str = "intent_field"
    mission_route_color: str = "#8CFFD0"
    mission_target_color: str = "#FFD34D"
    mission_route_width: float = 0.070
    mission_goal_gap: float = 0.085
    mission_target_radius: float = 0.065
    mission_glow_strength: float = 0.90
    mission_animation_speed: float = 0.80
    mission_grid_intensity: float = 0.55

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("projector dimensions must be positive")
        if len(self.marker_ids) != 4 or len(set(self.marker_ids)) != 4:
            raise ValueError("exactly four unique marker IDs are required")
        if tuple(self.marker_ids) != DEFAULT_MARKER_IDS:
            raise ValueError("this calibration surface expects marker IDs 1, 2, 3, 4")
        if not math.isfinite(self.marker_size_m) or self.marker_size_m <= 0:
            raise ValueError("marker_size_m must be positive")
        if len(self.handles) != 4:
            raise ValueError("exactly four projector handles are required")
        for index, point in enumerate(self.handles):
            x, y = _finite_point(point, f"handles[{index}]")
            if not (-self.width <= x <= self.width * 2):
                raise ValueError(f"handles[{index}].x is outside the safe editing range")
            if not (-self.height <= y <= self.height * 2):
                raise ValueError(f"handles[{index}].y is outside the safe editing range")
        polygon = np.asarray(self.handles, dtype=np.float32)
        if abs(float(cv2.contourArea(polygon))) < 100.0:
            raise ValueError("projector handles form a degenerate quadrilateral")
        if not 4 <= self.grid_divisions <= 40:
            raise ValueError("grid_divisions must be between 4 and 40")
        if not 1 <= self.line_thickness <= 12:
            raise ValueError("line_thickness must be between 1 and 12")
        if self.visualization_mode not in {"engineering", "mission"}:
            raise ValueError("visualization_mode must be engineering or mission")
        if self.mission_preset not in {"intent_field", "beacon_trail", "safety_corridor"}:
            raise ValueError("unsupported mission_preset")
        _hex_color(self.mission_route_color, "mission_route_color")
        _hex_color(self.mission_target_color, "mission_target_color")
        for name, value, lower, upper in (
            ("mission_route_width", self.mission_route_width, 0.025, 0.140),
            ("mission_goal_gap", self.mission_goal_gap, 0.020, 0.220),
            ("mission_target_radius", self.mission_target_radius, 0.030, 0.140),
            ("mission_glow_strength", self.mission_glow_strength, 0.10, 1.00),
            ("mission_animation_speed", self.mission_animation_speed, 0.10, 2.50),
            ("mission_grid_intensity", self.mission_grid_intensity, 0.15, 1.00),
        ):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")

    @property
    def homography(self) -> np.ndarray:
        floor = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        projector = np.asarray(self.handles, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(floor, projector)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("projector homography contains non-finite values")
        return matrix

    def transform(self, points: Iterable[tuple[float, float]]) -> np.ndarray:
        source = np.asarray(tuple(points), dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(source, self.homography).reshape(-1, 2)

    def with_handles(self, handles: Any) -> "ProjectorCalibration":
        parsed = tuple(_finite_point(point, f"handles[{index}]") for index, point in enumerate(handles))
        return replace(self, handles=parsed)

    def with_style(self, *, grid_divisions: Any, line_thickness: Any) -> "ProjectorCalibration":
        return replace(
            self,
            grid_divisions=int(grid_divisions),
            line_thickness=int(line_thickness),
        )

    def with_visualization(self, payload: Any) -> "ProjectorCalibration":
        if not isinstance(payload, dict):
            raise ValueError("visualization settings must be an object")
        current = self.visualization_dict()
        unknown = set(payload) - set(current)
        if unknown:
            raise ValueError(f"unsupported visualization setting: {sorted(unknown)[0]}")
        current.update(payload)
        return replace(
            self,
            visualization_mode=str(current["mode"]),
            mission_preset=str(current["preset"]),
            mission_route_color=_hex_color(current["route_color"], "route_color"),
            mission_target_color=_hex_color(current["target_color"], "target_color"),
            mission_route_width=float(current["route_width"]),
            mission_goal_gap=float(current["goal_gap"]),
            mission_target_radius=float(current["target_radius"]),
            mission_glow_strength=float(current["glow_strength"]),
            mission_animation_speed=float(current["animation_speed"]),
            mission_grid_intensity=float(current["grid_intensity"]),
        )

    def visualization_dict(self) -> dict[str, Any]:
        return {
            "mode": self.visualization_mode,
            "preset": self.mission_preset,
            "route_color": self.mission_route_color,
            "target_color": self.mission_target_color,
            "route_width": self.mission_route_width,
            "goal_gap": self.mission_goal_gap,
            "target_radius": self.mission_target_radius,
            "glow_strength": self.mission_glow_strength,
            "animation_speed": self.mission_animation_speed,
            "grid_intensity": self.mission_grid_intensity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "projector": {"width": self.width, "height": self.height},
            "markers": {
                "dictionary": self.marker_dictionary,
                "ids": list(self.marker_ids),
                "size_m": self.marker_size_m,
            },
            "handles": [list(point) for point in self.handles],
            "style": {
                "grid_divisions": self.grid_divisions,
                "line_thickness": self.line_thickness,
            },
            "visualization": self.visualization_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ProjectorCalibration":
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported projector calibration format")
        projector = payload.get("projector") or {}
        markers = payload.get("markers") or {}
        style = payload.get("style") or {}
        visualization = payload.get("visualization") or {}
        return cls(
            width=int(projector.get("width", DEFAULT_PROJECTOR_SIZE[0])),
            height=int(projector.get("height", DEFAULT_PROJECTOR_SIZE[1])),
            marker_dictionary=str(markers.get("dictionary", "DICT_4X4_50")),
            marker_ids=tuple(int(value) for value in markers.get("ids", DEFAULT_MARKER_IDS)),
            marker_size_m=float(markers.get("size_m", DEFAULT_MARKER_SIZE_M)),
            handles=tuple(
                _finite_point(point, f"handles[{index}]")
                for index, point in enumerate(payload.get("handles", _default_handles()))
            ),
            grid_divisions=int(style.get("grid_divisions", 12)),
            line_thickness=int(style.get("line_thickness", 2)),
            visualization_mode=str(visualization.get("mode", "engineering")),
            mission_preset=str(visualization.get("preset", "intent_field")),
            mission_route_color=_hex_color(
                visualization.get("route_color", "#8CFFD0"), "route_color"
            ),
            mission_target_color=_hex_color(
                visualization.get("target_color", "#FFD34D"), "target_color"
            ),
            mission_route_width=float(visualization.get("route_width", 0.070)),
            mission_goal_gap=float(visualization.get("goal_gap", 0.085)),
            mission_target_radius=float(visualization.get("target_radius", 0.065)),
            mission_glow_strength=float(visualization.get("glow_strength", 0.90)),
            mission_animation_speed=float(visualization.get("animation_speed", 0.80)),
            mission_grid_intensity=float(visualization.get("grid_intensity", 0.55)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProjectorCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination


class CalibrationState:
    """Thread-safe latest-wins state shared by the browser and display loop."""

    def __init__(self, calibration: ProjectorCalibration) -> None:
        self._lock = Lock()
        self._calibration = calibration
        self._version = 0
        self._last_update_ms = 0.0

    def snapshot(self) -> tuple[ProjectorCalibration, int, float]:
        with self._lock:
            return self._calibration, self._version, self._last_update_ms

    def update_handles(self, handles: Any, *, update_ms: float = 0.0) -> ProjectorCalibration:
        with self._lock:
            self._calibration = self._calibration.with_handles(handles)
            self._version += 1
            self._last_update_ms = float(update_ms)
            return self._calibration

    def update_style(
        self, *, grid_divisions: Any, line_thickness: Any
    ) -> ProjectorCalibration:
        with self._lock:
            self._calibration = self._calibration.with_style(
                grid_divisions=grid_divisions, line_thickness=line_thickness
            )
            self._version += 1
            return self._calibration

    def update_visualization(self, payload: Any) -> ProjectorCalibration:
        with self._lock:
            self._calibration = self._calibration.with_visualization(payload)
            self._version += 1
            return self._calibration
