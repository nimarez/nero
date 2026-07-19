"""Low-latency Vive controller state for projector-floor interaction."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import numpy as np


FLOOR_TARGETS = (
    (0.5, 0.5),
    (0.18, 0.18),
    (0.82, 0.18),
    (0.82, 0.82),
    (0.18, 0.82),
)


@dataclass(frozen=True, slots=True)
class MotionPose:
    sequence: int
    controller_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    received_at: float
    tracking_valid: bool


def map_floor_position(
    position: tuple[float, float, float],
    origin: tuple[float, float, float],
    *,
    span_x_m: float = 3.0,
    span_y_m: float = 2.2,
) -> tuple[float, float]:
    """Map Vive's Z-up floor plane into the normalized marker quadrilateral."""

    return (
        0.5 + (position[0] - origin[0]) / span_x_m,
        0.5 - (position[1] - origin[1]) / span_y_m,
    )


class MotionTracker:
    """Poll the receiver's atomic latest-pose file without adding network hops."""

    def __init__(
        self,
        pose_path: str | Path = "/run/nero/vive_pose.json",
        center_path: str | Path = "~/.config/nero/controller-center.json",
        mapping_path: str | Path = "~/.config/nero/controller-floor-mapping.json",
        heading_path: str | Path = "~/.config/nero/controller-heading.json",
        *,
        stale_after_s: float = 0.15,
    ) -> None:
        self.pose_path = Path(pose_path)
        self.center_path = Path(center_path).expanduser()
        self.mapping_path = Path(mapping_path).expanduser()
        self.heading_path = Path(heading_path).expanduser()
        self.stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._latest: MotionPose | None = None
        self._origin = self._load_origin()
        self._mapping = self._load_mapping()
        self._heading_offset = self._load_heading_offset()
        self._calibration_samples: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._calibration_active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _load_origin(self) -> tuple[float, float, float] | None:
        try:
            values = json.loads(self.center_path.read_text(encoding="utf-8"))["position"]
            if len(values) != 3:
                return None
            return tuple(float(value) for value in values)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_mapping(self) -> tuple[tuple[float, float, float], ...] | None:
        try:
            rows = json.loads(self.mapping_path.read_text(encoding="utf-8"))["matrix"]
            matrix = tuple(tuple(float(value) for value in row) for row in rows)
            return matrix if len(matrix) == 2 and all(len(row) == 3 for row in matrix) else None
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_heading_offset(self) -> float | None:
        try:
            value = float(json.loads(self.heading_path.read_text(encoding="utf-8"))["yaw_offset"])
            return value if math.isfinite(value) else None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def start(self) -> "MotionTracker":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="vive-motion", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        last_stamp = -1
        while not self._stop.is_set():
            try:
                stamp = self.pose_path.stat().st_mtime_ns
                if stamp == last_stamp:
                    time.sleep(0.002)
                    continue
                last_stamp = stamp
                payload: dict[str, Any] = json.loads(self.pose_path.read_text(encoding="utf-8"))
                transport = payload.get("transport") or {}
                pose = MotionPose(
                    sequence=int(payload["sequence"]),
                    controller_id=str(payload["controller_id"]),
                    position=tuple(float(value) for value in payload["position"]),
                    quaternion_xyzw=tuple(
                        float(value) for value in payload["quaternion_xyzw"]
                    ),
                    received_at=float(transport.get("received_at", 0.0)),
                    tracking_valid=bool(payload["tracking_valid"]),
                )
                with self._lock:
                    self._latest = pose
                    should_center = self._origin is None and self._valid_locked(pose)
                if should_center:
                    self.recenter()
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                time.sleep(0.01)

    def _valid_locked(self, pose: MotionPose | None) -> bool:
        return bool(
            pose
            and pose.tracking_valid
            and 0.0 <= time.time() - pose.received_at <= self.stale_after_s
        )

    def recenter(self) -> bool:
        with self._lock:
            pose = self._latest
            if not self._valid_locked(pose):
                return False
            assert pose is not None
            self._origin = pose.position
            origin = self._origin
        self._save_origin(origin)
        return True

    def _save_origin(self, origin: tuple[float, float, float]) -> None:
        self.center_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.center_path.name}.", dir=self.center_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "position": list(origin)}, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.center_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def begin_floor_calibration(self) -> dict[str, Any]:
        with self._lock:
            # Always capture center again. A Lighthouse relocalization can make
            # a previously saved origin stale even when tracking is currently valid.
            self._calibration_samples = []
            self._calibration_active = True
        return self.calibration_status()

    def capture_floor_point(self) -> dict[str, Any]:
        with self._lock:
            pose = self._latest
            if not self._calibration_active:
                raise ValueError("floor calibration is not active")
            if not self._valid_locked(pose):
                raise ValueError("controller is not currently tracked")
            assert pose is not None
            target = FLOOR_TARGETS[len(self._calibration_samples)]
            self._calibration_samples.append(((pose.position[0], pose.position[1]), target))
            samples = tuple(self._calibration_samples)
            origin = pose.position if len(samples) == 1 else None
            if origin is not None:
                self._origin = origin

        if origin is not None:
            self._save_origin(origin)

        if len(samples) >= 3:
            source = np.asarray([[x, y, 1.0] for (x, y), _ in samples], dtype=np.float64)
            destination = np.asarray([uv for _, uv in samples], dtype=np.float64)
            solution, _, rank, _ = np.linalg.lstsq(source, destination, rcond=None)
            if rank < 3:
                raise ValueError("captured controller points are not spread across the floor")
            matrix = tuple(tuple(float(value) for value in row) for row in solution.T)
            with self._lock:
                self._mapping = matrix
            self._save_mapping(matrix, samples)

        with self._lock:
            if len(self._calibration_samples) == len(FLOOR_TARGETS):
                self._calibration_active = False
        return self.calibration_status()

    def _save_mapping(
        self,
        matrix: tuple[tuple[float, float, float], ...],
        samples: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    ) -> None:
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.mapping_path.name}.", dir=self.mapping_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "version": 1,
                        "matrix": [list(row) for row in matrix],
                        "samples": [
                            {"vive_xy": list(xy), "floor_uv": list(uv)} for xy, uv in samples
                        ],
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
            os.replace(temporary, self.mapping_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def calibration_status(self) -> dict[str, Any]:
        with self._lock:
            captured = len(self._calibration_samples)
            active = self._calibration_active
            mapping_ready = self._mapping is not None
        target = FLOOR_TARGETS[captured] if active and captured < len(FLOOR_TARGETS) else None
        return {
            "active": active,
            "captured": captured,
            "total": len(FLOOR_TARGETS),
            "target_uv": list(target) if target else None,
            "mapping_ready": mapping_ready,
        }

    @staticmethod
    def _apply_mapping(
        position: tuple[float, float, float],
        matrix: tuple[tuple[float, float, float], ...],
    ) -> tuple[float, float]:
        x, y = position[:2]
        return (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        )

    @staticmethod
    def _room_frame(
        mapping: tuple[tuple[float, float, float], ...],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Derive an orthonormal metric room frame from the projected grid."""

        linear = np.asarray(
            ((mapping[0][0], mapping[0][1]), (mapping[1][0], mapping[1][1])),
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(linear))) < 1e-9:
            raise ValueError("controller floor mapping is degenerate")
        inverse = np.linalg.inv(linear)
        x_axis = inverse @ np.asarray((1.0, 0.0))
        x_axis /= np.linalg.norm(x_axis)
        grid_up = inverse @ np.asarray((0.0, -1.0))
        y_axis = np.asarray((-x_axis[1], x_axis[0]))
        if float(np.dot(y_axis, grid_up)) < 0.0:
            y_axis = -y_axis
        return x_axis, y_axis, math.atan2(float(x_axis[1]), float(x_axis[0]))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + pi) % (2.0 * pi) - pi

    @staticmethod
    def _lighthouse_yaw(quaternion_xyzw: tuple[float, float, float, float]) -> float:
        x, y, z, w = quaternion_xyzw
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @classmethod
    def _room_pose(
        cls,
        pose: MotionPose,
        origin: tuple[float, float, float],
        mapping: tuple[tuple[float, float, float], ...],
        heading_offset: float | None,
    ) -> dict[str, Any]:
        x_axis, y_axis, axis_yaw = cls._room_frame(mapping)
        delta = np.asarray(pose.position[:2], dtype=np.float64) - np.asarray(
            origin[:2], dtype=np.float64
        )
        raw_yaw = cls._wrap_angle(cls._lighthouse_yaw(pose.quaternion_xyzw) - axis_yaw)
        yaw = cls._wrap_angle(raw_yaw + (heading_offset or 0.0))
        return {
            "x": float(np.dot(delta, x_axis)),
            "y": float(np.dot(delta, y_axis)),
            "yaw": yaw,
            "t": pose.received_at,
            "valid": True,
            "source": f"vive:{pose.controller_id}",
            "frame_id": "room_floor",
            "heading_calibrated": heading_offset is not None,
        }

    def calibrate_robot_forward(self) -> dict[str, Any]:
        """Define the controller's current taped orientation as robot +X."""

        with self._lock:
            pose = self._latest
            mapping = self._mapping
            if not self._valid_locked(pose):
                raise ValueError("controller is not currently tracked")
            if mapping is None:
                raise ValueError("floor mapping is not calibrated")
            assert pose is not None
            _, _, axis_yaw = self._room_frame(mapping)
            raw_yaw = self._wrap_angle(self._lighthouse_yaw(pose.quaternion_xyzw) - axis_yaw)
            self._heading_offset = self._wrap_angle(-raw_yaw)
            heading_offset = self._heading_offset

        self.heading_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.heading_path.name}.", dir=self.heading_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "version": 1,
                        "yaw_offset": heading_offset,
                        "controller_id": pose.controller_id,
                        "captured_at": time.time(),
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
            os.replace(temporary, self.heading_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return self.snapshot()

    @classmethod
    def _room_points_to_floor_uv(
        cls,
        points: list[list[float]] | list[tuple[float, float]],
        origin: tuple[float, float, float],
        mapping: tuple[tuple[float, float, float], ...],
    ) -> list[list[float]]:
        x_axis, y_axis, _ = cls._room_frame(mapping)
        origin_xy = np.asarray(origin[:2], dtype=np.float64)
        result = []
        for x, y in points:
            lighthouse_xy = origin_xy + float(x) * x_axis + float(y) * y_axis
            uv = cls._apply_mapping((float(lighthouse_xy[0]), float(lighthouse_xy[1]), 0.0), mapping)
            result.append([float(uv[0]), float(uv[1])])
        return result

    def room_points_to_floor_uv(
        self, points: list[list[float]] | list[tuple[float, float]]
    ) -> list[list[float]]:
        with self._lock:
            origin = self._origin
            mapping = self._mapping
        if origin is None or mapping is None:
            return []
        return self._room_points_to_floor_uv(points, origin, mapping)

    @classmethod
    def _robot_frame_uv(
        cls,
        robot_pose: dict[str, Any],
        origin: tuple[float, float, float],
        mapping: tuple[tuple[float, float, float], ...],
    ) -> dict[str, Any]:
        x, y, yaw = robot_pose["x"], robot_pose["y"], robot_pose["yaw"]
        forward = np.asarray((math.cos(yaw), math.sin(yaw)))
        left = np.asarray((-forward[1], forward[0]))
        center = np.asarray((x, y))
        grid_lines: list[list[list[float]]] = []
        for lateral in (-0.4, 0.0, 0.4):
            start = center - 0.45 * forward + lateral * left
            end = center + 0.85 * forward + lateral * left
            grid_lines.append(cls._room_points_to_floor_uv([start.tolist(), end.tolist()], origin, mapping))
        for longitudinal in (-0.4, 0.0, 0.4):
            start = center + longitudinal * forward - 0.45 * left
            end = center + longitudinal * forward + 0.45 * left
            grid_lines.append(cls._room_points_to_floor_uv([start.tolist(), end.tolist()], origin, mapping))
        footprint = [
            center + 0.38 * forward + 0.24 * left,
            center + 0.38 * forward - 0.24 * left,
            center - 0.32 * forward - 0.24 * left,
            center - 0.32 * forward + 0.24 * left,
        ]
        return {
            "center": cls._room_points_to_floor_uv([center.tolist()], origin, mapping)[0],
            "x_axis": cls._room_points_to_floor_uv(
                [center.tolist(), (center + 0.78 * forward).tolist()], origin, mapping
            ),
            "y_axis": cls._room_points_to_floor_uv(
                [center.tolist(), (center + 0.58 * left).tolist()], origin, mapping
            ),
            "grid_lines": grid_lines,
            "footprint": cls._room_points_to_floor_uv(
                [point.tolist() for point in footprint], origin, mapping
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pose = self._latest
            origin = self._origin
            mapping = self._mapping
            heading_offset = self._heading_offset
            valid = self._valid_locked(pose)
        uv = None
        ring_uv = None
        robot_pose = None
        robot_frame_uv = None
        if valid and pose:
            # Vive X/Y are the room's floor-plane axes. Z is controller height,
            # so intentionally discard it: every height maps straight down to
            # the same physical floor point.
            floor_position = pose.position
            if mapping:
                uv = self._apply_mapping(floor_position, mapping)
                radius_m = 0.28
                ring_uv = [
                    self._apply_mapping(
                        (
                            floor_position[0] + radius_m * cos(index * 2 * pi / 96),
                            floor_position[1] + radius_m * sin(index * 2 * pi / 96),
                            floor_position[2],
                        ),
                        mapping,
                    )
                    for index in range(96)
                ]
                if origin is not None:
                    robot_pose = self._room_pose(pose, origin, mapping, heading_offset)
                    if robot_pose["heading_calibrated"]:
                        robot_frame_uv = self._robot_frame_uv(robot_pose, origin, mapping)
            elif origin:
                uv = map_floor_position(floor_position, origin)
                ring_uv = [
                    map_floor_position(
                        (
                            floor_position[0] + 0.28 * cos(index * 2 * pi / 96),
                            floor_position[1] + 0.28 * sin(index * 2 * pi / 96),
                            floor_position[2],
                        ),
                        origin,
                    )
                    for index in range(96)
                ]
        return {
            "valid": valid,
            "centered": origin is not None,
            "controller_id": pose.controller_id if pose else None,
            "sequence": pose.sequence if pose else None,
            "position": list(pose.position) if pose else None,
            "quaternion_xyzw": list(pose.quaternion_xyzw) if pose else None,
            "origin": list(origin) if origin else None,
            "uv": list(uv) if uv else None,
            "ring_uv": [list(point) for point in ring_uv] if ring_uv else None,
            "age_ms": (time.time() - pose.received_at) * 1000 if pose else None,
            "height_compensated": False,
            "vertical_ignored": True,
            "robot_pose": robot_pose,
            "robot_frame_uv": robot_frame_uv,
            "calibration": self.calibration_status(),
        }
