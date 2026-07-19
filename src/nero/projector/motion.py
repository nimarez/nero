"""Low-latency Vive controller state for projector-floor interaction."""

from __future__ import annotations

import json
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
        *,
        stale_after_s: float = 0.15,
    ) -> None:
        self.pose_path = Path(pose_path)
        self.center_path = Path(center_path).expanduser()
        self.mapping_path = Path(mapping_path).expanduser()
        self.stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._latest: MotionPose | None = None
        self._origin = self._load_origin()
        self._mapping = self._load_mapping()
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
        return True

    def begin_floor_calibration(self) -> dict[str, Any]:
        with self._lock:
            # The operator already established center with ``recenter``. Reuse
            # that exact pose as point one so the guided pass starts at a corner.
            self._calibration_samples = (
                [((self._origin[0], self._origin[1]), FLOOR_TARGETS[0])]
                if self._origin is not None
                else []
            )
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pose = self._latest
            origin = self._origin
            mapping = self._mapping
            valid = self._valid_locked(pose)
        uv = None
        ring_uv = None
        if valid and pose:
            if mapping:
                uv = self._apply_mapping(pose.position, mapping)
                radius_m = 0.28
                ring_uv = [
                    self._apply_mapping(
                        (
                            pose.position[0] + radius_m * cos(index * 2 * pi / 96),
                            pose.position[1] + radius_m * sin(index * 2 * pi / 96),
                            pose.position[2],
                        ),
                        mapping,
                    )
                    for index in range(96)
                ]
            elif origin:
                uv = map_floor_position(pose.position, origin)
                ring_uv = [
                    map_floor_position(
                        (
                            pose.position[0] + 0.28 * cos(index * 2 * pi / 96),
                            pose.position[1] + 0.28 * sin(index * 2 * pi / 96),
                            pose.position[2],
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
            "origin": list(origin) if origin else None,
            "uv": list(uv) if uv else None,
            "ring_uv": [list(point) for point in ring_uv] if ring_uv else None,
            "age_ms": (time.time() - pose.received_at) * 1000 if pose else None,
            "calibration": self.calibration_status(),
        }
