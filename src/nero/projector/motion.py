"""Low-latency Vive controller state for projector-floor interaction."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        *,
        stale_after_s: float = 0.15,
    ) -> None:
        self.pose_path = Path(pose_path)
        self.center_path = Path(center_path).expanduser()
        self.stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._latest: MotionPose | None = None
        self._origin = self._load_origin()
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pose = self._latest
            origin = self._origin
            valid = self._valid_locked(pose)
        uv = map_floor_position(pose.position, origin) if valid and pose and origin else None
        return {
            "valid": valid,
            "centered": origin is not None,
            "controller_id": pose.controller_id if pose else None,
            "sequence": pose.sequence if pose else None,
            "position": list(pose.position) if pose else None,
            "origin": list(origin) if origin else None,
            "uv": list(uv) if uv else None,
            "age_ms": (time.time() - pose.received_at) * 1000 if pose else None,
        }
