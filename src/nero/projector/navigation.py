"""Shared room-frame pose, goal, and trajectory state for projector handoff."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


FRAME_ID = "room_floor"
MAX_TRAJECTORY_POINTS = 2048


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ProjectorNavigationState:
    """Thread-safe contract boundary; this class never commands the robot."""

    def __init__(
        self,
        goal_path: str | Path = "~/.config/nero/projector-goal.json",
    ) -> None:
        self.goal_path = Path(goal_path).expanduser()
        self._lock = threading.Lock()
        self._goal = self._load_goal()
        self._trajectory: dict[str, Any] | None = None
        self._version = 0

    def _load_goal(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.goal_path.read_text(encoding="utf-8"))
            return self._validated_goal(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _validated_goal(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("goal must be a JSON object")
        return {
            "x": _finite(payload.get("x"), "goal.x"),
            "y": _finite(payload.get("y"), "goal.y"),
            "yaw": _wrap_angle(_finite(payload.get("yaw", 0.0), "goal.yaw")),
            "frame_id": FRAME_ID,
            "source": str(payload.get("source") or "operator"),
        }

    def set_goal(self, payload: Any) -> dict[str, Any]:
        goal = self._validated_goal(payload)
        goal["saved_at"] = time.time()
        self.goal_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.goal_path.name}.", dir=self.goal_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, **goal}, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.goal_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        with self._lock:
            self._goal = goal
            self._trajectory = None
            self._version += 1
        return self.snapshot()

    def set_trajectory(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("trajectory must be a JSON object")
        raw_points = payload.get("waypoints")
        if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= MAX_TRAJECTORY_POINTS:
            raise ValueError("trajectory.waypoints must contain 2-2048 points")
        points = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"trajectory.waypoints[{index}] must be [x, y]")
            points.append(
                [
                    _finite(point[0], f"trajectory.waypoints[{index}].x"),
                    _finite(point[1], f"trajectory.waypoints[{index}].y"),
                ]
            )
        trajectory = {
            "frame_id": FRAME_ID,
            "waypoints": points,
            "source": str(payload.get("source") or "nima"),
            "updated_at": time.time(),
        }
        with self._lock:
            self._trajectory = trajectory
            self._version += 1
        return self.snapshot()

    def snapshot(self, robot_pose: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            goal = dict(self._goal) if self._goal else None
            trajectory = dict(self._trajectory) if self._trajectory else None
            version = self._version
        if trajectory is None and goal and robot_pose and robot_pose.get("valid"):
            trajectory = {
                "frame_id": FRAME_ID,
                "waypoints": [
                    [float(robot_pose["x"]), float(robot_pose["y"])],
                    [float(goal["x"]), float(goal["y"])],
                ],
                "source": "direct-preview",
                "updated_at": time.time(),
            }
        return {
            "version": 1,
            "state_version": version,
            "frame_id": FRAME_ID,
            "robot_pose": dict(robot_pose) if robot_pose else None,
            "goal_pose": goal,
            "trajectory": trajectory,
            "control_authority": "none",
        }
