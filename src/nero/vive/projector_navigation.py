"""Fail-closed client for PR 8's projector navigation handoff."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)


class ProjectorNavigationSource:
    """Cache room-frame Vive robot and object poses from the projector server."""

    def __init__(
        self,
        base_url: str,
        *,
        stale_after_s: float = 0.25,
        request_timeout_s: float = 0.20,
        poll_rate_hz: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        opener: Callable[..., Any] = urlopen,
        autostart: bool = True,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("projector URL must begin with http:// or https://")
        values = (stale_after_s, request_timeout_s, poll_rate_hz)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("projector timing values must be positive and finite")
        self.base_url = base_url.rstrip("/")
        self.stale_after_s = stale_after_s
        self.request_timeout_s = request_timeout_s
        self.poll_rate_hz = poll_rate_hz
        self._monotonic = monotonic
        self._opener = opener
        self._lock = threading.Lock()
        self._robot_pose: np.ndarray | None = None
        self._goal_pose: np.ndarray | None = None
        self._received_at: float | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        if autostart:
            self._thread = threading.Thread(
                target=self._poll,
                name="projector-navigation",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _pose(payload: Any, name: str) -> np.ndarray:
        if not isinstance(payload, dict):
            raise ValueError(f"{name} is unavailable")
        if payload.get("frame_id") != "room_floor":
            raise ValueError(f"{name} is not in room_floor")
        values = np.array([payload.get("x"), payload.get("y"), payload.get("yaw")], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite x, y, and yaw")
        return values

    def poll_once(self) -> None:
        """Fetch and atomically validate one navigation snapshot."""
        with self._opener(
            f"{self.base_url}/api/navigation/state",
            timeout=self.request_timeout_s,
        ) as response:
            payload = json.load(response)
        if payload.get("version") != 1 or payload.get("frame_id") != "room_floor":
            raise ValueError("unsupported projector navigation contract")
        robot = payload.get("robot_pose")
        if not isinstance(robot, dict) or not robot.get("valid"):
            raise ValueError("projector Vive robot pose is invalid")
        if not robot.get("heading_calibrated"):
            raise ValueError("projector robot heading is not calibrated")
        robot_pose = self._pose(robot, "robot_pose")
        goal_pose = self._pose(payload.get("goal_pose"), "goal_pose")
        with self._lock:
            self._robot_pose = robot_pose
            self._goal_pose = goal_pose
            self._received_at = self._monotonic()

    def _poll(self) -> None:
        period = 1.0 / self.poll_rate_hz
        while not self._closed:
            started = self._monotonic()
            try:
                self.poll_once()
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                logger.debug("projector navigation unavailable: %s", error)
            elapsed = self._monotonic() - started
            time.sleep(max(0.001, period - elapsed))

    def current_navigation(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one frame-consistent robot/object snapshot, or fail closed."""
        with self._lock:
            if (
                self._robot_pose is None
                or self._goal_pose is None
                or self._received_at is None
                or self._monotonic() - self._received_at > self.stale_after_s
            ):
                return None
            return self._robot_pose.copy(), self._goal_pose.copy()

    def current_pose(self) -> np.ndarray | None:
        state = self.current_navigation()
        return None if state is None else state[0]

    def publish_trajectory(self, points: np.ndarray) -> None:
        """Return Nero's actual approach path for the projector overlay."""
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not 2 <= len(values) <= 2048:
            raise ValueError("trajectory must contain 2-2048 [x, y] points")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory points must be finite")
        body = json.dumps(
            {"waypoints": values.tolist(), "source": "nero-vive-pursuit"},
            separators=(",", ":"),
        ).encode()
        request = Request(
            f"{self.base_url}/api/navigation/trajectory",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self.request_timeout_s) as response:
            response.read()

    def close(self) -> None:
        self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
