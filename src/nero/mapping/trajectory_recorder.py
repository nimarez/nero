"""Trajectory recorder for mapping missions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in the trajectory."""
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


class TrajectoryRecorder:
    """Records and manages robot trajectories during mapping."""

    def __init__(self, output_dir: str = "output/trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trajectory: list[TrajectoryPoint] = []
        self._is_recording = False
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start recording trajectory."""
        self._is_recording = True
        self._start_time = time.time()
        self._trajectory.clear()
        logger.info("Started trajectory recording")

    def stop(self) -> None:
        """Stop recording trajectory."""
        self._is_recording = False
        logger.info(f"Stopped recording. {len(self._trajectory)} points recorded")

    def add_point(self, pose: np.ndarray) -> None:
        """Add a point to the trajectory.

        Args:
            pose: 4x4 transformation matrix
        """
        if not self._is_recording:
            return

        timestamp = time.time() - (self._start_time or time.time())

        # Extract position
        x, y, z = pose[:3, 3]

        # Extract orientation as RPY
        from scipy.spatial.transform import Rotation
        R = pose[:3, :3]
        rpy = Rotation.from_matrix(R).as_euler("xyz")
        roll, pitch, yaw = rpy

        point = TrajectoryPoint(
            timestamp=timestamp,
            x=float(x),
            y=float(y),
            z=float(z),
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
        )
        self._trajectory.append(point)

    def get_trajectory(self) -> list[TrajectoryPoint]:
        return list(self._trajectory)

    def get_length(self) -> float:
        """Get total trajectory length in meters."""
        if len(self._trajectory) < 2:
            return 0.0

        length = 0.0
        for i in range(1, len(self._trajectory)):
            p1 = self._trajectory[i - 1]
            p2 = self._trajectory[i]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            length += np.sqrt(dx**2 + dy**2 + dz**2)

        return length

    def get_bounds(self) -> dict:
        """Get trajectory bounding box."""
        if not self._trajectory:
            return {"min": [0, 0, 0], "max": [0, 0, 0]}

        xs = [p.x for p in self._trajectory]
        ys = [p.y for p in self._trajectory]
        zs = [p.z for p in self._trajectory]

        return {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        }

    def save(self, filename: Optional[str] = None) -> str:
        """Save trajectory to file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"

        path = self.output_dir / filename

        data = {
            "version": "1.0",
            "num_points": len(self._trajectory),
            "length_meters": self.get_length(),
            "bounds": self.get_bounds(),
            "points": [asdict(p) for p in self._trajectory],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved trajectory to {path}")
        return str(path)

    def load(self, path: str) -> list[TrajectoryPoint]:
        """Load trajectory from file.

        Args:
            path: Path to trajectory file

        Returns:
            List of trajectory points
        """
        with open(path) as f:
            data = json.load(f)

        self._trajectory = [
            TrajectoryPoint(**p) for p in data["points"]
        ]
        logger.info(f"Loaded {len(self._trajectory)} points from {path}")
        return self._trajectory

    def is_recording(self) -> bool:
        return self._is_recording

    def get_point_count(self) -> int:
        return len(self._trajectory)