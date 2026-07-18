"""Thread-safe timestamped IMU buffering for ORB-SLAM3."""

from __future__ import annotations

import threading
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def stamp_seconds(stamp: object) -> float:
    """Convert a ROS-style sec/nanosec stamp to seconds."""
    return float(getattr(stamp, "sec")) + float(getattr(stamp, "nanosec")) * 1e-9


@dataclass(frozen=True)
class IMUMeasurement:
    timestamp: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]

    def as_orbslam_tuple(self) -> tuple[float, ...]:
        return (*self.accel, *self.gyro, self.timestamp)


class IMUBuffer:
    """Keeps ordered measurements and drains camera-frame intervals exactly once."""

    def __init__(self, maxlen: int = 4000):
        self._samples: deque[IMUMeasurement] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, measurement: IMUMeasurement) -> None:
        if not np.isfinite(measurement.timestamp):
            raise ValueError("IMU timestamp must be finite")
        values = (*measurement.accel, *measurement.gyro)
        if not np.all(np.isfinite(values)):
            raise ValueError("IMU acceleration and gyro values must be finite")
        with self._lock:
            if self._samples and measurement.timestamp <= self._samples[-1].timestamp:
                return
            self._samples.append(measurement)

    def extend(self, measurements: Iterable[IMUMeasurement]) -> None:
        for measurement in measurements:
            self.append(measurement)

    def between(self, start: float | None, end: float) -> list[IMUMeasurement]:
        """Return samples in ``(start, end]`` while retaining newer samples."""
        with self._lock:
            selected = [
                sample
                for sample in self._samples
                if (start is None or sample.timestamp > start)
                and sample.timestamp <= end
            ]
            while self._samples and self._samples[0].timestamp <= end:
                self._samples.popleft()
        return selected

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class K1IMUSource:
    """Internal K1 ROS IMU subscriber; agents never need to pass sensor handles."""

    def __init__(self, iface: str | None = None, maxlen: int = 4000):
        self.buffer = IMUBuffer(maxlen=maxlen)
        self.iface = iface or os.getenv("BOOSTER_NET_IF", "lo")
        self._subscriber = None

    def start(self) -> None:
        try:
            import booster_robotics_sdk_python as br
        except ImportError as exc:
            raise RuntimeError(
                "official Booster SDK is unavailable on this platform"
            ) from exc

        br.ChannelFactory.Instance().Init(0, self.iface)

        def callback(message: object) -> None:
            stamp = stamp_seconds(message.header.stamp)
            accel = message.linear_acceleration
            gyro = message.angular_velocity
            self.buffer.append(
                IMUMeasurement(
                    timestamp=stamp,
                    accel=(float(accel.x), float(accel.y), float(accel.z)),
                    gyro=(float(gyro.x), float(gyro.y), float(gyro.z)),
                )
            )

        self._subscriber = br.B1RosImuSubscriber(callback)
        self._subscriber.InitChannel()
        logger.info("K1 IMU subscriber active on interface %s", self.iface)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.CloseChannel()
            self._subscriber = None
