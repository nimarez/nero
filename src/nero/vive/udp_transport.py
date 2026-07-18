"""Versioned UDP transport for Vive poses computed on a Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from nero.vive.pose_source import DEFAULT_SURVIVE_ARGS, PoseSource, TimedPose, VivePoseSource

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_HOST = "10.77.0.1"
DEFAULT_PORT = 43100
MAX_DATAGRAM_BYTES = 4096
DEFAULT_STALE_AFTER_S = 0.15


def _vector(value: Any, length: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain {length} numbers") from error
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain {length} finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class PosePacket:
    """One transport-neutral 6-DoF controller sample.

    ``timestamp`` is Unix time at the Pi when the libsurvive pose was received.
    Positions are metres, linear velocity is m/s, and angular velocity is rad/s.
    Quaternions are always ordered x, y, z, w.
    """

    sequence: int
    timestamp: float
    controller_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    tracking_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "controller_id": self.controller_id,
            "position": list(self.position),
            "quaternion_xyzw": list(self.quaternion_xyzw),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "tracking_valid": self.tracking_valid,
        }

    def encode(self) -> bytes:
        return json.dumps(
            self.to_dict(), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> "PosePacket":
        if len(data) > MAX_DATAGRAM_BYTES:
            raise ValueError("pose datagram is too large")
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("pose datagram is not valid JSON") from error
        if not isinstance(payload, dict) or payload.get("version") != PROTOCOL_VERSION:
            raise ValueError("unsupported pose protocol version")
        controller_id = payload.get("controller_id")
        if not isinstance(controller_id, str) or not controller_id or len(controller_id) > 64:
            raise ValueError("controller_id must be 1-64 characters")
        sequence = payload.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        timestamp = float(payload.get("timestamp"))
        if not math.isfinite(timestamp) or timestamp <= 0:
            raise ValueError("timestamp must be a positive finite Unix time")
        tracking_valid = payload.get("tracking_valid")
        if not isinstance(tracking_valid, bool):
            raise ValueError("tracking_valid must be a boolean")
        return cls(
            sequence=sequence,
            timestamp=timestamp,
            controller_id=controller_id,
            position=_vector(payload.get("position"), 3, "position"),
            quaternion_xyzw=_vector(payload.get("quaternion_xyzw"), 4, "quaternion_xyzw"),
            linear_velocity=_vector(payload.get("linear_velocity"), 3, "linear_velocity"),
            angular_velocity=_vector(payload.get("angular_velocity"), 3, "angular_velocity"),
            tracking_valid=tracking_valid,
        )

    def robot_pose(self) -> dict[str, float | bool | str]:
        """Project the 6-DoF packet into the team's planar RobotPose contract."""
        x, y, z, w = self.quaternion_xyzw
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return {
            "x": self.position[0],
            "y": self.position[1],
            "yaw": yaw,
            "t": self.timestamp,
            "valid": self.tracking_valid,
            "source": f"vive:{self.controller_id}",
        }


class PoseKinematics:
    """Estimate linear and angular velocity from consecutive poses."""

    def __init__(self) -> None:
        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None
        self._time: float | None = None

    def update(
        self, position: Sequence[float], quaternion_xyzw: Sequence[float], sample_time: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        current_position = np.asarray(position, dtype=np.float64)
        current_quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
        quaternion_norm = float(np.linalg.norm(current_quaternion))
        if current_position.shape != (3,) or current_quaternion.shape != (4,):
            raise ValueError("invalid pose dimensions")
        if not np.all(np.isfinite(current_position)) or not math.isfinite(quaternion_norm):
            raise ValueError("pose contains non-finite values")
        if quaternion_norm < 1e-9:
            raise ValueError("quaternion has zero length")
        current_quaternion /= quaternion_norm

        linear = np.zeros(3, dtype=np.float64)
        angular = np.zeros(3, dtype=np.float64)
        if self._time is not None and self._position is not None and self._quaternion is not None:
            dt = sample_time - self._time
            if 1e-6 < dt < 1.0:
                linear = (current_position - self._position) / dt
                previous_conjugate = np.array(
                    [-self._quaternion[0], -self._quaternion[1], -self._quaternion[2], self._quaternion[3]]
                )
                x1, y1, z1, w1 = current_quaternion
                x2, y2, z2, w2 = previous_conjugate
                delta = np.array(
                    [
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    ],
                    dtype=np.float64,
                )
                if delta[3] < 0:
                    delta = -delta
                vector_norm = float(np.linalg.norm(delta[:3]))
                if vector_norm > 1e-9:
                    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(delta[3])))
                    angular = delta[:3] / vector_norm * (angle / dt)

        self._position = current_position
        self._quaternion = current_quaternion
        self._time = sample_time
        return tuple(linear.tolist()), tuple(angular.tolist())


class PoseUdpPublisher:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sock: socket.socket | None = None,
    ) -> None:
        self.destination = (host, port)
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0
        self._kinematics: dict[str, PoseKinematics] = {}

    def send_pose(self, pose: TimedPose) -> PosePacket:
        sample_time = self._monotonic_clock()
        position = _vector(pose.position, 3, "position")
        quaternion = _vector(pose.quaternion_xyzw, 4, "quaternion_xyzw")
        estimator = self._kinematics.setdefault(pose.name, PoseKinematics())
        try:
            linear, angular = estimator.update(position, quaternion, sample_time)
            valid = True
        except ValueError:
            linear = angular = (0.0, 0.0, 0.0)
            valid = False
        packet = PosePacket(
            sequence=self._sequence,
            timestamp=self._wall_clock(),
            controller_id=pose.name,
            position=position,
            quaternion_xyzw=quaternion,
            linear_velocity=linear,
            angular_velocity=angular,
            tracking_valid=valid,
        )
        self._socket.sendto(packet.encode(), self.destination)
        self._sequence += 1
        return packet

    def stream(self, source: PoseSource) -> None:
        for pose in source.poll():
            packet = self.send_pose(pose)
            if packet.sequence and packet.sequence % 100 == 0:
                logger.info(
                    "sent %d poses to %s:%d | %s position=%s",
                    packet.sequence,
                    *self.destination,
                    packet.controller_id,
                    tuple(round(value, 3) for value in packet.position),
                )


@dataclass(frozen=True, slots=True)
class ReceivedPose:
    packet: PosePacket
    sender: tuple[str, int]
    received_at: float
    latency_s: float
    dropped_since_previous: int
    out_of_order: bool

    def current_packet(self, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> PosePacket:
        if time.time() - self.received_at <= stale_after_s:
            return self.packet
        return replace(self.packet, tracking_valid=False)


class PoseUdpReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        *,
        sock: socket.socket | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if sock is None:
            self._socket.bind((host, port))
        self._wall_clock = wall_clock
        self._last_sequence: dict[tuple[str, str, int], int] = {}

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._socket.getsockname()[:2]
        return str(host), int(port)

    def receive(self, timeout_s: float | None = None) -> ReceivedPose:
        self._socket.settimeout(timeout_s)
        data, sender = self._socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
        packet = PosePacket.decode(data)
        received_at = self._wall_clock()
        sequence_key = (packet.controller_id, str(sender[0]), int(sender[1]))
        previous = self._last_sequence.get(sequence_key)
        out_of_order = previous is not None and packet.sequence <= previous
        dropped = max(0, packet.sequence - previous - 1) if previous is not None else 0
        if not out_of_order:
            self._last_sequence[sequence_key] = packet.sequence
        return ReceivedPose(
            packet=packet,
            sender=(str(sender[0]), int(sender[1])),
            received_at=received_at,
            latency_s=max(0.0, received_at - packet.timestamp),
            dropped_since_previous=dropped,
            out_of_order=out_of_order,
        )


def publisher_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish libsurvive Vive poses over UDP")
    parser.add_argument("--host", default=DEFAULT_HOST, help="POS destination address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, survive_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = VivePoseSource(survive_args or DEFAULT_SURVIVE_ARGS)
    PoseUdpPublisher(args.host, args.port).stream(source)
    return 0


def receiver_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Receive Vive pose UDP packets")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--json", action="store_true", help="Print one JSON packet per line")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receiver = PoseUdpReceiver(args.bind, args.port)
    logger.info("listening for Vive poses on %s:%d", *receiver.address)
    while True:
        received = receiver.receive()
        if args.json:
            print(json.dumps(received.packet.to_dict(), separators=(",", ":")), flush=True)
        elif received.packet.sequence % 100 == 0:
            logger.info(
                "%s seq=%d latency=%.1fms dropped=%d position=%s",
                received.packet.controller_id,
                received.packet.sequence,
                received.latency_s * 1000.0,
                received.dropped_since_previous,
                tuple(round(value, 3) for value in received.packet.position),
            )


if __name__ == "__main__":
    raise SystemExit(publisher_main())
