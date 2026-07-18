from __future__ import annotations

import math
import socket
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nero.vive.pose_source import TimedPose
from nero.vive.udp_transport import (
    LatestPoseWriter,
    PoseKinematics,
    PosePacket,
    PoseUdpPublisher,
    PoseUdpReceiver,
    ReceivedPose,
)


def sample_packet(sequence: int = 7) -> PosePacket:
    return PosePacket(
        sequence=sequence,
        timestamp=1_750_000_000.25,
        controller_id="WW0",
        position=(1.0, 2.0, 3.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=(0.1, 0.2, 0.3),
        angular_velocity=(0.0, 0.0, 0.5),
        tracking_valid=True,
    )


def test_packet_round_trip_and_robot_pose() -> None:
    packet = sample_packet()
    decoded = PosePacket.decode(packet.encode())
    assert decoded == packet
    assert decoded.robot_pose() == {
        "x": 1.0,
        "y": 2.0,
        "yaw": 0.0,
        "t": 1_750_000_000.25,
        "valid": True,
        "source": "vive:WW0",
    }


def test_packet_rejects_unknown_version() -> None:
    data = sample_packet().encode().replace(b'"version":1', b'"version":2')
    with pytest.raises(ValueError, match="unsupported"):
        PosePacket.decode(data)


def test_kinematics_estimates_translation_and_rotation() -> None:
    estimator = PoseKinematics()
    linear, angular = estimator.update((0, 0, 0), (0, 0, 0, 1), 10.0)
    assert linear == (0.0, 0.0, 0.0)
    assert angular == (0.0, 0.0, 0.0)

    half_angle = math.pi / 4
    linear, angular = estimator.update(
        (1, 0, 0), (0, 0, math.sin(half_angle), math.cos(half_angle)), 10.5
    )
    assert linear == pytest.approx((2.0, 0.0, 0.0))
    assert angular == pytest.approx((0.0, 0.0, math.pi))


def test_udp_loopback_tracks_sequence_and_latency() -> None:
    receiver = PoseUdpReceiver("127.0.0.1", 0, wall_clock=lambda: 1000.010)
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    publisher = PoseUdpPublisher(
        *receiver.address,
        wall_clock=lambda: 1000.0,
        monotonic_clock=iter((20.0, 20.1)).__next__,
        sock=send_socket,
    )
    first = TimedPose(
        name="WW0",
        timestamp=0.0,
        position=np.array([0.0, 0.0, 0.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    second = TimedPose(
        name="WW0",
        timestamp=0.1,
        position=np.array([0.1, 0.0, 0.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )

    publisher.send_pose(first)
    received_first = receiver.receive(timeout_s=1)
    publisher.send_pose(second)
    received_second = receiver.receive(timeout_s=1)

    assert received_first.packet.sequence == 0
    assert received_second.packet.sequence == 1
    assert received_second.packet.linear_velocity == pytest.approx((1.0, 0.0, 0.0))
    assert received_second.latency_s == pytest.approx(0.010)
    assert received_second.dropped_since_previous == 0
    assert not received_second.out_of_order


def test_stream_filters_lighthouses_and_selects_device() -> None:
    class FakeSource:
        def poll(self) -> Iterator[TimedPose]:
            for name in ("LH0", "WW0", "WW1"):
                yield TimedPose(
                    name=name,
                    timestamp=0.0,
                    position=np.zeros(3),
                    quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                )

    class FakeSocket:
        def __init__(self) -> None:
            self.datagrams: list[bytes] = []

        def sendto(self, data: bytes, destination: tuple[str, int]) -> None:
            self.datagrams.append(data)

    fake_socket = FakeSocket()
    publisher = PoseUdpPublisher(
        sock=fake_socket,  # type: ignore[arg-type]
        device_id="WW1",
        wall_clock=lambda: 1000.0,
        monotonic_clock=lambda: 10.0,
    )
    publisher.stream(FakeSource())

    assert len(fake_socket.datagrams) == 1
    assert PosePacket.decode(fake_socket.datagrams[0]).controller_id == "WW1"


def test_latest_pose_writer_exposes_full_and_planar_state(tmp_path: Path) -> None:
    output = tmp_path / "vive_pose.json"
    received = ReceivedPose(
        packet=sample_packet(),
        sender=("10.77.0.212", 54321),
        received_at=1_750_000_000.26,
        latency_s=0.01,
        dropped_since_previous=2,
        out_of_order=False,
    )

    LatestPoseWriter(output).write(received)

    state = __import__("json").loads(output.read_text())
    assert state["controller_id"] == "WW0"
    assert state["robot_pose"]["source"] == "vive:WW0"
    assert state["transport"]["latency_ms"] == pytest.approx(10.0)
    assert state["transport"]["dropped_since_previous"] == 2
