from __future__ import annotations

import json

import numpy as np
import pytest

from nero.vive.ros_bridge import ViveCalibration, read_latest_pose
from nero.vive.udp_transport import PosePacket


def state_payload(*, valid: bool = True) -> dict:
    return {
        "version": 1,
        "sequence": 42,
        "timestamp": 1_750_000_000.0,
        "controller_id": "WW0",
        "position": [1.0, 2.0, 3.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity": [0.1, 0.2, 0.3],
        "angular_velocity": [0.0, 0.0, 0.5],
        "tracking_valid": valid,
        "transport": {"received_at": 100.0},
    }


def test_read_latest_pose_preserves_fresh_valid_state(tmp_path) -> None:
    path = tmp_path / "vive_pose.json"
    path.write_text(json.dumps(state_payload()))

    state = read_latest_pose(path, stale_after_s=0.15, wall_clock=lambda: 100.1)

    assert state.packet.tracking_valid is True
    assert state.packet.position == (1.0, 2.0, 3.0)
    assert state.age_s == pytest.approx(0.1)


def test_read_latest_pose_fails_closed_when_stale(tmp_path) -> None:
    path = tmp_path / "vive_pose.json"
    path.write_text(json.dumps(state_payload()))

    state = read_latest_pose(path, stale_after_s=0.15, wall_clock=lambda: 100.151)

    assert state.packet.tracking_valid is False


def test_read_latest_pose_preserves_explicit_tracking_loss(tmp_path) -> None:
    path = tmp_path / "vive_pose.json"
    path.write_text(json.dumps(state_payload(valid=False)))

    state = read_latest_pose(path, stale_after_s=0.15, wall_clock=lambda: 100.01)

    assert state.packet.tracking_valid is False


def test_read_latest_pose_rejects_zero_quaternion(tmp_path) -> None:
    path = tmp_path / "vive_pose.json"
    payload = state_payload()
    payload["quaternion_xyzw"] = [0.0, 0.0, 0.0, 0.0]
    path.write_text(json.dumps(payload))

    state = read_latest_pose(path, stale_after_s=0.15, wall_clock=lambda: 100.01)

    assert state.packet.tracking_valid is False


def test_calibration_transforms_controller_pose_into_map_body_pose(tmp_path) -> None:
    path = tmp_path / "vive_calibration.json"
    path.write_text(
        json.dumps(
            {
                "map_from_lighthouse": {
                    "translation": [10.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 2.0],
                },
                "base_from_controller": {
                    "translation": [0.2, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )
    )
    packet = PosePacket.decode(json.dumps(state_payload()).encode())

    transform = ViveCalibration.load(path).map_from_base(packet)

    assert transform[:3, 3] == pytest.approx([10.8, 2.0, 3.0])
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)


def test_calibration_rejects_zero_quaternion(tmp_path) -> None:
    path = tmp_path / "vive_calibration.json"
    path.write_text(
        json.dumps(
            {
                "map_from_lighthouse": {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 0.0],
                },
                "base_from_controller": {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="map_from_lighthouse.*nonzero"):
        ViveCalibration.load(path)
