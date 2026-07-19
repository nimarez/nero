from __future__ import annotations

import io
import json

import numpy as np

from nero.vive.rerun_stream import JsonLinePoseSource, _pose_from_json_line, parse_args


POSE_JSON = """{
  "version": 1,
  "sequence": 42,
  "timestamp": 1750000000.25,
  "controller_id": "WW0",
  "position": [1.0, 2.0, 3.0],
  "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
  "tracking_valid": true
}"""


def test_pose_from_receiver_json() -> None:
    pose = _pose_from_json_line(POSE_JSON)

    assert pose.name == "WW0"
    assert pose.timestamp == 1750000000.25
    np.testing.assert_array_equal(pose.position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(pose.quaternion_xyzw, [0.0, 0.0, 0.0, 1.0])


def test_json_line_source_skips_invalid_lines() -> None:
    compact_pose = json.dumps(json.loads(POSE_JSON), separators=(",", ":"))
    source = JsonLinePoseSource(io.StringIO(f"not-json\n{compact_pose}\n"))

    poses = list(source.poll())

    assert [pose.name for pose in poses] == ["WW0"]


def test_invalid_tracking_state_is_not_rendered() -> None:
    payload = json.loads(POSE_JSON)
    payload["tracking_valid"] = False

    pose = _pose_from_json_line(json.dumps(payload))

    assert pose.tracking_valid is False


def test_cli_selects_remote_endpoint_and_local_viewer() -> None:
    args, survive_args = parse_args(["--ssh-host", "pos", "--spawn"])

    assert args.ssh_host == "pos"
    assert args.spawn is True
    assert survive_args == []
