import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from nero.agents.vive_pursuit_agent import parse_args, run_agent
from nero.vive.projector_navigation import ProjectorNavigationSource


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _state(*, goal=(1.0, 2.0, 0.5), heading_calibrated=True):
    return {
        "version": 1,
        "frame_id": "room_floor",
        "robot_pose": {
            "x": 0.1,
            "y": -0.2,
            "yaw": 0.3,
            "valid": True,
            "heading_calibrated": heading_calibrated,
            "frame_id": "room_floor",
        },
        "goal_pose": {
            "x": goal[0],
            "y": goal[1],
            "yaw": goal[2],
            "frame_id": "room_floor",
        },
    }


def test_projector_navigation_source_validates_and_expires_snapshot():
    now = [5.0]
    source = ProjectorNavigationSource(
        "http://projector:8765",
        stale_after_s=0.25,
        monotonic=lambda: now[0],
        opener=lambda *_args, **_kwargs: _Response(json.dumps(_state()).encode()),
        autostart=False,
    )

    source.poll_once()
    robot, goal = source.current_navigation()
    assert robot == pytest.approx([0.1, -0.2, 0.3])
    assert goal == pytest.approx([1.0, 2.0, 0.5])
    now[0] += 0.26
    assert source.current_navigation() is None


def test_projector_navigation_requires_heading_calibration():
    source = ProjectorNavigationSource(
        "http://projector:8765",
        opener=lambda *_args, **_kwargs: _Response(
            json.dumps(_state(heading_calibrated=False)).encode()
        ),
        autostart=False,
    )

    with pytest.raises(ValueError, match="heading is not calibrated"):
        source.poll_once()


def test_projector_navigation_posts_nero_trajectory():
    requests = []

    def open_request(request, **_kwargs):
        requests.append(request)
        return _Response(b"{}")

    source = ProjectorNavigationSource(
        "http://projector:8765", opener=open_request, autostart=False
    )
    source.publish_trajectory(np.array([[0.0, 0.0], [1.0, 2.0]]))

    payload = json.loads(requests[0].data)
    assert requests[0].full_url.endswith("/api/navigation/trajectory")
    assert payload == {
        "waypoints": [[0.0, 0.0], [1.0, 2.0]],
        "source": "nero-vive-pursuit",
    }


def test_vive_agent_uses_projector_goal_and_returns_planned_path():
    commands = []
    trajectories = []
    now = [0.0]
    poses = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]
    source = SimpleNamespace(
        current_navigation=lambda: (poses.pop(0), np.array([1.0, 0.0, np.pi])),
        publish_trajectory=lambda points: trajectories.append(points.copy()),
    )
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: commands.append((0.0, 0.0, 0.0)),
        set_velocity=lambda *values: commands.append(values),
    )
    args = parse_args(["--projector-url", "http://projector:8765", "--acknowledge-blind-motion"])

    run_agent(
        robot,
        source,
        args,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert len(trajectories) == 1
    assert trajectories[0][0] == pytest.approx([0.0, 0.0])
    assert trajectories[0][-1] == pytest.approx([0.5, 0.0])
    assert commands[-1] == (0.0, 0.0, 0.0)
