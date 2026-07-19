from types import SimpleNamespace

import numpy as np
import pytest

from nero.agents.vive_pursuit_agent import parse_args, run_agent
from nero.navigation.vive_pursuit import VivePursuitController


def test_vive_pursuit_drives_toward_world_goal() -> None:
    command = VivePursuitController().compute_command(
        np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0]), 0.5
    )

    assert command.linear_x > 0
    assert command.angular_z == pytest.approx(0.0)


def test_vive_pursuit_turns_before_goal_behind_robot() -> None:
    command = VivePursuitController().compute_command(
        np.array([0.0, 0.0, 0.0]), np.array([-2.0, 0.0]), 0.5
    )

    assert command.linear_x == 0
    assert abs(command.angular_z) > 0


def test_vive_pursuit_faces_box_at_stand_off() -> None:
    controller = VivePursuitController()
    command = controller.compute_command(np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.0]), 0.5)

    assert command.linear_x == 0
    assert command.angular_z < 0
    assert not controller.has_arrived(np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.0]), 0.5)
    assert controller.has_arrived(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0]), 0.5)


def test_blind_acknowledgement_is_required() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--goal", "1", "2"])


def test_run_agent_stops_when_pose_is_unavailable() -> None:
    now = [0.0]
    commands = []
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: commands.append((0.0, 0.0, 0.0)),
        set_velocity=lambda *values: commands.append(values),
    )
    source = SimpleNamespace(current_pose=lambda: None)
    args = parse_args(
        [
            "--goal",
            "1",
            "0",
            "--startup-timeout",
            "0.1",
            "--max-runtime",
            "1",
            "--acknowledge-blind-motion",
        ]
    )

    with pytest.raises(RuntimeError, match="unavailable or stale"):
        run_agent(
            robot,
            source,
            args,
            monotonic=lambda: now[0],
            sleep=lambda duration: now.__setitem__(0, now[0] + duration),
        )

    assert commands and all(command == (0.0, 0.0, 0.0) for command in commands)


def test_run_agent_reaches_goal_and_stops() -> None:
    commands = []
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: commands.append((0.0, 0.0, 0.0)),
        set_velocity=lambda *values: commands.append(values),
    )
    source = SimpleNamespace(current_pose=lambda: np.array([0.0, 0.0, 0.0]))
    args = parse_args(["--goal", "0.5", "0", "--stand-off", "0.5", "--acknowledge-blind-motion"])

    run_agent(robot, source, args, sleep=lambda _duration: None)

    assert commands[-1] == (0.0, 0.0, 0.0)
