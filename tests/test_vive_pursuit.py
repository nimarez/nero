from types import SimpleNamespace

import numpy as np
import pytest

from nero.agents.vive_pursuit_agent import parse_args, run_agent
from nero.navigation.vive_pursuit import (
    VivePathTracker,
    VivePursuitController,
    object_approach_pose,
    plan_object_approach,
)


def test_object_approach_pose_is_in_front_and_faces_object() -> None:
    approach = object_approach_pose(np.array([0.0, 0.0, -np.pi / 2]), 0.5)

    assert approach[:2] == pytest.approx([0.0, -0.5])
    assert [np.cos(approach[2]), np.sin(approach[2])] == pytest.approx([0.0, 1.0])


def test_object_approach_path_has_requested_terminal_tangent() -> None:
    path = plan_object_approach(
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -np.pi / 2]),
        0.5,
        spacing=0.02,
    )

    assert path.points[0] == pytest.approx([1.0, 1.0])
    assert path.points[-1] == pytest.approx([0.0, -0.5])
    final_tangent = path.points[-1] - path.points[-2]
    final_tangent /= np.linalg.norm(final_tangent)
    assert final_tangent == pytest.approx([0.0, 1.0], abs=0.02)


def test_object_approach_path_detours_without_cusps_for_opposing_headings() -> None:
    path = plan_object_approach(
        np.array([0.0, 0.0, np.pi]),
        np.array([1.5, 0.0, 0.0]),
        0.5,
        spacing=0.02,
    )

    segments = np.diff(path.points, axis=0)
    directions = segments / np.linalg.norm(segments, axis=1)[:, None]
    assert np.max(np.abs(path.points[:, 1])) >= 0.5
    assert np.all(np.sum(directions[1:] * directions[:-1], axis=1) > 0.0)
    assert directions[0] == pytest.approx([-1.0, 0.0], abs=0.03)
    assert directions[-1] == pytest.approx([-1.0, 0.0], abs=0.03)


def test_path_tracker_never_moves_its_nearest_index_backward() -> None:
    tracker = VivePathTracker(np.column_stack((np.arange(6, dtype=float), np.zeros(6))))

    assert tracker.lookahead(np.array([3.1, 0.0]), 1.0) == pytest.approx([4.0, 0.0])
    assert tracker.index == 3
    tracker.lookahead(np.array([0.0, 0.0]), 1.0)
    assert tracker.index == 3


def test_path_speed_uses_remaining_route_instead_of_short_lookahead() -> None:
    command = VivePursuitController().compute_path_command(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, 0.0]),
        np.array([5.5, 0.0, 0.0]),
        remaining_distance=5.5,
    )

    assert command.linear_x == pytest.approx(0.1)


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
        parse_args(["--goal", "1", "2", "0"])


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
    telemetry = SimpleNamespace(
        plans=[], publish_plan=lambda plan, timestamp: telemetry.plans.append(plan)
    )
    args = parse_args(
        [
            "--goal",
            "0.5",
            "0",
            str(np.pi),
            "--stand-off",
            "0.5",
            "--acknowledge-blind-motion",
        ]
    )

    run_agent(robot, source, args, telemetry=telemetry, sleep=lambda _duration: None)

    assert commands[-1] == (0.0, 0.0, 0.0)
    assert telemetry.plans == []
