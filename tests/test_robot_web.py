from types import SimpleNamespace

import pytest

import nero.robot_web as robot_web


def test_robot_web_defaults_to_pure_pursuit_and_forwards_policy_options():
    args, policy_args = robot_web.parse_args(
        [
            "--skip-camera-preflight",
            "--disable-safety",
            "--object-backend",
            "aruco",
            "--aruco-map",
            "config/aruco_markers.json",
            "--aruco-dictionary",
            "DICT_4X4_50",
        ]
    )

    assert args.policy == "pure-pursuit"
    assert args.web_port == 8080
    assert args.web_path == "/rerun"
    assert args.advertise_host == "10.2.1.130"
    assert policy_args == [
        "--disable-safety",
        "--object-backend",
        "aruco",
        "--aruco-map",
        "config/aruco_markers.json",
        "--aruco-dictionary",
        "DICT_4X4_50",
    ]


def test_robot_web_rejects_disabling_required_observability():
    with pytest.raises(SystemExit):
        robot_web.parse_args(["--no-ros-observability"])


def test_robot_web_starts_bridge_then_terminal_policy_and_cleans_up(monkeypatch):
    calls = []

    class BridgeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate")
            self.returncode = -15

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return self.returncode

    bridge = BridgeProcess()
    monkeypatch.setattr(
        robot_web.subprocess,
        "Popen",
        lambda command: calls.append(("bridge", command)) or bridge,
    )
    monkeypatch.setattr(
        robot_web,
        "_wait_for_port",
        lambda process, port: calls.append(("web", port)),
    )
    monkeypatch.setattr(
        robot_web.subprocess,
        "run",
        lambda command, check: calls.append(("policy", command)) or SimpleNamespace(returncode=0),
    )

    robot_web.main(
        [
            "--skip-camera-preflight",
            "--policy",
            "pure-pursuit",
            "--object-backend",
            "aruco",
        ]
    )

    bridge_command = calls[0][1]
    policy_command = calls[2][1]
    assert bridge_command[1:4] == [
        "-m",
        "nero.observability.rerun_bridge",
        "--serve-web",
    ]
    assert policy_command[1:6] == [
        "-m",
        "nero.agents.pure_pursuit_agent",
        "--no-display",
        "--command-source",
        "terminal",
    ]
    assert policy_command[-2:] == ["--object-backend", "aruco"]
    assert calls[-2:] == ["terminate", ("wait", 5)]
