import subprocess
import sys
from types import SimpleNamespace

import nero.remote_command as remote_command


def test_no_rerun_opens_only_the_remote_command_relay(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["nero-command", "--host", "robot.local", "--no-rerun"],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: calls.append((command, check))
        or SimpleNamespace(returncode=0),
    )

    remote_command.main()

    command, check = calls[0]
    assert command[:3] == ["ssh", "-t", "booster@robot.local"]
    assert "nero-command-relay" in command[3]
    assert "run_rerun_bridge" not in command[3]
    assert check is False


def test_default_command_owns_viewer_and_remote_bridge(monkeypatch):
    viewer = SimpleNamespace()
    calls = []
    stopped = []
    monkeypatch.setattr(sys, "argv", ["nero-command"])
    monkeypatch.setattr(remote_command, "_start_viewer", lambda port, limit: viewer)
    monkeypatch.setattr(remote_command, "_mac_address_for", lambda _: "10.2.7.101")
    monkeypatch.setattr(remote_command, "_stop_viewer", stopped.append)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: calls.append((command, check))
        or SimpleNamespace(returncode=0),
    )

    remote_command.main()

    command, _ = calls[0]
    assert command[:3] == ["ssh", "-t", "booster@10.2.1.130"]
    assert "run_rerun_bridge.sh" in command[3]
    assert "10.2.7.101:9876" in command[3]
    assert "nero-command-relay" in command[3]
    assert stopped == [viewer]


def test_stop_viewer_terminates_owned_process(monkeypatch):
    signals = []
    viewer = SimpleNamespace(
        pid=123,
        wait=lambda timeout: setattr(viewer, "wait_timeout", timeout),
    )
    def killpg(pid, sent_signal):
        signals.append((pid, sent_signal))
        if sent_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(remote_command.os, "killpg", killpg)

    remote_command._stop_viewer(viewer)

    assert signals == [(123, remote_command.signal.SIGTERM), (123, 0)]
    assert viewer.wait_timeout == 5
