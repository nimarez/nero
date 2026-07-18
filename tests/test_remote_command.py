import builtins
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import nero.remote_command as remote_command
import pytest


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
    assert command[:2] == ["ssh", "-t"]
    assert "ConnectTimeout=5" in command
    assert "ServerAliveInterval=5" in command
    assert "ServerAliveCountMax=3" in command
    assert command[-2] == "booster@robot.local"
    assert "nero-command-relay" in command[-1]
    assert "--ack-timeout 5" in command[-1]
    assert "run_rerun_bridge" not in command[-1]
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
    assert command[-2] == "booster@10.2.1.130"
    assert "run_rerun_bridge.sh" in command[-1]
    assert "10.2.7.101:9876" in command[-1]
    assert "kill -0" in command[-1]
    assert "relay_pid=$!" in command[-1]
    assert 'while kill -0 "$bridge_pid"' in command[-1]
    assert 'wait "$relay_pid"' in command[-1]
    assert "nero-rerun-bridge.log" in command[-1]
    assert "nero-command-relay" in command[-1]
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


def test_start_viewer_waits_until_the_port_is_ready(monkeypatch):
    viewer = SimpleNamespace(pid=42, poll=lambda: None)
    ports = iter([False, False, True])
    popen_calls = []
    monkeypatch.setattr(remote_command, "_port_is_open", lambda *args: next(ports))
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: popen_calls.append((command, kwargs)) or viewer,
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    assert remote_command._start_viewer(9876, "512MB") is viewer
    command, kwargs = popen_calls[0]
    assert command[-2:] == ["--port", "9876"]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["NERO_RERUN_MEMORY_LIMIT"] == "512MB"


def test_start_viewer_reports_early_process_failure(monkeypatch):
    viewer = SimpleNamespace(pid=42, poll=lambda: 9)
    stopped = []
    monkeypatch.setattr(remote_command, "_port_is_open", lambda *args: False)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(remote_command, "_stop_viewer", stopped.append)

    with pytest.raises(RuntimeError, match="status 9"):
        remote_command._start_viewer(9876, "4GB")

    assert stopped == [viewer]


def test_existing_viewer_is_reused_without_spawning(monkeypatch):
    monkeypatch.setattr(remote_command, "_port_is_open", lambda *args: True)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("must not spawn a second viewer"),
    )

    assert remote_command._start_viewer(9876, "4GB") is None


def test_stop_viewer_escalates_when_process_group_survives(monkeypatch):
    signals = []
    viewer = SimpleNamespace(pid=123)
    viewer.wait = lambda timeout: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("viewer", timeout)
    )
    clock = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal))
    )

    remote_command._stop_viewer(viewer)

    assert signals == [(123, signal.SIGTERM), (123, 0), (123, signal.SIGKILL)]


def test_stop_viewer_tolerates_macos_process_group_probe_denial(monkeypatch):
    signals = []
    viewer = SimpleNamespace(pid=123, wait=lambda timeout: None)

    def killpg(pid, sent_signal):
        signals.append((pid, sent_signal))
        if sent_signal == 0:
            raise PermissionError("process group changed ownership")

    monkeypatch.setattr(os, "killpg", killpg)

    remote_command._stop_viewer(viewer)

    assert signals == [(123, signal.SIGTERM), (123, 0)]


def test_ssh_failure_still_closes_owned_viewer(monkeypatch):
    viewer = SimpleNamespace()
    stopped = []
    monkeypatch.setattr(sys, "argv", ["nero-command", "--rerun-host", "127.0.0.1"])
    monkeypatch.setattr(remote_command, "_start_viewer", lambda *args: viewer)
    monkeypatch.setattr(remote_command, "_stop_viewer", stopped.append)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=255)
    )

    with pytest.raises(SystemExit) as error:
        remote_command.main()

    assert error.value.code == 255
    assert stopped == [viewer]


def test_main_quotes_remote_repo_and_socket(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nero-command",
            "--no-rerun",
            "--repo",
            "/tmp/repo with ' quote",
            "--socket",
            "/tmp/socket with space",
        ],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: calls.append(command) or SimpleNamespace(returncode=0),
    )

    remote_command.main()

    remote = calls[0][-1]
    assert "cd '/tmp/repo with '\"'\"' quote'" in remote
    assert "--socket '/tmp/socket with space'" in remote


def test_relay_round_trips_over_a_real_unix_socket(monkeypatch, capsys):
    socket_path = f"/tmp/nero-relay-{os.getpid()}-{time.time_ns()}.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    received = []

    def serve():
        client, _ = server.accept()
        with client:
            received.append(client.recv(4096))
            client.sendall(b"accepted\n")

    thread = threading.Thread(target=serve)
    thread.start()
    commands = iter(["chair", "quit"])
    monkeypatch.setattr(sys, "argv", ["nero-command-relay", "--socket", socket_path])
    monkeypatch.setattr(builtins, "input", lambda _: next(commands))

    remote_command.relay_main()
    thread.join(timeout=1)
    server.close()
    os.unlink(socket_path)

    assert not thread.is_alive()
    assert received == [b"chair\n"]
    assert "accepted: chair" in capsys.readouterr().out


def test_relay_times_out_if_policy_never_acknowledges(monkeypatch, capsys):
    socket_path = f"/tmp/nero-relay-timeout-{os.getpid()}-{time.time_ns()}.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    def serve():
        client, _ = server.accept()
        with client:
            client.recv(4096)
            time.sleep(0.05)

    thread = threading.Thread(target=serve)
    thread.start()
    commands = iter(["chair", "quit"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["nero-command-relay", "--socket", socket_path, "--ack-timeout", "0.01"],
    )
    monkeypatch.setattr(builtins, "input", lambda _: next(commands))

    remote_command.relay_main()
    thread.join(timeout=1)
    server.close()
    os.unlink(socket_path)

    assert not thread.is_alive()
    assert "Command was not sent: timed out" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["nero-command", "--rerun-port", "0"],
        ["nero-command", "--connect-timeout", "0"],
        ["nero-command", "--keepalive-interval", "0"],
        ["nero-command", "--keepalive-count", "0"],
        ["nero-command", "--ack-timeout", "0"],
    ],
)
def test_main_rejects_unsafe_network_settings(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        remote_command.main()
    assert error.value.code == 2


def test_relay_rejects_nonpositive_ack_timeout(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nero-command-relay", "--ack-timeout", "0"])
    with pytest.raises(SystemExit) as error:
        remote_command.relay_main()
    assert error.value.code == 2
