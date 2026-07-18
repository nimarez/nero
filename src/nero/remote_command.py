"""Interactive Mac-to-K1 navigation command relay."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path

DEFAULT_RERUN_PORT = 9876
DEFAULT_ACK_TIMEOUT = 5.0
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_KEEPALIVE_INTERVAL = 5
DEFAULT_KEEPALIVE_COUNT = 3


def _mac_address_for(robot_host: str) -> str:
    """Return the Mac address used to reach the robot without sending traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((robot_host, 9))
        return str(probe.getsockname()[0])
    finally:
        probe.close()


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _start_viewer(port: int, memory_limit: str) -> subprocess.Popen[bytes] | None:
    """Start a local viewer unless another one already owns its receive port."""
    if _port_is_open("127.0.0.1", port):
        print(f"Using the Rerun viewer already listening on port {port}.", flush=True)
        return None

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_rerun_viewer.sh"
    environment = os.environ.copy()
    environment["NERO_RERUN_MEMORY_LIMIT"] = memory_limit
    viewer = subprocess.Popen(
        [str(script), "--port", str(port)],
        cwd=repo_root,
        env=environment,
        start_new_session=True,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if _port_is_open("127.0.0.1", port):
            print(f"Rerun is listening on port {port}.", flush=True)
            return viewer
        return_code = viewer.poll()
        if return_code is not None:
            _stop_viewer(viewer)
            raise RuntimeError(f"Rerun viewer exited with status {return_code}")
        time.sleep(0.1)
    _stop_viewer(viewer)
    raise RuntimeError(f"Rerun did not begin listening on port {port}")


def _stop_viewer(viewer: subprocess.Popen[bytes] | None) -> None:
    if viewer is None:
        return
    # The native macOS viewer may outlive its CLI launcher. It inherits the
    # dedicated process group created above, so stop the whole owned group.
    try:
        os.killpg(viewer.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        viewer.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.killpg(viewer.pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    try:
        os.killpg(viewer.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def relay_main() -> None:
    """Run on the K1 and relay SSH stdin into the agent's Unix socket."""
    parser = argparse.ArgumentParser(
        description="Relay object names to a local Nero agent"
    )
    parser.add_argument("--socket", default="/tmp/nero-navigation.sock")
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=DEFAULT_ACK_TIMEOUT,
        help="Seconds to wait for the robot policy to accept a command",
    )
    args = parser.parse_args()
    if args.ack_timeout <= 0:
        parser.error("--ack-timeout must be positive")

    print("Nero object command terminal. Type an object name, or 'quit'.", flush=True)
    while True:
        try:
            command = input("object> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not command:
            continue
        if command.lower() in {"quit", "exit", "q"}:
            return
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(args.ack_timeout)
        try:
            client.connect(args.socket)
            client.sendall((command + "\n").encode())
            acknowledgement = client.recv(128).decode(errors="replace").strip()
        except OSError as exc:
            print(f"Command was not sent: {exc}", flush=True)
            continue
        finally:
            client.close()
        print(f"{acknowledgement}: {command}", flush=True)


def main() -> None:
    """Start Rerun and open the interactive relay through the user's SSH client."""
    parser = argparse.ArgumentParser(
        description="Start live Rerun visualization and control a running Nero K1 policy"
    )
    parser.add_argument("--host", default=os.getenv("NERO_ROBOT_HOST", "10.2.1.130"))
    parser.add_argument("--user", default=os.getenv("NERO_ROBOT_USER", "booster"))
    parser.add_argument(
        "--repo",
        default=os.getenv("NERO_ROBOT_REPO", "/home/booster/Workspace/nero"),
    )
    parser.add_argument("--socket", default="/tmp/nero-navigation.sock")
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=DEFAULT_ACK_TIMEOUT,
        help="Seconds to wait for the robot policy to accept a command",
    )
    parser.add_argument(
        "--rerun-host",
        help="Mac address reachable from the robot (default: determine automatically)",
    )
    parser.add_argument("--rerun-port", type=int, default=DEFAULT_RERUN_PORT)
    parser.add_argument("--rerun-memory-limit", default="4GB")
    parser.add_argument(
        "--no-rerun", action="store_true", help="Open only the command terminal"
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="SSH connection timeout in seconds",
    )
    parser.add_argument(
        "--keepalive-interval",
        type=int,
        default=DEFAULT_KEEPALIVE_INTERVAL,
        help="Seconds between SSH keepalives",
    )
    parser.add_argument(
        "--keepalive-count",
        type=int,
        default=DEFAULT_KEEPALIVE_COUNT,
        help="Missed SSH keepalives before disconnecting",
    )
    args = parser.parse_args()
    if not 1 <= args.rerun_port <= 65535:
        parser.error("--rerun-port must be between 1 and 65535")
    if args.ack_timeout <= 0:
        parser.error("--ack-timeout must be positive")
    for name in ("connect_timeout", "keepalive_interval", "keepalive_count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    viewer = None
    try:
        if args.no_rerun:
            remote_command = (
                f"cd {shlex.quote(args.repo)} && uv run nero-command-relay "
                f"--socket {shlex.quote(args.socket)} "
                f"--ack-timeout {args.ack_timeout:g}"
            )
        else:
            viewer = _start_viewer(args.rerun_port, args.rerun_memory_limit)
            rerun_host = args.rerun_host or _mac_address_for(args.host)
            rerun_url = f"{rerun_host}:{args.rerun_port}"
            remote_script = "\n".join(
                (
                    f"cd {shlex.quote(args.repo)}",
                    f"NERO_RERUN_URL={shlex.quote(rerun_url)} "
                    "./scripts/run_rerun_bridge.sh >/tmp/nero-rerun-bridge.log 2>&1 &",
                    "bridge_pid=$!",
                    "relay_pid=",
                    'cleanup() { kill "$bridge_pid" ${relay_pid:+"$relay_pid"} '
                    '2>/dev/null || true; wait "$bridge_pid" '
                    '${relay_pid:+"$relay_pid"} 2>/dev/null || true; }',
                    "trap cleanup EXIT",
                    "trap 'exit 130' HUP INT TERM",
                    "sleep 0.5",
                    'if ! kill -0 "$bridge_pid" 2>/dev/null; then '
                    "cat /tmp/nero-rerun-bridge.log >&2; exit 1; fi",
                    f"uv run nero-command-relay --socket {shlex.quote(args.socket)} "
                    f"--ack-timeout {args.ack_timeout:g} <&0 &",
                    "relay_pid=$!",
                    'while kill -0 "$bridge_pid" 2>/dev/null && '
                    'kill -0 "$relay_pid" 2>/dev/null; do sleep 0.25; done',
                    'if ! kill -0 "$bridge_pid" 2>/dev/null; then '
                    "cat /tmp/nero-rerun-bridge.log >&2; exit 1; fi",
                    'wait "$relay_pid"',
                )
            )
            remote_command = f"bash -lc {shlex.quote(remote_script)}"
            print(f"Connecting the robot telemetry bridge to {rerun_url}.", flush=True)

        completed = subprocess.run(
            [
                "ssh",
                "-t",
                "-o",
                f"ConnectTimeout={args.connect_timeout}",
                "-o",
                f"ServerAliveInterval={args.keepalive_interval}",
                "-o",
                f"ServerAliveCountMax={args.keepalive_count}",
                f"{args.user}@{args.host}",
                remote_command,
            ],
            check=False,
        )
        if completed.returncode not in (0, 130):
            raise SystemExit(completed.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_viewer(viewer)


if __name__ == "__main__":
    main()
