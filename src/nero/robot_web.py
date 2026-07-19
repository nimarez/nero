"""Run a physical Nero policy and its browser visualization on the robot."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence

logger = logging.getLogger(__name__)

POLICY_MODULES = {
    "orb-slam": "nero.agents.orb_slam_agent",
    "pure-pursuit": "nero.agents.pure_pursuit_agent",
}


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Nero on the robot with a browser-hosted Rerun viewer",
    )
    parser.add_argument("--policy", choices=tuple(POLICY_MODULES), default="pure-pursuit")
    parser.add_argument("--camera-start-timeout", type=float, default=120.0)
    parser.add_argument("--skip-camera-preflight", action="store_true")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-path", default="/rerun")
    parser.add_argument(
        "--advertise-host",
        default=os.getenv("NERO_ROBOT_HOST", "10.2.1.130"),
        help="Robot hostname or IP printed for the browser URL",
    )
    parser.add_argument("--viewer-port", type=int, default=8081)
    parser.add_argument("--websocket-port", type=int, default=9877)
    parser.add_argument("--server-memory-limit", default="256MB")
    parser.add_argument("--debug", action="store_true")
    args, policy_args = parser.parse_known_args(argv)
    if args.camera_start_timeout <= 0:
        parser.error("--camera-start-timeout must be positive")
    for name in ("web_port", "viewer_port", "websocket_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if len({args.web_port, args.viewer_port, args.websocket_port}) != 3:
        parser.error("web, viewer, and WebSocket ports must be different")
    if args.web_path.rstrip("/") in {"", "/"}:
        parser.error("--web-path must name a path such as /rerun")
    if "--no-ros-observability" in policy_args:
        parser.error("robot-hosted Rerun requires ROS observability")
    return args, policy_args


def _module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _wait_for_port(process: subprocess.Popen, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Rerun bridge exited during startup with status {return_code}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Rerun web gateway did not listen on port {port} within {timeout:g}s")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_rerun_web_bridge(
    *,
    web_port: int = 8080,
    web_path: str = "/rerun",
    viewer_port: int = 8081,
    websocket_port: int = 9877,
    server_memory_limit: str = "256MB",
    debug: bool = False,
    ensure_viz_extra: bool = False,
    startup_timeout: float | None = None,
) -> subprocess.Popen:
    """Start the robot-local ROS-to-Rerun bridge and wait for its web gateway."""
    timeout = (
        120.0 if ensure_viz_extra else 15.0
    ) if startup_timeout is None else float(startup_timeout)
    if timeout <= 0:
        raise ValueError("Rerun startup timeout must be positive")
    if ensure_viz_extra:
        bridge_command = [
            "uv",
            "run",
            "--locked",
            "--extra",
            "viz",
            "nero-rerun",
        ]
    else:
        bridge_command = _module_command("nero.observability.rerun_bridge")
    bridge_command.extend(
        [
            "--serve-web",
            "--web-port",
            str(web_port),
            "--web-path",
            web_path,
            "--viewer-port",
            str(viewer_port),
            "--websocket-port",
            str(websocket_port),
            "--server-memory-limit",
            server_memory_limit,
        ]
    )
    if debug:
        bridge_command.append("--debug")

    bridge = subprocess.Popen(bridge_command)
    try:
        _wait_for_port(bridge, web_port, timeout=timeout)
    except Exception:
        _stop_process(bridge)
        raise
    return bridge


def main(argv: Sequence[str] | None = None) -> None:
    args, policy_args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.skip_camera_preflight:
        preflight = subprocess.run(
            _module_command(
                "nero.k1_preflight",
                "--timeout",
                f"{args.camera_start_timeout:g}",
            ),
            check=False,
        )
        if preflight.returncode:
            raise SystemExit(preflight.returncode)

    bridge = None
    try:
        bridge = start_rerun_web_bridge(
            web_port=args.web_port,
            web_path=args.web_path,
            viewer_port=args.viewer_port,
            websocket_port=args.websocket_port,
            server_memory_limit=args.server_memory_limit,
            debug=args.debug,
        )
        print(
            f"Rerun: http://{args.advertise_host}:{args.web_port}{args.web_path}",
            flush=True,
        )
        policy_command = _module_command(
            POLICY_MODULES[args.policy],
            "--no-display",
            "--command-source",
            "terminal",
            "--no-web-rerun",
            *(["--debug"] if args.debug else []),
            *policy_args,
        )
        completed = subprocess.run(policy_command, check=False)
        if completed.returncode not in (0, 130, -2, -15):
            raise SystemExit(completed.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_process(bridge)


if __name__ == "__main__":
    main()
