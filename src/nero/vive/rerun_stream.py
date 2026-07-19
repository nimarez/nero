"""Stream Lighthouse-tracked poses into a Rerun viewer.

Renders each tracked device as a live 3D coordinate frame plus a trajectory
trail. The viewer usually runs on a laptop while this runs on the robot/host the
tracked device is wired to, connected over a reverse SSH tunnel::

    # laptop:  rerun --port 9876
    # laptop:  ssh -R 9876:localhost:9876 user@host
    # host:    uv run nero-vive-rerun

``rerun-sdk`` is an optional dependency (the ``eval`` group), so it is imported
lazily and this module stays importable without it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any, TextIO

import numpy as np

from nero.vive.pose_source import DEFAULT_SURVIVE_ARGS, PoseSource, TimedPose, VivePoseSource

logger = logging.getLogger(__name__)

DEFAULT_GRPC_URL = "rerun+http://127.0.0.1:9876/proxy"
DEFAULT_TRAIL_LEN = 1024
DEFAULT_AXIS_LENGTH = 0.15  # metres
DEFAULT_LATEST_FILE = "/run/nero/vive_pose.json"
DEFAULT_GRID_EXTENT = 2.0  # metres from world origin
DEFAULT_GRID_SPACING = 0.25  # metres
WORLD_ENTITY = "world"
LOG_EVERY = 100  # progress-log cadence, in pose updates

_AXES = np.eye(3, dtype=np.float32)  # unit X, Y, Z in the device's local frame
_AXIS_COLORS = np.array([[255, 60, 60], [60, 255, 60], [80, 120, 255]], dtype=np.uint8)

_REMOTE_FILE_STREAM = r"""
import json
import sys
import time

path = sys.argv[1]
last = None
while True:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        key = (
            payload.get("controller_id"),
            payload.get("sequence"),
            payload.get("tracking_valid"),
        )
        if key != last:
            print(json.dumps(payload, separators=(",", ":")), flush=True)
            last = key
    except (OSError, ValueError):
        pass
    time.sleep(1.0 / 60.0)
"""


def _import_rerun() -> Any:
    """Import ``rerun`` lazily so the module loads without the optional dep."""
    try:
        import rerun as rr
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "rerun-sdk is not installed. Install the optional group: `uv sync --group eval`."
        ) from error
    return rr


def _pose_from_json_line(line: str) -> TimedPose:
    """Decode one receiver state line into the renderer's neutral pose type."""
    payload = json.loads(line)
    name = payload["controller_id"]
    position = np.asarray(payload["position"], dtype=np.float64)
    quaternion = np.asarray(payload["quaternion_xyzw"], dtype=np.float64)
    if not isinstance(name, str) or not name:
        raise ValueError("controller_id must be a non-empty string")
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("invalid Vive pose dimensions")
    return TimedPose(
        name=name,
        timestamp=float(payload["timestamp"]),
        position=position,
        quaternion_xyzw=quaternion,
        tracking_valid=payload.get("tracking_valid") is True,
    )


class JsonLinePoseSource:
    """Read receiver state from a line-delimited JSON stream."""

    def __init__(self, lines: TextIO) -> None:
        self._lines = lines

    def poll(self) -> Iterator[TimedPose]:
        for line in self._lines:
            if not line.strip():
                continue
            try:
                yield _pose_from_json_line(line)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                logger.warning("ignoring invalid pose state: %s", error)


class SshLatestPoseSource:
    """Subscribe to the atomic latest-pose file on a remote host over SSH."""

    def __init__(self, host: str, path: str = DEFAULT_LATEST_FILE) -> None:
        self._host = host
        self._path = path

    def poll(self) -> Iterator[TimedPose]:
        logger.info("Subscribing to %s:%s over SSH", self._host, self._path)
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", self._host, "python3", "-u", "-", self._path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(_REMOTE_FILE_STREAM)
        process.stdin.close()
        try:
            yield from JsonLinePoseSource(process.stdout).poll()
        finally:
            terminated_by_us = process.poll() is None
            if terminated_by_us:
                process.terminate()
            return_code = process.wait(timeout=5)
            if return_code and not terminated_by_us:
                raise RuntimeError(f"SSH pose subscription exited with status {return_code}")


def _log_pose(rr: Any, pose: TimedPose, trail: deque[np.ndarray], axis_length: float) -> None:
    """Log one device's current frame and its accumulated trajectory."""
    entity = f"{WORLD_ENTITY}/{pose.name}"
    rr.log(
        entity,
        rr.Transform3D(
            translation=pose.position,
            quaternion=rr.Quaternion(xyzw=pose.quaternion_xyzw),
        ),
    )
    # A Transform3D alone draws nothing -- it is only a coordinate transform. Log
    # geometry *under* it so the device is actually visible, oriented by it.
    rr.log(f"{entity}/axes", rr.Arrows3D(vectors=_AXES * axis_length, colors=_AXIS_COLORS))
    rr.log(f"{entity}/origin", rr.Points3D([[0.0, 0.0, 0.0]], radii=0.02))

    # Trails belong in world space: nesting them under `entity` would re-apply
    # the device's current transform to every past point and smear the path.
    trail.append(pose.position)
    if len(trail) > 1:
        rr.log(
            f"{WORLD_ENTITY}/trails/{pose.name}",
            rr.LineStrips3D([np.asarray(trail, dtype=np.float32)]),
        )


def _log_world_reference(rr: Any) -> None:
    """Give the live pose a visible metric frame of reference."""
    rr.log(
        f"{WORLD_ENTITY}/origin_axes",
        rr.Arrows3D(vectors=_AXES * 0.3, colors=_AXIS_COLORS),
        static=True,
    )
    coordinates = np.arange(
        -DEFAULT_GRID_EXTENT,
        DEFAULT_GRID_EXTENT + DEFAULT_GRID_SPACING / 2,
        DEFAULT_GRID_SPACING,
    )
    strips = []
    for coordinate in coordinates:
        strips.append(
            np.array(
                [[-DEFAULT_GRID_EXTENT, coordinate, 0.0], [DEFAULT_GRID_EXTENT, coordinate, 0.0]],
                dtype=np.float32,
            )
        )
        strips.append(
            np.array(
                [[coordinate, -DEFAULT_GRID_EXTENT, 0.0], [coordinate, DEFAULT_GRID_EXTENT, 0.0]],
                dtype=np.float32,
            )
        )
    rr.log(
        f"{WORLD_ENTITY}/ground_grid",
        rr.LineStrips3D(strips, colors=[130, 130, 130, 90], radii=0.002),
        static=True,
    )


def stream(
    source: PoseSource,
    url: str = DEFAULT_GRPC_URL,
    app_id: str = "nero_vive",
    trail_len: int = DEFAULT_TRAIL_LEN,
    axis_length: float = DEFAULT_AXIS_LENGTH,
    spawn: bool = False,
) -> None:
    """Consume poses from ``source`` and log each one to Rerun."""
    rr = _import_rerun()
    rr.init(app_id)
    if spawn:
        logger.info("Opening the local Rerun viewer")
        rr.spawn()
    else:
        logger.info("Connecting to Rerun viewer at %s", url)
        rr.connect_grpc(url)
    rr.log(WORLD_ENTITY, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_world_reference(rr)

    trails: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=trail_len))
    tracking_state: dict[str, bool] = {}
    for frame, pose in enumerate(source.poll(), start=1):
        # `set_time_sequence` is the 0.22 spelling; later versions renamed it to
        # `set_time(..., sequence=)`. The project pins rerun-sdk==0.22.1.
        rr.set_time_sequence("frame", frame)
        if tracking_state.get(pose.name) != pose.tracking_valid:
            status = (
                f"# {pose.name}: TRACKING OK\nReceiving a valid Lighthouse optical pose."
                if pose.tracking_valid
                else f"# {pose.name}: TRACKING LOST\nWaiting for a valid Lighthouse optical pose."
            )
            rr.log("tracking_status", rr.TextDocument(status, media_type="text/markdown"))
            tracking_state[pose.name] = pose.tracking_valid
        if not pose.tracking_valid:
            continue
        _log_pose(rr, pose, trails[pose.name], axis_length)
        if frame % LOG_EVERY == 0:
            x, y, z = pose.position
            logger.info(
                "logged %d poses | %s at x=%+.3f y=%+.3f z=%+.3f m", frame, pose.name, x, y, z
            )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse our flags; unrecognised ones are forwarded to libsurvive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_GRPC_URL, help="Rerun viewer gRPC proxy URL.")
    parser.add_argument("--app-id", default="nero_vive")
    parser.add_argument("--trail-len", type=int, default=DEFAULT_TRAIL_LEN)
    parser.add_argument("--axis-length", type=float, default=DEFAULT_AXIS_LENGTH)
    parser.add_argument(
        "--ssh-host",
        help="Subscribe to the POS latest-pose endpoint over SSH (for example: pos).",
    )
    parser.add_argument("--latest-file", default=DEFAULT_LATEST_FILE)
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Open a native Rerun viewer locally instead of connecting to --url.",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, survive_args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    source: PoseSource
    if args.ssh_host:
        if survive_args:
            parser_error = "libsurvive arguments cannot be combined with --ssh-host"
            logger.error(parser_error)
            return 2
        source = SshLatestPoseSource(args.ssh_host, args.latest_file)
    else:
        source = VivePoseSource(survive_args or DEFAULT_SURVIVE_ARGS)
    try:
        stream(
            source,
            url=args.url,
            app_id=args.app_id,
            trail_len=args.trail_len,
            axis_length=args.axis_length,
            spawn=args.spawn,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
