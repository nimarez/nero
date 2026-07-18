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
import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np

from nero.vive.pose_source import DEFAULT_SURVIVE_ARGS, PoseSource, TimedPose, VivePoseSource

logger = logging.getLogger(__name__)

DEFAULT_GRPC_URL = "rerun+http://127.0.0.1:9876/proxy"
DEFAULT_TRAIL_LEN = 1024
DEFAULT_AXIS_LENGTH = 0.15  # metres
WORLD_ENTITY = "world"
LOG_EVERY = 100  # progress-log cadence, in pose updates

_AXES = np.eye(3, dtype=np.float32)  # unit X, Y, Z in the device's local frame
_AXIS_COLORS = np.array([[255, 60, 60], [60, 255, 60], [80, 120, 255]], dtype=np.uint8)


def _import_rerun() -> Any:
    """Import ``rerun`` lazily so the module loads without the optional dep."""
    try:
        import rerun as rr
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "rerun-sdk is not installed. Install the optional group: "
            "`uv sync --group eval`."
        ) from error
    return rr


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


def stream(
    source: PoseSource,
    url: str = DEFAULT_GRPC_URL,
    app_id: str = "nero_vive",
    trail_len: int = DEFAULT_TRAIL_LEN,
    axis_length: float = DEFAULT_AXIS_LENGTH,
) -> None:
    """Consume poses from ``source`` and log each one to Rerun."""
    rr = _import_rerun()
    rr.init(app_id)
    logger.info("Connecting to Rerun viewer at %s", url)
    rr.connect_grpc(url=url)
    rr.log(WORLD_ENTITY, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    trails: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=trail_len))
    for frame, pose in enumerate(source.poll(), start=1):
        # `set_time_sequence` is the 0.22 spelling; later versions renamed it to
        # `set_time(..., sequence=)`. The project pins rerun-sdk==0.22.1.
        rr.set_time_sequence("frame", frame)
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
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, survive_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source: PoseSource = VivePoseSource(survive_args or DEFAULT_SURVIVE_ARGS)
    try:
        stream(
            source,
            url=args.url,
            app_id=args.app_id,
            trail_len=args.trail_len,
            axis_length=args.axis_length,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
