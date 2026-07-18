"""Read 6-DoF poses from Lighthouse-tracked devices via libsurvive.

No SteamVR and no headset are required. Lighthouse base stations only *emit* IR
sweeps and have no data connection to the host; every pose is computed from the
photodiodes on a *tracked device* (a Vive controller or Tracker) connected over
USB. A base station on its own produces nothing.

``pysurvive`` ships inside the libsurvive source tree rather than on PyPI, so it
is imported lazily: importing this module never fails on a machine (or in CI)
where libsurvive is not built.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# One base station is a valid (if fragile) setup; libsurvive otherwise waits for two.
DEFAULT_SURVIVE_ARGS: tuple[str, ...] = ("--lighthouse-count", "1")


@dataclass(frozen=True, slots=True)
class TimedPose:
    """A single timestamped 6-DoF pose in the tracking system's world frame."""

    name: str
    timestamp: float
    position: np.ndarray  # (3,) metres: x, y, z
    quaternion_xyzw: np.ndarray  # (4,) x, y, z, w


class PoseSource(Protocol):
    """Any provider of timestamped 6-DoF poses (Lighthouse, mocap, sim oracle)."""

    def poll(self) -> Iterator[TimedPose]:
        """Yield poses as they arrive, until the source stops."""
        ...


def _decode_name(obj: Any) -> str:
    """libsurvive object names come back as bytes (e.g. ``b'WW0'``)."""
    raw = obj.Name()
    return raw.decode("ascii", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)


def _to_timed_pose(name: str, raw_pose: Any) -> TimedPose:
    """Convert a libsurvive pose into our convention.

    ``SimpleObject.Pose()`` returns ``(SurvivePose, timecode)`` on current
    bindings and a bare ``SurvivePose`` on older ones; both are accepted.
    libsurvive orders quaternions ``(w, x, y, z)`` whereas numpy/scipy/Rerun all
    expect ``(x, y, z, w)``.
    """
    if isinstance(raw_pose, tuple):
        survive_pose, timestamp = raw_pose[0], float(raw_pose[1])
    else:
        survive_pose, timestamp = raw_pose, 0.0

    position = np.array(
        [survive_pose.Pos[0], survive_pose.Pos[1], survive_pose.Pos[2]], dtype=np.float64
    )
    w, x, y, z = (survive_pose.Rot[i] for i in range(4))
    return TimedPose(
        name=name,
        timestamp=timestamp,
        position=position,
        quaternion_xyzw=np.array([x, y, z, w], dtype=np.float64),
    )


class VivePoseSource:
    """:class:`PoseSource` backed by libsurvive Lighthouse tracking.

    Device names follow libsurvive's convention: ``WW0`` is a wired watchman (a
    controller/tracker on a USB cable), ``WM0`` a wireless one, ``TR0`` a Vive
    Tracker, and ``LH0``/``LH1`` the base stations themselves.
    """

    def __init__(self, survive_args: Sequence[str] = DEFAULT_SURVIVE_ARGS) -> None:
        self._survive_args = tuple(survive_args)
        self._context: Any | None = None

    def _ensure_context(self) -> Any:
        """Create the libsurvive context, importing ``pysurvive`` on first use."""
        if self._context is not None:
            return self._context
        try:
            import pysurvive
        except ImportError as error:  # pragma: no cover - depends on local build
            raise RuntimeError(
                "pysurvive is not importable. Build libsurvive and set both "
                "PYTHONPATH=<libsurvive>/bindings/python and "
                "LD_LIBRARY_PATH=<libsurvive>/bin. See src/nero/vive/README.md."
            ) from error

        # libsurvive parses this list like argv; element 0 is the program name.
        self._context = pysurvive.SimpleContext(["nero-vive", *self._survive_args])
        startup = [_decode_name(obj) for obj in self._context.Objects()]
        logger.info("libsurvive objects at startup: %s", ", ".join(startup) or "(none yet)")
        return self._context

    def poll(self) -> Iterator[TimedPose]:
        """Yield poses as libsurvive solves them (blocks between updates)."""
        context = self._ensure_context()
        while context.Running():
            updated = context.NextUpdated()
            if not updated:
                continue
            yield _to_timed_pose(_decode_name(updated), updated.Pose())
