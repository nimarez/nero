"""HTC Vive / Lighthouse 6-DoF tracking for Nero.

Provides an external, sub-centimetre pose reference independent of ORB-SLAM3 --
useful as ground truth for SLAM evaluation, and as a live 3D view in Rerun.

``pysurvive`` and ``rerun`` are both imported lazily by the submodules, so this
package is importable without libsurvive built or the ``eval`` group installed.
"""

from __future__ import annotations

from nero.vive.pose_source import PoseSource, TimedPose, VivePoseSource
from nero.vive.udp_transport import PosePacket, PoseUdpPublisher, PoseUdpReceiver

__all__ = [
    "PosePacket",
    "PoseSource",
    "PoseUdpPublisher",
    "PoseUdpReceiver",
    "TimedPose",
    "VivePoseSource",
]
