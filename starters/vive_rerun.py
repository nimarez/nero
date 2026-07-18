#!/usr/bin/env python
"""STARTER (needs hardware) - HTC Vive tracker -> absolute pose -> Rerun.

The Vive base stations give the tracker drift-free absolute pose; this is our M2
localization backbone. Attach a Vive tracker to the robot, plug the receiver dongle
into the box (USB), then run this to stream the pose into Rerun.

Run (on the box / a machine with the Vive receiver):
    uv run --with openvr --with rerun-sdk python starters/vive_rerun.py

Primary path uses OpenVR (SteamVR must be running). Lightweight alternative:
libsurvive / pysurvive (no SteamVR) - see TODO at bottom.

STATUS: written, NOT hardware-verified. Calibrate the base-station->floor transform
once (align the Vive origin to the ArUco floor origin), then wire the RobotPose
publish into the mrhack bus.
"""
from __future__ import annotations
import math
import sys
import time


def mat34_to_xyzyaw(m):
    """OpenVR pose is a 3x4 [R|t]. Vive up-axis is y; the floor is the x-z plane.
    Return (x, y, z, yaw) with yaw about the up axis."""
    x, y, z = m[0][3], m[1][3], m[2][3]
    yaw = math.atan2(m[0][2], m[2][2])
    return x, y, z, yaw


def main():
    try:
        import openvr
    except ImportError:
        print("openvr missing. Run: uv run --with openvr --with rerun-sdk python starters/vive_rerun.py")
        sys.exit(1)
    import rerun as rr

    rr.init("mrhack_vive", spawn=False)
    rr.connect()  # connect to the Rerun viewer / Booster Studio's built-in Rerun log server

    vr = openvr.init(openvr.VRApplication_Other)
    print("OpenVR up. Watching for tracker/controller devices... (Ctrl-C to stop)")
    try:
        while True:
            poses = vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                p = poses[i]
                if not p.bPoseIsValid:
                    continue
                if vr.getTrackedDeviceClass(i) not in (
                    openvr.TrackedDeviceClass_GenericTracker,
                    openvr.TrackedDeviceClass_Controller,
                ):
                    continue
                x, y, z, yaw = mat34_to_xyzyaw(p.mDeviceToAbsoluteTracking)
                fx, fy = x, z  # map Vive (x, z) -> our floor frame (x, y)
                rr.log(f"vive/device_{i}", rr.Points3D([[x, y, z]], radii=0.03))
                rr.log(f"vive/device_{i}/pose", rr.Transform3D(translation=[x, y, z]))
                print(f"dev{i} floor=({fx:+.2f},{fy:+.2f}) yaw={math.degrees(yaw):+6.1f}deg")
                # TODO: apply base-station->floor calibration (align Vive origin to the ArUco floor origin).
                # TODO: publish RobotPose(x=fx, y=fy, yaw=yaw, t=time.time()) on the mrhack bus (M2).
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()

# --- Lightweight alternative (no SteamVR): libsurvive / pysurvive ---
#   ctx = pysurvive.init(sys.argv); while pysurvive.poll(ctx) == 0: ... get_pose(...)
#   github.com/collabora/libsurvive - open SDK, absolute pose from lighthouse sweeps over USB.
