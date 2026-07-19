#!/usr/bin/env python3
"""FALLBACK Vive→UDP bridge for the Pi.

NOTE: the CANONICAL Vive stream is **PR #6** — `nero.vive.udp_transport` (`nero-vive-udp-send` on the
Pi → `nero-vive-udp-receive` on jscore → /run/nero/vive_pose.json). `follow_circle.py` reads that file.
Prefer PR #6. Keep this only as a minimal tailnet fallback if the M2 services aren't available: it
reads the Lighthouse pose via libsurvive (nero.vive.pose_source) and forwards JSON {x,y,yaw,t} over UDP.

    PYTHONPATH=<nero>/src:<libsurvive>/bindings/python LD_LIBRARY_PATH=<libsurvive>/bin \
      python vive_bridge.py --host <jscore-ip> --device TR0

Device names (libsurvive): TR0 = Vive Tracker, WW0 = wired controller, LH0/LH1 = base stations.
"""
from __future__ import annotations
import argparse
import json
import math
import socket
import sys


def yaw_from_quat_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="jscore IP (UDP target)")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--device", default="TR0", help="libsurvive device (TR0 tracker / WW0 controller)")
    a = ap.parse_args()

    from nero.vive.pose_source import VivePoseSource   # needs pysurvive on the Pi

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    src = VivePoseSource()
    sent = 0
    print(f"[fallback] forwarding {a.device} -> udp://{a.host}:{a.port} (prefer PR #6)", file=sys.stderr)
    for tp in src.poll():
        if a.device and tp.name != a.device:
            continue
        x, y, _z = (float(v) for v in tp.position)
        yaw = yaw_from_quat_xyzw(tp.quaternion_xyzw)
        sock.sendto(json.dumps({"x": x, "y": y, "yaw": yaw, "t": tp.timestamp}).encode(), (a.host, a.port))
        sent += 1
        if sent % 120 == 0:
            print(f"sent {sent}: {a.device} x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
