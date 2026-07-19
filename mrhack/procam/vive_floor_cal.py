#!/usr/bin/env python3
"""Vive -> floor SE(2) calibration: touch the 4 tags with the Vive controller, solve the transform.

Run on jscore with the Vive UDP stream live (vive_bridge.py running on the Pi):
    PYTHONPATH=<nero>/src ~/Prismos-x/venv/bin/python vive_floor_cal.py

For each tag it waits for you to rest the controller on that tag, then samples the Vive pose. With
the 4 (vive x,y) <-> (floor x,y) correspondences it solves the rigid SE(2) via
mrhack.localization.frame_align.umeyama_se2 (the same fit our SLAM<->Vive alignment uses) and writes
/tmp/vive_floor.json {"R":2x2,"t":2}, which follow_circle.py loads.

The 4 tag floor positions (metres) come from the same tag rectification procam_true uses, so the
Vive frame and the projected circle share one floor frame.
"""
from __future__ import annotations
import argparse
import json
import socket
import sys

import numpy as np

from mrhack.localization.frame_align import umeyama_se2

UDP_PORT = 9101
TAG_SIZE_M = 0.15


def read_vive(port=UDP_PORT, samples=15, timeout=5.0):
    """Average a few UDP pose samples -> a steady (x, y)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.settimeout(timeout)
    xs, ys = [], []
    while len(xs) < samples:
        data, _ = s.recvfrom(8192)
        d = json.loads(data)
        xs.append(float(d["x"]))
        ys.append(float(d["y"]))
    s.close()
    return float(np.mean(xs)), float(np.mean(ys))


def tag_floor_positions(tag_size_m=TAG_SIZE_M):
    """Metric floor positions of tags 0..3 (metres), from the ArUco square rectification."""
    import cv2
    from context_snippets import capture_color, detect_tags
    from procam_true import build_cam_to_floor, EXPECTED_IDS
    raw = detect_tags(capture_color())
    tags = {i: np.asarray(raw[i], float).reshape(4, 2) for i in EXPECTED_IDS if i in raw}
    if len(tags) < 4:
        raise SystemExit(f"need tags 0-3 to set the floor frame; got {sorted(tags)}")
    H, ref, score = build_cam_to_floor(tags)
    centers = np.array([tags[i].mean(0) for i in EXPECTED_IDS], np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(centers, H).reshape(-1, 2) * tag_size_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", help='JSON: 4 [x,y] tag floor positions (m), order 0,1,2,3. Omit to auto-detect.')
    ap.add_argument("--tag-size-m", type=float, default=TAG_SIZE_M)
    a = ap.parse_args()

    floor = np.array(json.loads(a.floor), float) if a.floor else tag_floor_positions(a.tag_size_m)
    print("floor tag positions (m):", floor.round(3).tolist())

    vive = []
    for i in range(4):
        input(f"  rest the Vive controller on tag {i} -> {floor[i].round(3).tolist()} m, then press Enter... ")
        vx, vy = read_vive()
        vive.append([vx, vy])
        print(f"    vive = ({vx:+.3f}, {vy:+.3f})")
    vive = np.array(vive)

    R, t, yaw, rms = umeyama_se2(vive, floor)   # vive -> floor
    json.dump({"R": R.tolist(), "t": t.tolist(), "yaw": yaw, "rms": rms}, open("/tmp/vive_floor.json", "w"))
    print(f"vive->floor SE(2): yaw={np.degrees(yaw):+.1f} deg, fit rms={rms * 1000:.1f} mm "
          f"-> /tmp/vive_floor.json", file=sys.stderr)


if __name__ == "__main__":
    main()
