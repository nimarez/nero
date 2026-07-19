#!/usr/bin/env python3
"""Vive -> floor SE(2) calibration: touch the 4 tags with the Vive controller, solve the transform.

Run on jscore with the PR #6 Vive stream live (`nero-vive-udp-receive` writing /run/nero/vive_pose.json):
    PYTHONPATH=<nero>/src ~/Prismos-x/venv/bin/python vive_floor_cal.py

For each tag it waits for you to rest the controller on that tag, then samples a few fresh poses. With
the 4 (vive x,y) <-> (floor x,y) correspondences it solves the rigid SE(2) via
mrhack.localization.frame_align.umeyama_se2 (the same fit our SLAM<->Vive alignment uses) and writes
/tmp/vive_floor.json {"R":2x2,"t":2}, which follow_circle.py loads.

The 4 tag floor positions (metres) come from the same tag rectification procam_true uses, so the Vive
frame and the projected circle share one floor frame.
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

from mrhack.localization.frame_align import umeyama_se2

VIVE_POSE_FILE = "/run/nero/vive_pose.json"
TAG_SIZE_M = 0.15


def read_vive(path=VIVE_POSE_FILE, samples=15, timeout=6.0):
    """Average a few FRESH /run/nero/vive_pose.json samples -> a steady (x, y)."""
    xs, ys, last_seq, deadline = [], [], None, time.time() + timeout
    while len(xs) < samples and time.time() < deadline:
        try:
            d = json.load(open(path))
            if d.get("tracking_valid") and d.get("sequence") != last_seq:
                last_seq = d.get("sequence")
                xs.append(float(d["position"][0]))
                ys.append(float(d["position"][1]))
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.02)
    if len(xs) < 3:
        raise SystemExit(f"not enough fresh Vive samples from {path} (is nero-vive-udp-receive running?)")
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
