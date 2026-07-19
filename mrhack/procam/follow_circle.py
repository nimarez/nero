#!/usr/bin/env python3
"""Closed loop: the projected safety circle FOLLOWS the K1, driven by the Vive tracker.

Transform chain (fixed once at calibration, then each tick is cheap):
    Vive pose (libsurvive, on the Pi)  --SE(2) vive->floor-->  floor metres
    floor metres  --H_floor2proj (ArUco tags + dragged handles)-->  projector pixels
Each tick: read the Vive pose -> floor -> draw the ANSI safety circle (+ heading) at the K1's floor
position -> warp to the projector -> project. The circle tracks the robot.

Vive input is the team's UDP transport (PR #6): `nero-vive-udp-send` on the Pi ->
`nero-vive-udp-receive` on jscore writes /run/nero/vive_pose.json (atomic latest pose, 150 ms fresh).
We just poll that file.

Run on the box (jscore), once the rig is up:
    PYTHONPATH=<nero>/src ~/Prismos-x/venv/bin/python follow_circle.py            # real
    ... --mock                                                                   # synthetic motion, real projector
Verify the transform math + the pose parser with NO hardware:
    uv run --with numpy --with opencv-python-headless python follow_circle.py --selftest

Box inputs it expects:
  - /tmp/procam_calib.json  : the 4 dragged projector handles (mapper GUI)
  - /tmp/vive_floor.json    : SE(2) vive->floor from vive_floor_cal.py -> {"R":2x2,"t":2}
  - 4 ArUco tags (DICT_4X4_50, ids 0-3) visible to the RealSense (IR)
  - /run/nero/vive_pose.json : the Vive pose, written by nero-vive-udp-receive (PR #6)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time

import cv2
import numpy as np

PROJ_W, PROJ_H = 1920, 1080
TAG_SIZE_M = 0.15               # physical ArUco side in metres (MEASURE it) -> sets the metric scale
VIVE_POSE_FILE = "/run/nero/vive_pose.json"
VIVE_MAX_AGE = 0.15             # PR #6 freshness deadline (s)
# Tracker->robot-ground offset in the TRACKER body frame (metres). The tracker rides ~30 in (0.762 m)
# up on the K1's back strap, so its x,y is NOT the foot: it swings out when the robot leans and sits
# behind centre on the strap. We project it to the ground THROUGH the tracker orientation, so the
# point stays under the robot as it tilts. Default assumes the tracker's body -z points down; set the
# horizontal back-strap component here (or calibrate with the robot upright on a known tag).
MOUNT_OFFSET_BODY = (0.0, 0.0, -0.762)


def transform_points(pts, H):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def vive_to_floor(R, t, x, y):
    p = np.asarray(R, float) @ np.array([x, y], float) + np.asarray(t, float)
    return float(p[0]), float(p[1])


def yaw_from_quat_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_R(q):
    """3x3 rotation matrix from a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def tracker_to_ground_xy(position, quat_xyzw, mount_offset=MOUNT_OFFSET_BODY):
    """Project the tracker's 3D pose to the robot's ground point (x, y): foot = tracker + R @ offset.
    Because the offset rides in the tracker frame, the projection stays under the robot as it leans."""
    foot = np.asarray(position, float) + quat_to_R(quat_xyzw) @ np.asarray(mount_offset, float)
    return float(foot[0]), float(foot[1])


def parse_vive_pose(d, max_age=VIVE_MAX_AGE, now=None, mount_offset=None):
    """Parse a /run/nero/vive_pose.json dict (PR #6) -> (x, y, yaw) if valid+fresh, else None.
    Projects the tracker DOWN to the robot's ground point (mount height) + yaw from quaternion_xyzw."""
    if not d.get("tracking_valid", False):
        return None
    if now is not None:
        ra = d.get("transport", {}).get("received_at")
        if ra is not None and (now - float(ra)) > max_age:
            return None
    mo = MOUNT_OFFSET_BODY if mount_offset is None else mount_offset
    gx, gy = tracker_to_ground_xy(d["position"], d["quaternion_xyzw"], mo)
    return gx, gy, yaw_from_quat_xyzw(d["quaternion_xyzw"])


def render_circle(H_floor2proj, cx, cy, radius_m, heading=None, thick=30, color=(255, 255, 255)):
    """Metric circle (radius_m, centre (cx,cy) in floor metres) drawn as it should appear on the floor."""
    out = np.zeros((PROJ_H, PROJ_W, 3), np.uint8)
    ang = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    ring = np.column_stack((cx + radius_m * np.cos(ang), cy + radius_m * np.sin(ang)))
    proj = transform_points(ring, H_floor2proj)
    cv2.polylines(out, [np.rint(proj).astype(np.int32)], True, color, thick, cv2.LINE_AA)
    if heading is not None:
        ctr = transform_points([[cx, cy]], H_floor2proj)[0]
        tip = transform_points([[cx + radius_m * math.cos(heading), cy + radius_m * math.sin(heading)]], H_floor2proj)[0]
        cv2.arrowedLine(out, tuple(np.int32(ctr)), tuple(np.int32(tip)), color, thick, cv2.LINE_AA, tipLength=0.25)
    return out


# --------------------------------------------------------------------- calibration (hardware)
def calibrate(tag_size_m=TAG_SIZE_M):
    """Detect the 4 tags -> metric floor rectangle (metres) -> pair with dragged handles -> H_floor2proj."""
    from context_snippets import capture_color, detect_tags, load_handles          # lazy: box only
    from procam_true import build_cam_to_floor, order_clockwise, EXPECTED_IDS       # reuse Sol's rectification
    raw = detect_tags(capture_color())
    tags = {i: np.asarray(raw[i], float).reshape(4, 2) for i in EXPECTED_IDS if i in raw}
    if len(tags) < 4:
        raise SystemExit(f"calibration needs tags 0-3; got {sorted(tags)}")
    H_cam2floor, ref, score = build_cam_to_floor(tags)
    centers = np.array([tags[i].mean(0) for i in EXPECTED_IDS])
    metric = order_clockwise(transform_points(centers, H_cam2floor) * tag_size_m)   # tag-units -> metres
    handles = order_clockwise(np.asarray(load_handles(), float).reshape(4, 2))
    H_floor2proj, _ = cv2.findHomography(metric.astype(np.float32), handles.astype(np.float32))
    print(f"calibrated (ref tag {ref}, score {score:.4f}); floor rect "
          f"{np.ptp(metric[:, 0]):.2f} x {np.ptp(metric[:, 1]):.2f} m")
    return H_floor2proj


def load_vive_floor():
    try:
        d = json.load(open("/tmp/vive_floor.json"))
        return np.array(d["R"], float), np.array(d["t"], float)
    except Exception:
        print("no /tmp/vive_floor.json -> identity vive->floor (run vive_floor_cal.py)", file=sys.stderr)
        return np.eye(2), np.zeros(2)


# --------------------------------------------------------------------- pose sources
def vive_file_source(path=VIVE_POSE_FILE, max_age=VIVE_MAX_AGE, hz=30.0):
    """Poll nero-vive-udp-receive's latest-state file (PR #6). Yields (x, y, yaw) only when fresh + valid."""
    while True:
        try:
            with open(path) as f:
                p = parse_vive_pose(json.load(f), max_age, time.time())
            if p is not None:
                yield p
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(1.0 / hz)


def mock_source():
    t0 = time.time()
    while True:
        t = time.time() - t0
        yield 0.6 * math.cos(0.4 * t), 0.6 * math.sin(0.4 * t), 0.4 * t
        time.sleep(1 / 30.0)


# --------------------------------------------------------------------- the loop
def run(mock=False, tag_size_m=TAG_SIZE_M):
    from context_snippets import project_png                                        # lazy: box only
    from mrhack.safety.safety_circle import safety_radius                           # ANSI radius (metres)
    H = calibrate(tag_size_m)
    R, t = load_vive_floor()
    src = mock_source() if mock else vive_file_source()
    last, last_t = None, time.time()
    for (vx, vy, vyaw) in src:
        fx, fy = vive_to_floor(R, t, vx, vy)
        now = time.time()
        speed = math.hypot(fx - last[0], fy - last[1]) / (now - last_t) if last and now > last_t else 0.0
        last, last_t = (fx, fy), now
        project_png(render_circle(H, fx, fy, safety_radius(speed), heading=vyaw))


# --------------------------------------------------------------------- selftest (no hardware)
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _selftest():
    ok = True

    def chk(n, c):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    th = math.pi / 2
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    fx, fy = vive_to_floor(R, np.array([1.0, 2.0]), 1.0, 0.0)
    chk("vive_to_floor SE(2): (1,0)->(1,3)", abs(fx - 1.0) < 1e-9 and abs(fy - 3.0) < 1e-9)

    H = np.array([[200.0, 0, 300.0], [0, 200.0, 400.0], [0, 0, 1.0]])   # 200 px/m, offset
    img = render_circle(H, 1.0, 1.0, 0.4)
    ctr = transform_points([[1.0, 1.0]], H)[0]
    chk("circle centre maps to (500,600) px", abs(ctr[0] - 500) < 1e-6 and abs(ctr[1] - 600) < 1e-6)
    ys, xs = np.where(img[:, :, 0] > 0)
    span = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    chk(f"circle span ~160px (+thick) got {span}", 140 <= span <= 215)
    fpose = render_circle(H, 1.0, 1.0, 0.4, heading=0.0)
    chk("heading arrow adds pixels", int((fpose[:, :, 0] > 0).sum()) > int((img[:, :, 0] > 0).sum()))

    # PR #6 vive_pose.json parsing
    now = 1_000_000.0
    d = {"tracking_valid": True, "position": [1.5, 2.5, 0.1],
         "quaternion_xyzw": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
         "transport": {"received_at": now}}
    p = parse_vive_pose(d, 0.15, now)
    chk("parse vive_pose.json -> (1.5,2.5,yaw=90deg)",
        p is not None and abs(p[0] - 1.5) < 1e-9 and abs(p[1] - 2.5) < 1e-9 and abs(_wrap(p[2] - math.pi / 2)) < 1e-6)
    chk("stale pose (age>150ms) rejected", parse_vive_pose(d, 0.15, now + 1.0) is None)
    chk("tracking_valid=False rejected", parse_vive_pose({**d, "tracking_valid": False}, 0.15, now) is None)

    # mount-height projection (tracker ~0.762 m up on the robot's back)
    up = {"tracking_valid": True, "position": [1.0, 2.0, 0.762], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "transport": {"received_at": now}}
    pu = parse_vive_pose(up, 0.15, now)
    chk("mount: upright tracker -> ground (1,2)", pu is not None and abs(pu[0] - 1.0) < 1e-9 and abs(pu[1] - 2.0) < 1e-9)
    pit = {**up, "quaternion_xyzw": [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]}   # 90deg pitch
    pp = parse_vive_pose(pit, 0.15, now)
    chk("mount: 90deg lean shifts ground by ~mount height", pp is not None and abs(pp[0] - (1.0 - 0.762)) < 1e-6 and abs(pp[1] - 2.0) < 1e-6)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tag-size-m", type=float, default=TAG_SIZE_M)
    ap.add_argument("--mount-height", type=float, default=0.762, help="tracker height above the ground (m); 30 in = 0.762")
    a = ap.parse_args()
    global MOUNT_OFFSET_BODY
    MOUNT_OFFSET_BODY = (MOUNT_OFFSET_BODY[0], MOUNT_OFFSET_BODY[1], -a.mount_height)
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    run(mock=a.mock, tag_size_m=a.tag_size_m)


if __name__ == "__main__":
    main()
