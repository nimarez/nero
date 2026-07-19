#!/usr/bin/env python3
"""Closed loop: the projected floor SCENE follows the K1 — safety barrier + safety circle + where it's
about to go (the bearing ray in the far/YOLO phase, the planned path in the close/SLAM phase).

Transform chain (fixed once at calibration, then each tick is cheap):
    Vive tracker (libsurvive, on the Pi, ~0.76 m up) --project to ground--> --SE(2) vive->floor-->
    floor metres  --H_floor2proj (ArUco tags + dragged handles)-->  projector pixels

Inputs:
  - Vive pose: the team's UDP transport (PR #6) -> /run/nero/vive_pose.json (position+quaternion, 150ms fresh).
    We project the tracker down to the robot's ground point (mount height + lean).
  - Nav telemetry (goal / bearing / planned path / phase): /run/nero/nav.json, bridged from the onboard
    controller's grid-space object position + planner (nav_bridge, not yet wired -> mock in the meantime).

Run on the box (jscore), rig up:
    PYTHONPATH=<nero>/src ~/Prismos-x/venv/bin/python follow_circle.py            # real
    ... --mock                                                                   # synthetic pose+nav, real projector
Verify all the math with NO hardware:
    uv run --with numpy --with opencv-python-headless python follow_circle.py --selftest
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
NAV_STATE_FILE = "/run/nero/nav.json"
VIVE_MAX_AGE = 0.15             # PR #6 freshness deadline (s)
BARRIER_RADIUS_M = 1.20         # projected keep-out BARRIER radius (m) — the "stay back" zone around the robot
# Tracker->robot-ground offset in the TRACKER body frame (m). Tracker rides ~30 in (0.762 m) up on the
# back strap; project it to the ground THROUGH the tracker orientation so it stays under the robot when
# it leans. Default assumes body -z points down; set the horizontal back-strap term or calibrate.
MOUNT_OFFSET_BODY = (0.0, 0.0, -0.762)

# colours (BGR)
C_BARRIER = (0, 140, 255)   # orange keep-out barrier
C_SAFETY = (255, 255, 255)  # white ANSI safety circle
C_BEARING = (0, 255, 255)   # yellow bearing ray (far / YOLO phase)
C_PATH = (0, 255, 0)        # green planned path (close / SLAM phase)
C_GOAL = (0, 0, 255)        # red goal marker


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
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def tracker_to_ground_xy(position, quat_xyzw, mount_offset=MOUNT_OFFSET_BODY):
    """Tracker 3D pose -> robot ground (x, y): foot = tracker + R @ offset (stays under the robot on lean)."""
    foot = np.asarray(position, float) + quat_to_R(quat_xyzw) @ np.asarray(mount_offset, float)
    return float(foot[0]), float(foot[1])


def parse_vive_pose(d, max_age=VIVE_MAX_AGE, now=None, mount_offset=None):
    """/run/nero/vive_pose.json dict -> (x, y, yaw) at the robot's ground point if valid+fresh, else None."""
    if not d.get("tracking_valid", False):
        return None
    if now is not None:
        ra = d.get("transport", {}).get("received_at")
        if ra is not None and (now - float(ra)) > max_age:
            return None
    mo = MOUNT_OFFSET_BODY if mount_offset is None else mount_offset
    gx, gy = tracker_to_ground_xy(d["position"], d["quaternion_xyzw"], mo)
    return gx, gy, yaw_from_quat_xyzw(d["quaternion_xyzw"])


# --------------------------------------------------------------------- floor drawing primitives
def _ring(out, H, cx, cy, radius_m, color, thick):
    ang = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    pts = transform_points(np.column_stack((cx + radius_m * np.cos(ang), cy + radius_m * np.sin(ang))), H)
    cv2.polylines(out, [np.rint(pts).astype(np.int32)], True, color, thick, cv2.LINE_AA)


def _arrow(out, H, x0, y0, x1, y1, color, thick):
    p = transform_points([[x0, y0], [x1, y1]], H)
    cv2.arrowedLine(out, tuple(np.int32(p[0])), tuple(np.int32(p[1])), color, thick, cv2.LINE_AA, tipLength=0.22)


def render_scene(H, pose, safety_r, barrier_r=BARRIER_RADIUS_M, nav=None, thick=24):
    """Compose the floor overlay at the robot: keep-out barrier + ANSI safety circle + heading, plus
    where it's about to go — the bearing ray (far/YOLO) or the planned path (close/SLAM) + goal marker."""
    cx, cy, yaw = pose
    out = np.zeros((PROJ_H, PROJ_W, 3), np.uint8)
    _ring(out, H, cx, cy, barrier_r, C_BARRIER, thick)                                   # outer keep-out barrier
    _ring(out, H, cx, cy, safety_r, C_SAFETY, thick)                                     # inner ANSI circle
    _arrow(out, H, cx, cy, cx + safety_r * math.cos(yaw), cy + safety_r * math.sin(yaw), C_SAFETY, thick)
    if nav:
        phase = nav.get("phase", "far")
        goal = nav.get("goal")
        if phase == "far":                                                               # heading toward the target
            b = nav.get("bearing")
            if b is None and goal is not None:
                b = math.atan2(goal[1] - cy, goal[0] - cx)
            if b is not None:
                reach = math.hypot(goal[0] - cx, goal[1] - cy) if goal is not None else 2.5
                _arrow(out, H, cx, cy, cx + reach * math.cos(b), cy + reach * math.sin(b), C_BEARING, thick)
        elif phase == "close" and nav.get("path"):                                       # the planned path
            pts = transform_points(np.asarray(nav["path"], float), H)
            cv2.polylines(out, [np.rint(pts).astype(np.int32)], False, C_PATH, thick, cv2.LINE_AA)
        if goal is not None:
            g = transform_points([goal], H)[0]
            cv2.drawMarker(out, tuple(np.int32(g)), C_GOAL, cv2.MARKER_TILTED_CROSS, 70, thick)
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


def load_nav_state(path=NAV_STATE_FILE):
    """Onboard nav telemetry bridged to a file: {phase, goal:[x,y], bearing, path:[[x,y]...]} in floor
    metres. Returns None if absent (scene falls back to just the safety rings)."""
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


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
def run(mock=False, tag_size_m=TAG_SIZE_M, barrier_r=BARRIER_RADIUS_M):
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
        nav = {"phase": "far", "goal": [fx + 2.0, fy], "bearing": vyaw} if mock else load_nav_state()
        project_png(render_scene(H, (fx, fy, vyaw), safety_radius(speed), barrier_r, nav))


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
    base = render_scene(H, (1.0, 1.0, 0.0), 0.4, barrier_r=0.9)
    ctr = transform_points([[1.0, 1.0]], H)[0]
    chk("scene centre maps to (500,600) px", abs(ctr[0] - 500) < 1e-6 and abs(ctr[1] - 600) < 1e-6)

    def _cnt(im, bgr):   # count pixels near a BGR colour (the line cores past anti-aliasing)
        return int(np.all(np.abs(im.astype(int) - np.array(bgr)) < 60, axis=2).sum())
    chk("barrier ring drawn (orange)", _cnt(base, C_BARRIER) > 0)

    far = render_scene(H, (1.0, 1.0, 0.0), 0.4, 0.9, nav={"phase": "far", "goal": [3.0, 1.0]})
    chk("far: yellow bearing ray drawn", _cnt(far, C_BEARING) > _cnt(base, C_BEARING))
    chk("far: red goal marker drawn", _cnt(far, C_GOAL) > _cnt(base, C_GOAL))
    close = render_scene(H, (1.0, 1.0, 0.0), 0.4, 0.9, nav={"phase": "close", "path": [[1.0, 1.0], [1.5, 1.2], [2.0, 1.0]]})
    chk("close: green path drawn", _cnt(close, C_PATH) > _cnt(base, C_PATH))

    # PR #6 vive_pose.json parsing + mount-height projection
    now = 1_000_000.0
    d = {"tracking_valid": True, "position": [1.5, 2.5, 0.1],
         "quaternion_xyzw": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], "transport": {"received_at": now}}
    p = parse_vive_pose(d, 0.15, now)
    chk("parse vive_pose.json -> (1.5,2.5,yaw=90deg)",
        p is not None and abs(p[0] - 1.5) < 1e-9 and abs(p[1] - 2.5) < 1e-9 and abs(_wrap(p[2] - math.pi / 2)) < 1e-6)
    chk("stale rejected", parse_vive_pose(d, 0.15, now + 1.0) is None)
    chk("invalid rejected", parse_vive_pose({**d, "tracking_valid": False}, 0.15, now) is None)
    up = {"tracking_valid": True, "position": [1.0, 2.0, 0.762], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "transport": {"received_at": now}}
    chk("mount: upright -> ground (1,2)", parse_vive_pose(up, 0.15, now)[:2] == (1.0, 2.0))
    pit = {**up, "quaternion_xyzw": [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]}
    pp = parse_vive_pose(pit, 0.15, now)
    chk("mount: 90deg lean shifts ground by ~mount height", abs(pp[0] - (1.0 - 0.762)) < 1e-6 and abs(pp[1] - 2.0) < 1e-6)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tag-size-m", type=float, default=TAG_SIZE_M)
    ap.add_argument("--barrier-radius", type=float, default=BARRIER_RADIUS_M, help="keep-out barrier radius (m)")
    ap.add_argument("--mount-height", type=float, default=0.762, help="tracker height above the ground (m); 30 in = 0.762")
    a = ap.parse_args()
    global MOUNT_OFFSET_BODY
    MOUNT_OFFSET_BODY = (MOUNT_OFFSET_BODY[0], MOUNT_OFFSET_BODY[1], -a.mount_height)
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    run(mock=a.mock, tag_size_m=a.tag_size_m, barrier_r=a.barrier_radius)


if __name__ == "__main__":
    main()
