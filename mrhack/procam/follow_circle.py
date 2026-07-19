#!/usr/bin/env python3
"""Closed loop: the projected floor SCENE follows the K1 — safety barrier + safety circle + where it's
about to go (bearing ray in the far/YOLO phase, planned path in the close/SLAM phase).

Transform chain (fixed once at calibration, then each tick is cheap):
    Vive tracker (libsurvive, on the Pi, ~0.76 m up) --project to ground--> --SE(2) vive->floor-->
    floor metres  --H_floor2proj (ArUco tags + dragged handles)-->  projector pixels
Heading (yaw) is rotated from the Vive frame into the floor frame by the SE(2) angle.

Inputs (box):
  - /run/nero/vive_pose.json : Vive pose (PR #6). We project the tracker DOWN to the robot ground point
    (mount height + lean) and FAIL CLOSED on stale/invalid poses (blank the projector, not a stale circle).
  - /run/nero/nav.json       : nav telemetry {phase, goal:[x,y], bearing, path:[[x,y]...]} in floor metres.

    PYTHONPATH=<nero>/src ~/Prismos-x/venv/bin/python follow_circle.py            # real
    ... --mock                                                                   # synthetic, real projector
    uv run --with numpy --with opencv-python-headless python follow_circle.py --selftest   # no hardware
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
TAG_SIZE_M = 0.15
VIVE_POSE_FILE = "/run/nero/vive_pose.json"
NAV_STATE_FILE = "/run/nero/nav.json"
VIVE_MAX_AGE = 0.15             # PR #6 freshness deadline (s)
DEADMAN_S = 0.30               # no fresh pose for this long -> blank the projector (safety)
BARRIER_RADIUS_M = 1.20
# Tracker->robot-ground offset in the TRACKER body frame (m). Tracker ~30 in (0.762 m) up on the back
# strap; projected DOWN through the tracker orientation so it stays under the robot on lean. Default
# assumes body -z points down; calibrate the mount rotation/offset for exactness (Sol review).
MOUNT_OFFSET_BODY = (0.0, 0.0, -0.762)

C_BARRIER = (0, 140, 255)   # orange keep-out barrier
C_SAFETY = (255, 255, 255)  # white ANSI safety circle
C_BEARING = (0, 255, 255)   # yellow bearing ray (far / YOLO)
C_PATH = (0, 255, 0)        # green planned path (close / SLAM)
C_GOAL = (0, 0, 255)        # red goal marker
C_LOST = (0, 0, 255)        # red tracking-lost banner


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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
    """/run/nero/vive_pose.json dict -> (x, y, yaw) at the robot ground point, or None. FAIL CLOSED:
    reject invalid tracking, missing/old/future timestamps, wrong shapes, non-finite, or a zero quaternion."""
    try:
        if not d.get("tracking_valid", False):
            return None
        if now is not None:                                   # freshness: fail closed
            ra = d.get("transport", {}).get("received_at")
            if ra is None or not math.isfinite(float(ra)):
                return None
            age = now - float(ra)
            if age > max_age or age < -0.05:                  # stale, or a future stamp beyond clock skew
                return None
        pos = np.asarray(d["position"], float)
        q = np.asarray(d["quaternion_xyzw"], float)
        if pos.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(q)):
            return None
        n = float(np.linalg.norm(q))
        if n < 1e-6:
            return None
        q = q / n                                             # normalise: rotation formulas need a unit quat
        mo = MOUNT_OFFSET_BODY if mount_offset is None else mount_offset
        gx, gy = tracker_to_ground_xy(pos, q, mo)
        return gx, gy, yaw_from_quat_xyzw(q)
    except (KeyError, ValueError, TypeError, IndexError):
        return None


# --------------------------------------------------------------------- floor drawing
def _finite_int(pts):
    """Nx2 float -> Nx2 int32, keeping only finite points inside a safe padded coordinate range."""
    pts = np.asarray(pts, float)
    ok = np.all(np.isfinite(pts), axis=1) & np.all(np.abs(pts) < 1e5, axis=1)
    return np.rint(pts[ok]).astype(np.int32), int(ok.sum())


def _ring(out, H, cx, cy, radius_m, color, thick):
    ang = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    pts, n = _finite_int(transform_points(np.column_stack((cx + radius_m * np.cos(ang), cy + radius_m * np.sin(ang))), H))
    if n >= 3:
        cv2.polylines(out, [pts], True, color, thick, cv2.LINE_AA)


def _arrow(out, H, x0, y0, x1, y1, color, thick):
    p, n = _finite_int(transform_points([[x0, y0], [x1, y1]], H))
    if n == 2:
        cv2.arrowedLine(out, tuple(p[0]), tuple(p[1]), color, thick, cv2.LINE_AA, tipLength=0.22)


def lost_frame(text="TRACKING LOST"):
    out = np.zeros((PROJ_H, PROJ_W, 3), np.uint8)
    cv2.putText(out, text, (PROJ_W // 2 - 430, PROJ_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, C_LOST, 8, cv2.LINE_AA)
    return out


def render_scene(H, pose, safety_r, barrier_r=BARRIER_RADIUS_M, nav=None, thick=24):
    """Floor overlay at the robot: keep-out barrier + ANSI safety circle + heading, plus where it's going
    (bearing ray in far/YOLO, planned path in close/SLAM) + goal marker."""
    cx, cy, yaw = pose
    out = np.zeros((PROJ_H, PROJ_W, 3), np.uint8)
    _ring(out, H, cx, cy, barrier_r, C_BARRIER, thick)
    _ring(out, H, cx, cy, safety_r, C_SAFETY, thick)
    _arrow(out, H, cx, cy, cx + safety_r * math.cos(yaw), cy + safety_r * math.sin(yaw), C_SAFETY, thick)
    if nav:
        phase = nav.get("phase")
        goal = nav.get("goal")
        if phase == "far":
            b, reach = None, 2.5
            if goal is not None:                                          # goal is authoritative for the bearing
                dx, dy = goal[0] - cx, goal[1] - cy
                if math.hypot(dx, dy) > 1e-3:
                    b, reach = math.atan2(dy, dx), math.hypot(dx, dy)
            elif nav.get("bearing") is not None and math.isfinite(nav["bearing"]):
                b = float(nav["bearing"])
            if b is not None:
                _arrow(out, H, cx, cy, cx + reach * math.cos(b), cy + reach * math.sin(b), C_BEARING, thick)
        elif phase == "close" and nav.get("path") and len(nav["path"]) >= 2:
            pts, n = _finite_int(transform_points(np.asarray(nav["path"], float), H))
            if n >= 2:
                cv2.polylines(out, [pts], False, C_PATH, thick, cv2.LINE_AA)
        if goal is not None and all(math.isfinite(v) for v in goal):
            g, n = _finite_int(transform_points([goal], H))
            if n == 1:
                cv2.drawMarker(out, tuple(g[0]), C_GOAL, cv2.MARKER_TILTED_CROSS, 70, thick)
    return out


# --------------------------------------------------------------------- calibration / state (hardware)
def calibrate(tag_size_m=TAG_SIZE_M):
    from context_snippets import capture_color, detect_tags, load_handles
    from procam_true import build_cam_to_floor, order_clockwise, EXPECTED_IDS
    raw = detect_tags(capture_color())
    tags = {i: np.asarray(raw[i], float).reshape(4, 2) for i in EXPECTED_IDS if i in raw}
    if len(tags) < 4:
        raise SystemExit(f"calibration needs tags 0-3; got {sorted(tags)}")
    H_cam2floor, ref, score = build_cam_to_floor(tags)
    centers = np.array([tags[i].mean(0) for i in EXPECTED_IDS])
    metric = order_clockwise(transform_points(centers, H_cam2floor) * tag_size_m)
    handles = order_clockwise(np.asarray(load_handles(), float).reshape(4, 2))
    H, _ = cv2.findHomography(metric.astype(np.float32), handles.astype(np.float32))
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise SystemExit("calibration produced an invalid homography (check tag geometry + dragged handles)")
    print(f"calibrated (ref tag {ref}, score {score:.4f}); floor rect "
          f"{np.ptp(metric[:, 0]):.2f} x {np.ptp(metric[:, 1]):.2f} m")
    return H


def load_vive_floor():
    """SE(2) vive->floor {R,t}. Validates it is a proper rotation (no reflection/scale/shear)."""
    try:
        d = json.load(open("/tmp/vive_floor.json"))
        R, t = np.array(d["R"], float), np.array(d["t"], float)
        if R.shape == (2, 2) and t.shape == (2,) and np.all(np.isfinite(R)) and np.all(np.isfinite(t)) \
           and np.allclose(R.T @ R, np.eye(2), atol=1e-3) and np.linalg.det(R) > 0:
            return R, t
        print("bad /tmp/vive_floor.json (not a proper SE(2)) -> identity", file=sys.stderr)
    except Exception:
        print("no /tmp/vive_floor.json -> identity vive->floor (run vive_floor_cal.py)", file=sys.stderr)
    return np.eye(2), np.zeros(2)


def load_nav_state(path=NAV_STATE_FILE):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------- pose sources
def vive_file_source(path=VIVE_POSE_FILE, max_age=VIVE_MAX_AGE, hz=30.0):
    """Poll nero-vive-udp-receive's latest-state file (PR #6). Yields (x,y,yaw) when fresh+valid, else None."""
    while True:
        p = None
        try:
            with open(path) as f:
                p = parse_vive_pose(json.load(f), max_age, time.time())
        except Exception:
            p = None
        yield p
        time.sleep(1.0 / hz)


def mock_source():
    t0 = time.time()
    while True:
        t = time.time() - t0
        yield 0.6 * math.cos(0.4 * t), 0.6 * math.sin(0.4 * t), 0.4 * t
        time.sleep(1 / 30.0)


# --------------------------------------------------------------------- the loop
def run(mock=False, tag_size_m=TAG_SIZE_M, barrier_r=BARRIER_RADIUS_M):
    from context_snippets import project_png
    from mrhack.safety.safety_circle import safety_radius
    H = calibrate(tag_size_m)
    R, t = load_vive_floor()
    se2_yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))          # rotate heading into the floor frame
    src = mock_source() if mock else vive_file_source()
    last, last_t, last_fresh = None, time.time(), time.time()
    for p in src:
        now = time.time()
        if p is None:                                            # deadman: no fresh pose -> blank, not a stale circle
            if now - last_fresh > DEADMAN_S:
                project_png(lost_frame())
                last = None
            continue
        last_fresh = now
        vx, vy, vyaw = p
        fx, fy = vive_to_floor(R, t, vx, vy)
        fyaw = _wrap(vyaw + se2_yaw)
        speed = math.hypot(fx - last[0], fy - last[1]) / (now - last_t) if last and now > last_t else 0.0
        last, last_t = (fx, fy), now
        nav = {"phase": "far", "goal": [fx + 2.0, fy]} if mock else load_nav_state()
        project_png(render_scene(H, (fx, fy, fyaw), safety_radius(speed), barrier_r, nav))


# --------------------------------------------------------------------- selftest (no hardware)
def _selftest():
    ok = True

    def chk(n, c):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    th = math.pi / 2
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    fx, fy = vive_to_floor(Rm, np.array([1.0, 2.0]), 1.0, 0.0)
    chk("vive_to_floor SE(2): (1,0)->(1,3)", abs(fx - 1.0) < 1e-9 and abs(fy - 3.0) < 1e-9)

    H = np.array([[200.0, 0, 300.0], [0, 200.0, 400.0], [0, 0, 1.0]])
    base = render_scene(H, (1.0, 1.0, 0.0), 0.4, barrier_r=0.9)
    ctr = transform_points([[1.0, 1.0]], H)[0]
    chk("scene centre maps to (500,600) px", abs(ctr[0] - 500) < 1e-6 and abs(ctr[1] - 600) < 1e-6)

    def _cnt(im, bgr):
        return int(np.all(np.abs(im.astype(int) - np.array(bgr)) < 60, axis=2).sum())
    chk("barrier ring drawn (orange)", _cnt(base, C_BARRIER) > 0)
    far = render_scene(H, (1.0, 1.0, 0.0), 0.4, 0.9, nav={"phase": "far", "goal": [3.0, 1.0]})
    chk("far: yellow bearing ray drawn", _cnt(far, C_BEARING) > _cnt(base, C_BEARING))
    chk("far: red goal marker drawn", _cnt(far, C_GOAL) > _cnt(base, C_GOAL))
    close = render_scene(H, (1.0, 1.0, 0.0), 0.4, 0.9, nav={"phase": "close", "path": [[1.0, 1.0], [1.5, 1.2], [2.0, 1.0]]})
    chk("close: green path drawn", _cnt(close, C_PATH) > _cnt(base, C_PATH))
    chk("one-point path draws nothing (no crash)", _cnt(render_scene(H, (1.0, 1.0, 0.0), 0.4, 0.9, nav={"phase": "close", "path": [[1.0, 1.0]]}), C_PATH) == 0)
    chk("lost_frame is red", _cnt(lost_frame(), C_LOST) > 0)

    now = 1_000_000.0
    d = {"tracking_valid": True, "position": [1.5, 2.5, 0.1],
         "quaternion_xyzw": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], "transport": {"received_at": now}}
    p = parse_vive_pose(d, 0.15, now)
    chk("parse vive_pose.json -> (1.5,2.5,yaw=90deg)",
        p is not None and abs(p[0] - 1.5) < 1e-9 and abs(p[1] - 2.5) < 1e-9 and abs(_wrap(p[2] - math.pi / 2)) < 1e-6)
    chk("stale rejected", parse_vive_pose(d, 0.15, now + 1.0) is None)
    chk("invalid tracking rejected", parse_vive_pose({**d, "tracking_valid": False}, 0.15, now) is None)
    chk("missing timestamp fails CLOSED", parse_vive_pose({**d, "transport": {}}, 0.15, now) is None)
    chk("NaN position rejected", parse_vive_pose({**d, "position": [float("nan"), 0, 0]}, 0.15, now) is None)
    nu = {**d, "position": [1.0, 2.0, 0.762], "quaternion_xyzw": [0.0, 0.0, 0.0, 3.0]}   # non-unit quat
    pn = parse_vive_pose(nu, 0.15, now)
    chk("non-unit quaternion normalised -> ground (1,2)", pn is not None and abs(pn[0] - 1.0) < 1e-9 and abs(pn[1] - 2.0) < 1e-9)
    up = {**d, "position": [1.0, 2.0, 0.762], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
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
