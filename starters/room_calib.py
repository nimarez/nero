#!/usr/bin/env python
"""STARTER - Vive room calibration + mapping (ArUco-free).

Use a Vive controller as a 3D probe to calibrate the whole room:
  1. FLOOR     - define the floor frame (origin + x-axis + plane) from touched points.
  2. PROJECTOR - 4-point projector->floor homography (project a crosshair, touch it).
  3. OBSTACLES - mark table/obstacle footprints (touch corners) -> 2D occupancy grid (for the planner).
  4. SWEEP     - hold the trigger and drag over surfaces -> a sparse point cloud (PLY).
Outputs room_calib.json (floor frame + H_proj2floor + bounds + obstacles) + *_occupancy.npy + room_cloud.ply.

Verify the math (no hardware; honours uv):
    uv run --with opencv-python --with numpy python starters/room_calib.py --selftest
Live (needs a Vive + SteamVR running + the projector as an extended display):
    uv run --with openvr --with opencv-python --with numpy python starters/room_calib.py --run

STATUS: floor-frame geometry, homography, occupancy rasterization and PLY export are
self-test-verified. The OpenVR controller I/O + projector display are written, NOT
hardware-verified. A Vive-controller sweep gives a SPARSE cloud (where you dragged it);
a DENSE room cloud comes from the OAK-D/K1 depth cameras or a Marble scan.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time

import numpy as np


# ---------------------------------------------------------------- math core (verified)
def fit_floor_frame(origin, x_point, plane_points):
    """origin/x_point/plane_points are 3D world points. Returns (R, O): floor axes as the
    columns of R (x, y, up), origin O on the floor.  floor = to_floor(R, O, world_point)."""
    O = np.asarray(origin, float)
    pts = np.asarray(plane_points, float)
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c)
    n = vh[2]                                  # plane normal = smallest singular vector
    if np.dot(n, np.array([0.0, 1.0, 0.0])) < 0:   # Vive is Y-up: make the floor normal point up
        n = -n
    z_axis = n / np.linalg.norm(n)
    xd = np.asarray(x_point, float) - O
    xd = xd - np.dot(xd, z_axis) * z_axis      # project the x-direction onto the floor plane
    x_axis = xd / np.linalg.norm(xd)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, O


def to_floor(R, O, p):
    """World point -> floor coords (x, y along the floor, z = height above it)."""
    return np.asarray(R, float).T @ (np.asarray(p, float) - np.asarray(O, float))


def projector_homography(proj_px, floor_xy):
    import cv2
    H, _ = cv2.findHomography(np.asarray(proj_px, float), np.asarray(floor_xy, float), 0)
    return H


def rasterize_occupancy(bounds, res, polygons_floor):
    """polygons_floor: list of [(x,y),...] obstacle footprints in floor metres.
    Returns an occupancy grid (0 free, 100 occupied) — nero OccupancyGrid convention."""
    import cv2
    xmin, ymin, xmax, ymax = bounds
    W = max(1, int(round((xmax - xmin) / res)))
    H = max(1, int(round((ymax - ymin) / res)))
    grid = np.zeros((H, W), dtype=np.int16)

    def px(x, y):
        return int((x - xmin) / res), int((y - ymin) / res)

    for poly in polygons_floor:
        pts = np.array([px(x, y) for (x, y) in poly], dtype=np.int32)
        cv2.fillPoly(grid, [pts], 100)
    return grid


def write_ply(points, path):
    pts = np.asarray(points, float).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    return path


# ---------------------------------------------------------------- OpenVR I/O (untested)
def _openvr_setup():
    import openvr
    return openvr, openvr.init(openvr.VRApplication_Other)


def _find_controller(openvr, vr):
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            return i
    return None


def _device_pos(openvr, vr, idx):
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    m = poses[idx].mDeviceToAbsoluteTracking
    return np.array([m[0][3], m[1][3], m[2][3]])


def _trigger_down(openvr, vr, idx):
    got, state = vr.getControllerState(idx)
    return bool(got and state.ulButtonPressed & (1 << openvr.k_EButton_SteamVR_Trigger))


def capture_point(openvr, vr, idx, prompt):
    print(f"{prompt}  -> pull the TRIGGER to capture")
    while not _trigger_down(openvr, vr, idx):
        time.sleep(0.02)
    pos = _device_pos(openvr, vr, idx)
    while _trigger_down(openvr, vr, idx):
        time.sleep(0.02)
    print(f"   captured {pos.round(3)}")
    return pos


def show_crosshair(px, py, proj_res, monitor_x, win="projector"):
    import cv2
    img = np.zeros((proj_res[1], proj_res[0], 3), np.uint8)
    cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 240, 4)
    cv2.circle(img, (px, py), 14, (0, 0, 255), -1)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, monitor_x, 0)                 # push onto the projector (extended display)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    cv2.waitKey(400)


def run(out_json, out_ply, proj_res=(1920, 1080), monitor_x=1920, grid_res=0.05):
    openvr, vr = _openvr_setup()
    idx = _find_controller(openvr, vr)
    if idx is None:
        print("No Vive controller found — is SteamVR running and the controller on?")
        return

    print("\n=== 1. FLOOR FRAME ===")
    O0 = capture_point(openvr, vr, idx, "Touch the ORIGIN on the floor")
    X0 = capture_point(openvr, vr, idx, "Touch a point ~1 m along +X (on the floor)")
    plane = [O0, X0]
    for k in range(2):
        plane.append(capture_point(openvr, vr, idx, f"Touch spread-out floor point {k + 1}/2"))
    R, O = fit_floor_frame(O0, X0, plane)
    print("floor frame set.")

    print("\n=== 2. PROJECTOR -> FLOOR (touch 4 projected crosshairs) ===")
    corners = [(int(proj_res[0] * fx), int(proj_res[1] * fy))
               for fx, fy in [(0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)]]
    proj_px, floor_xy = [], []
    for (px, py) in corners:
        show_crosshair(px, py, proj_res, monitor_x)
        p = capture_point(openvr, vr, idx, f"Touch the projected crosshair at pixel ({px},{py})")
        fx, fy, _ = to_floor(R, O, p)
        proj_px.append((px, py))
        floor_xy.append((fx, fy))
    H_proj2floor = projector_homography(proj_px, floor_xy)
    print("projector->floor homography set.")

    print("\n=== 3. OBSTACLES (tables, boxes, ...) ===")
    polygons = []
    while input("Map an obstacle footprint? [y/N] ").strip().lower() == "y":
        n = int(input("  how many corners? ") or "4")
        poly = []
        for k in range(n):
            p = capture_point(openvr, vr, idx, f"  corner {k + 1}/{n}")
            fx, fy, _ = to_floor(R, O, p)
            poly.append((fx, fy))
        polygons.append(poly)

    print("\n=== 4. SWEEP a point cloud (optional) ===")
    cloud = []
    if input("Sweep a cloud? HOLD trigger + drag, release to stop. [y/N] ").strip().lower() == "y":
        while not _trigger_down(openvr, vr, idx):
            time.sleep(0.02)
        while _trigger_down(openvr, vr, idx):
            cloud.append(to_floor(R, O, _device_pos(openvr, vr, idx)))
            time.sleep(1 / 30)
        print(f"   {len(cloud)} points swept.")

    allxy = floor_xy + [pt for poly in polygons for pt in poly] + [(c[0], c[1]) for c in cloud]
    xs = [a[0] for a in allxy] or [0.0, 1.0]
    ys = [a[1] for a in allxy] or [0.0, 1.0]
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    grid = rasterize_occupancy(bounds, grid_res, polygons)

    json.dump({
        "floor_R": R.tolist(), "floor_O": O.tolist(),
        "H_proj2floor": H_proj2floor.tolist(),
        "floor_bounds": bounds, "obstacles": polygons,
        "grid_res": grid_res, "grid_shape": list(grid.shape),
        "calib_time": time.time(),
    }, open(out_json, "w"), indent=2)
    np.save(out_json.replace(".json", "_occupancy.npy"), grid)
    if cloud:
        write_ply(cloud, out_ply)
    openvr.shutdown()
    print(f"\nwrote {out_json} + occupancy grid{' + ' + out_ply if cloud else ''}")


# ---------------------------------------------------------------- robot tracker -> foot offset
def _find_tracker(openvr, vr):
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
            return i
    return None


def _device_forward_yaw(openvr, vr, idx, R):
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    m = poses[idx].mDeviceToAbsoluteTracking
    fwd_world = np.array([-m[0][2], -m[1][2], -m[2][2]])   # OpenVR device forward = -Z
    f = np.asarray(R, float).T @ fwd_world
    return math.atan2(f[1], f[0])


def apply_robot_offset(tx, ty, tyaw, off):
    """Tracker floor pose (x, y, yaw) -> robot FOOT-CENTER pose (x, y, yaw)."""
    ryaw = (tyaw - off["yaw_offset"] + math.pi) % (2 * math.pi) - math.pi
    c, s = math.cos(ryaw), math.sin(ryaw)
    fx = tx - (c * off["dx"] - s * off["dy"])
    fy = ty - (s * off["dx"] + c * off["dy"])
    return fx, fy, ryaw


def robot_offset(out_json, calib_json):
    """Calibrate the back-strap tracker -> foot-center offset. Face the robot along floor +X first."""
    openvr, vr = _openvr_setup()
    calib = json.load(open(calib_json))
    R = np.array(calib["floor_R"]); O = np.array(calib["floor_O"])
    tr = _find_tracker(openvr, vr); ct = _find_controller(openvr, vr)
    if tr is None or ct is None:
        print("Need BOTH a tracker (on the robot's back strap) and a controller (in hand).")
        return
    print("Stand the K1 upright, tracker on its back strap, FACING the floor +X axis.")
    input("Press ENTER to read the tracker...")
    tpos = to_floor(R, O, _device_pos(openvr, vr, tr))
    tyaw = _device_forward_yaw(openvr, vr, tr, R)
    foot = to_floor(R, O, capture_point(openvr, vr, ct, "Touch the floor at the MIDPOINT of the robot's feet"))
    off = {"dx": float(tpos[0] - foot[0]), "dy": float(tpos[1] - foot[1]), "dz": float(tpos[2]),
           "yaw_offset": float(tyaw),
           "note": "robot_foot_pose = apply_robot_offset(tracker_x, tracker_y, tracker_yaw, this)"}
    json.dump(off, open(out_json, "w"), indent=2)
    openvr.shutdown()
    print(f"wrote {out_json}: dx={off['dx']:.3f} dy={off['dy']:.3f} dz={off['dz']:.3f} yaw_off={off['yaw_offset']:.3f}")


# ---------------------------------------------------------------- self-test (no hardware)
def _selftest():
    ok = True
    # 1) floor frame round-trip on a synthetic tilted floor
    u = np.array([0.08, 1.0, 0.05]); u /= np.linalg.norm(u)
    xa = np.array([1.0, 0.0, 0.0]); xa = xa - np.dot(xa, u) * u; xa /= np.linalg.norm(xa)
    Rt = np.column_stack([xa, np.cross(u, xa), u])
    Ot = np.array([0.5, 0.1, 0.3])
    w = lambda fx, fy: Ot + Rt @ np.array([fx, fy, 0.0])
    R, O = fit_floor_frame(w(0, 0), w(1, 0), [w(0, 0), w(1, 0), w(0, 1), w(1, 1)])
    fx, fy, fz = to_floor(R, O, w(0.3, 0.7))
    e1 = math.sqrt((fx - 0.3) ** 2 + (fy - 0.7) ** 2 + fz ** 2)
    print(f"floor-frame round-trip error: {e1 * 1000:.4f} mm"); ok &= e1 < 1e-6

    # 2) projector homography recovers a known map
    Htrue = np.array([[0.004, 0.0002, -1.3], [0.0001, 0.0043, -0.95], [0.00002, 0.00003, 1.0]])
    def hp(x, y):
        v = Htrue @ np.array([x, y, 1.0]); return v[0] / v[2], v[1] / v[2]
    px = [(300, 200), (1600, 220), (1550, 880), (280, 860)]
    fl = [hp(x, y) for x, y in px]
    H = projector_homography(px, fl)
    v = H @ np.array([900, 500, 1.0]); got = (v[0] / v[2], v[1] / v[2]); tru = hp(900, 500)
    e2 = math.hypot(got[0] - tru[0], got[1] - tru[1])
    print(f"projector homography fresh-point error: {e2 * 1000:.4f} mm"); ok &= e2 < 1e-4

    # 3) occupancy: a 1x1 m box centred at (1,1) in a 2x2 m room
    grid = rasterize_occupancy([0, 0, 2, 2], 0.05, [[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]])
    inside = grid[20, 20]; outside = grid[2, 2]
    print(f"occupancy: inside={inside} (want 100), outside={outside} (want 0), shape={grid.shape}")
    ok &= inside == 100 and outside == 0

    # 4) PLY export
    import os, tempfile
    p = os.path.join(tempfile.gettempdir(), "room_cloud_selftest.ply")
    write_ply([[0, 0, 0], [1, 2, 3], [4, 5, 6]], p)
    n = sum(1 for _ in open(p))
    print(f"PLY export: {n} lines (7 header + 3 verts = 10)"); ok &= n == 10

    # 5) robot tracker->foot offset round-trip
    off = {"dx": -0.15, "dy": 0.05, "yaw_offset": 0.3, "dz": 0.9}
    foot_true, ryaw_true = (1.0, 2.0), 0.5
    c, s = math.cos(ryaw_true), math.sin(ryaw_true)
    tx = foot_true[0] + c * off["dx"] - s * off["dy"]
    ty = foot_true[1] + s * off["dx"] + c * off["dy"]
    fx, fy, ryaw = apply_robot_offset(tx, ty, ryaw_true + off["yaw_offset"], off)
    e5 = math.hypot(fx - foot_true[0], fy - foot_true[1]) + abs(ryaw - ryaw_true)
    print(f"robot-offset round-trip error: {e5 * 1000:.4f} mm"); ok &= e5 < 1e-9

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--robot-offset", action="store_true", help="calibrate the back-strap tracker -> foot offset")
    ap.add_argument("--calib", default="calib/room_calib.json", help="room calib to load for --robot-offset")
    ap.add_argument("--out", default="calib/room_calib.json")
    ap.add_argument("--ply", default="calib/room_cloud.ply")
    ap.add_argument("--proj-w", type=int, default=1920)
    ap.add_argument("--proj-h", type=int, default=1080)
    ap.add_argument("--monitor-x", type=int, default=1920, help="x-offset of the projector display")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.robot_offset:
        import os
        os.makedirs("calib", exist_ok=True)
        robot_offset("calib/robot_offset.json", a.calib)
        return
    if a.run:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        run(a.out, a.ply, proj_res=(a.proj_w, a.proj_h), monitor_x=a.monitor_x)
    else:
        print("nothing to do; pass --selftest, --run, or --robot-offset")


if __name__ == "__main__":
    main()
