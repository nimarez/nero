#!/usr/bin/env python
"""STARTER (needs hardware) - Intel RealSense @ the projector: dense point cloud + depth deprojection.

Co-located with the projector (3/4, ~7-8 ft), the RealSense gives:
  - a DENSE point cloud of the room (depth -> 3D)  [the point cloud you wanted, from the projection view]
  - object FLOOR positions via depth deprojection (pixel + depth -> 3D -> floor)  [feeds M3a Goal]
  - procam verification (it sees exactly what the projector paints)

Point-cloud capture (needs the RealSense):
    uv run --with pyrealsense2 --with numpy python starters/realsense_capture.py --cloud --out calib/realsense_cloud.ply
Verify the deprojection math (no hardware):
    uv run --with numpy python starters/realsense_capture.py --selftest

STATUS: deprojection math self-tested; the RealSense I/O is written, NOT hardware-verified.
The camera->floor extrinsic (from calibration) turns camera-frame points into the shared floor
frame; that hook is marked with a TODO below.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np


def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel (u, v) + depth z (m) -> 3D point in the camera frame (m)."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


def cloud_from_depth(depth_m, fx, fy, cx, cy, stride=3, zmin=0.15, zmax=6.0):
    """Vectorised deprojection of a depth image -> Nx3 camera-frame points."""
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    z = depth_m[::stride, ::stride]
    valid = (z > zmin) & (z < zmax)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x[valid], y[valid], z[valid]], axis=1)


def write_ply(pts, path):
    pts = np.asarray(pts, float).reshape(-1, 3)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    return path


def capture_cloud(out, stride=3):
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            pipe.wait_for_frames()                      # let auto-exposure settle
        frames = align.process(pipe.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
        pts = cloud_from_depth(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy, stride)
        write_ply(pts, out)
        print(f"wrote {out}: {len(pts)} points  (intrinsics fx={intr.fx:.1f} cx={intr.ppx:.1f})")
        # TODO: floor_pts = (R_cam2floor @ pts.T).T + t_cam2floor   (extrinsic from calibration)
        #       -> feed the occupancy grid + the Rerun scene, and deproject a detection's pixel -> Goal.
    finally:
        pipe.stop()


def _selftest():
    fx, fy, cx, cy = 600.0, 600.0, 424.0, 240.0
    p3 = np.array([0.3, -0.2, 2.5])
    u = fx * p3[0] / p3[2] + cx
    v = fy * p3[1] / p3[2] + cy
    back = deproject(u, v, p3[2], fx, fy, cx, cy)
    err = float(np.linalg.norm(back - p3))
    print(f"deproject round-trip: {err * 1000:.4f} mm")
    # a tiny synthetic depth image -> cloud
    depth = np.full((20, 30), 2.0, np.float32)
    depth[0, 0] = 0.0                                    # invalid pixel dropped
    pts = cloud_from_depth(depth, fx, fy, cx, cy, stride=1)
    print(f"cloud_from_depth: {len(pts)} pts (want {20 * 30 - 1})")
    ok = err < 1e-6 and len(pts) == 20 * 30 - 1
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="calib/realsense_cloud.ply")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.cloud:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        capture_cloud(a.out, a.stride)
    else:
        print("pass --cloud or --selftest")


if __name__ == "__main__":
    main()
