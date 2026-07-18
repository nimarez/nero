"""M6 projector renderer. Two cores:
  render_floor_canvas(pose, traj, setpoint, bounds) -> BGR image drawn in FLOOR coords
      (black = projector-off; this is the showable/testable part, no projector needed)
  warp_to_projector(canvas, H_proj2floor, ...) -> the same overlay mapped into projector pixels

Self-test:  uv run --with opencv-python --with numpy python -m mrhack.projector.renderer
"""
from __future__ import annotations
import math
import numpy as np


def _P(x, y, bounds, ppm, H):
    xmin, ymin, xmax, ymax = bounds
    return int((x - xmin) * ppm), H - 1 - int((y - ymin) * ppm)   # y up (row 0 = top = ymax)


def render_floor_canvas(pose, traj, setpoint, bounds, ppm=160, safety_r=0.42):
    """Draw path + goal + safety ring + heading arrow + lookahead in floor coords -> BGR image."""
    import cv2
    xmin, ymin, xmax, ymax = bounds
    W = max(2, int((xmax - xmin) * ppm))
    H = max(2, int((ymax - ymin) * ppm))
    img = np.zeros((H, W, 3), np.uint8)
    P = lambda x, y: _P(x, y, bounds, ppm, H)
    if traj and traj.points:
        pts = np.array([P(p.x, p.y) for p in traj.points], np.int32)
        cv2.polylines(img, [pts], False, (230, 160, 80), 3)                       # path
        g = traj.points[-1]
        cv2.drawMarker(img, P(g.x, g.y), (110, 220, 120), cv2.MARKER_STAR, 30, 3)  # goal
    if setpoint:
        cv2.circle(img, P(setpoint.x, setpoint.y), int(0.16 * ppm), (100, 210, 255), -1)  # lookahead
    if pose:
        if setpoint:
            cv2.line(img, P(pose.x, pose.y), P(setpoint.x, setpoint.y), (100, 210, 255), 2)
        cv2.circle(img, P(pose.x, pose.y), int(safety_r * ppm), (40, 150, 255), 3)         # safety ring
        hx, hy = pose.x + 0.42 * math.cos(pose.yaw), pose.y + 0.42 * math.sin(pose.yaw)
        cv2.arrowedLine(img, P(pose.x, pose.y), P(hx, hy), (255, 230, 130), 4, tipLength=0.3)  # heading
    return img


def warp_to_projector(floor_canvas, H_proj2floor, bounds, ppm, proj_res):
    """Map the floor-pixel canvas into projector pixels via inv(H_proj2floor)."""
    import cv2
    xmin, ymin, xmax, ymax = bounds
    A = np.array([[1.0 / ppm, 0, xmin], [0, -1.0 / ppm, ymax], [0, 0, 1.0]])   # canvas_px -> floor_m
    M = np.linalg.inv(np.asarray(H_proj2floor, float)) @ A                      # canvas_px -> projector_px
    return cv2.warpPerspective(floor_canvas, M, tuple(proj_res))


def _selftest():
    from ..contracts import RobotPose, Setpoint, TrajPoint, Trajectory
    traj = Trajectory([TrajPoint(0.05 * i, 0.0, 0.0) for i in range(40)], 1, 0.0)
    pose = RobotPose(0.5, 0.3, 0.2, 0.0)
    sp = Setpoint(1.0, 0.1, 0.5, 1.0, False, 0.0)
    bounds = [-0.3, -0.6, 2.2, 0.8]
    img = render_floor_canvas(pose, traj, sp, bounds)
    assert img.ndim == 3 and img.max() > 0, "floor canvas is empty"
    H = [[0.004, 0.0002, -1.3], [0.0001, 0.0043, -0.95], [0.00002, 0.00003, 1.0]]
    proj = warp_to_projector(img, H, bounds, 160, (1920, 1080))
    assert proj.shape == (1080, 1920, 3), proj.shape
    print(f"floor canvas {img.shape}, drawn px {int((img.max(2) > 0).sum())}; projector warp {proj.shape}")
    print("SELFTEST: PASS")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
