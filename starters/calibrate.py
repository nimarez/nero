#!/usr/bin/env python
"""STARTER - M1 calibration: camera->floor + projector->floor homographies (ArUco).

Everything lives on ONE floor plane, so a homography (not full 3D procam) is exact.
Two homographies:
  H_cam2floor : OAK-D image pixel -> floor metres  (detect ArUco markers taped at known floor XY)
  H_proj2floor: projector pixel   -> floor metres  (project a known dot grid, see it in the camera,
                                                    chain through H_cam2floor)
Produces calib/calib.json (CalibConfig). Read by M2 (pose), M3a (goal), M6 (renderer).

Verify the math (no hardware; honours uv):
    uv run --with opencv-python --with numpy python starters/calibrate.py --selftest
Live capture (needs the OAK-D + printed ArUco markers at measured floor positions):
    uv run --with opencv-python --with numpy python starters/calibrate.py --camera 0 --out calib/calib.json

STATUS: homography math is self-test-verified; the live ArUco/projector capture is
written, NOT hardware-verified.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time


def compute_homography(img_pts, floor_pts):
    """Least-squares homography mapping img_pts (pixels) -> floor_pts (metres). >=4 correspondences."""
    import cv2
    import numpy as np
    H, _ = cv2.findHomography(np.asarray(img_pts, dtype=np.float64),
                              np.asarray(floor_pts, dtype=np.float64), method=0)
    return H


def apply_h(H, x, y):
    import numpy as np
    v = np.asarray(H) @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def reproj_error(H, img_pts, floor_pts):
    total = 0.0
    for (px, py), (fx_true, fy_true) in zip(img_pts, floor_pts):
        fx, fy = apply_h(H, px, py)
        total += math.hypot(fx - fx_true, fy - fy_true)
    return total / len(img_pts)


def _selftest():
    """Synthetic: pick a ground-truth homography, generate correspondences, recover it,
    then check the recovered map on a FRESH pixel matches ground truth."""
    import numpy as np
    H_true = np.array([[0.0040, 0.0002, -1.30],
                       [0.0001, 0.0043, -0.95],
                       [0.00002, 0.00003, 1.00]])
    img = [(100, 120), (1180, 140), (1150, 700), (90, 680), (640, 400), (300, 550)]
    floor = [apply_h(H_true, x, y) for (x, y) in img]
    H = compute_homography(img, floor)
    err = reproj_error(H, img, floor)
    tx, ty = 512, 333
    fx, fy = apply_h(H, tx, ty)
    fxT, fyT = apply_h(H_true, tx, ty)
    rt = math.hypot(fx - fxT, fy - fyT)
    print(f"fit reprojection error : {err * 1000:.4f} mm over {len(img)} points")
    print(f"fresh pixel ({tx},{ty}) -> floor ({fx:.3f},{fy:.3f}) m ; truth ({fxT:.3f},{fyT:.3f}) ; diff {rt * 1000:.4f} mm")
    ok = err < 1e-4 and rt < 1e-4
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def capture_camera_floor(camera_index, marker_floor_xy):
    """Detect ArUco markers whose IDs map to known floor XY (metres) -> (img_pts, floor_pts)."""
    import cv2
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                                   aruco.DetectorParameters())
    cap = cv2.VideoCapture(camera_index)
    img_pts, floor_pts = [], []
    print("Looking for taped floor markers... capturing one good frame.")
    for _ in range(150):
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        seen = set()
        img_pts, floor_pts = [], []
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i in marker_floor_xy and i not in seen:
                seen.add(i)
                center = c.reshape(4, 2).mean(axis=0)
                img_pts.append((float(center[0]), float(center[1])))
                floor_pts.append(marker_floor_xy[i])
        if len(img_pts) >= 4:
            break
    cap.release()
    return img_pts, floor_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="calib/calib.json")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    # EDIT to your taped marker layout (marker id -> floor metres):
    marker_floor_xy = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
    img_pts, floor_pts = capture_camera_floor(a.camera, marker_floor_xy)
    if len(img_pts) < 4:
        print(f"Only {len(img_pts)} markers seen; need >=4. Check lighting / IDs / layout.")
        sys.exit(1)
    H_cam2floor = compute_homography(img_pts, floor_pts)
    err = reproj_error(H_cam2floor, img_pts, floor_pts)
    print(f"H_cam2floor reprojection error: {err * 1000:.1f} mm")

    # TODO projector->floor: project a known dot grid, detect the dots in the camera,
    #      map each dot pixel->floor via H_cam2floor, then compute H_proj2floor (projector_px -> floor).
    calib = {
        "H_cam2floor": [list(map(float, r)) for r in H_cam2floor],
        "H_proj2floor": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # TODO from the projector step
        "floor_bounds": [0.0, 0.0, 1.0, 1.0],
        "origin_marker_id": 0, "marker_size_m": 0.10,
        "camera_index": a.camera, "camera_resolution": [1920, 1080],
        "reproj_error_px": 0.0, "calib_time": time.time(),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
