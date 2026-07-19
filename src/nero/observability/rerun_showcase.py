#!/usr/bin/env python3
"""
rerun_showcase.py -- the "digital twin" Rerun visualization for the K1 demo.

One 3D scene, scrubbable on Rerun's timeline:
  - the ROOM as a photoreal Gaussian-splat point cloud (Marble by World Labs
    export, or any colored .ply) -> falls back to a synthetic room if absent
  - the K1 walking through the twin at its live Vive/floor pose
  - the ANSI safety envelope on the floor, which INFLATES + reddens the instant
    tracking/detection confidence drops (the functional, not decorative, bit)
  - the planned path as a glowing ribbon (far: bearing ray, close: A* path)
  - the head-gaze cone locking onto the YOLO-detected target
  - live speed / safety-radius / confidence time-series

Modes:
  python rerun_showcase.py --mock --spawn        # synthetic run, opens the viewer
  python rerun_showcase.py --mock --save out.rrd # headless -> shareable recording
  python rerun_showcase.py --live --spawn        # on the box: reads /run/nero/*.json
  ... --room path/to/new_room.ply                # drop the Marble splat in here

The room .ply and the live JSON are optional; every mode degrades gracefully so
the scene always renders something compelling.
"""

import argparse
import gzip
import json
import math
import os
import struct
import time

import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover
    raise SystemExit("pip/uv install rerun-sdk  (uv run --with rerun-sdk --with numpy ...)")

# ---- captured-from-hardware constants (from the projector calibration bundle) ----
BOX_GOAL = (1.4585, 0.4638)          # the real box goal, floor metres
TAG_IDS = (1, 2, 3, 4)
TAG_SIZE_M = 0.13
HEAD_HEIGHT_M = 1.15                  # K1 head above the floor

# ---- palette (RGB) ----
C_ROBOT = (70, 130, 180)
C_HEADING = (255, 210, 0)
C_PATH = (0, 255, 90)
C_BEARING = (0, 220, 255)
C_GAZE = (0, 235, 255)
C_BARRIER = (255, 140, 0)
C_TAG = (240, 240, 240)
C_TARGET_LOCK = (0, 255, 90)
C_TARGET_SEEK = (255, 70, 70)

# ---- ANSI keep-out sizing ----
REACH_M = 0.60
STOP_PER_MS = 0.55                   # stopping distance per m/s of speed
UNCERT_INFLATE_M = 0.85              # extra radius as confidence -> 0
BARRIER_MARGIN_M = 0.45


# --------------------------------------------------------------------------- #
# version-robust Rerun shims (the SDK renames these across releases)           #
# --------------------------------------------------------------------------- #
def _set_time(seconds):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", seconds)
    else:                                             # Rerun >= 0.23
        rr.set_time("t", duration=seconds)


def _scalar(value):
    return rr.Scalars(value) if hasattr(rr, "Scalars") else rr.Scalar(value)


def _lerp(a, b, u):
    return tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def circle(cx, cy, r, z=0.012, n=96):
    a = np.linspace(0, 2 * math.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)])


def yaw_mat(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def safety_radius(speed, conf):
    return REACH_M + STOP_PER_MS * max(0.0, speed) + UNCERT_INFLATE_M * (1.0 - conf)


# --------------------------------------------------------------------------- #
# room: Marble/splat .ply, else synthetic                                      #
# --------------------------------------------------------------------------- #
def _read_ply(path):
    """.ply (colored cloud or 3DGS f_dc) -> raw (xyz, rgb) or None."""
    try:
        from plyfile import PlyData
    except ImportError:
        print("  room: install plyfile for .ply (uv --with plyfile) -> synthetic")
        return None
    v = PlyData.read(path)["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    names = set(v.dtype.names)
    if {"red", "green", "blue"} <= names:
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)
        rgb = np.clip((0.5 + 0.28209479 * dc) * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 190, np.uint8)
    return xyz, rgb


def _read_spz(path):
    """Niantic SPZ (gzipped Gaussian splat) -> raw (xyz, rgb) or None.
    Decodes 24-bit fixed-point positions + DC colors; ignores scale/rot/alpha/SH."""
    raw = open(path, "rb").read()
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    magic, _ver, n = struct.unpack_from("<III", data, 0)
    if magic != 0x5053474E:
        print("  room: not an SPZ file (bad magic) -> synthetic")
        return None
    _sh, frac, _flags, _res = struct.unpack_from("<BBBB", data, 12)
    pos = np.frombuffer(data, np.uint8, count=n * 9, offset=16).reshape(n, 3, 3).astype(np.int32)
    vals = pos[:, :, 0] | (pos[:, :, 1] << 8) | (pos[:, :, 2] << 16)     # 24-bit LE
    vals = np.where(vals >= (1 << 23), vals - (1 << 24), vals)          # sign
    xyz = (vals.astype(np.float32) / (1 << frac))
    col_off = 16 + n * 9 + n * 3 + n * 3 + n * 1                        # pos+scale+rot+alpha
    col = np.frombuffer(data, np.uint8, count=n * 3, offset=col_off).reshape(n, 3).astype(np.float32)
    f = (col / 255.0 - 0.5) / 0.15                                      # SPZ DC dequant
    rgb = (np.clip(0.5 + 0.28209479 * f, 0, 1) * 255).astype(np.uint8)
    print("  room: decoded SPZ %d gaussians (%d frac bits)" % (n, frac))
    return xyz, rgb


def _finalize_room(raw, up_axis, scale, max_points):
    if raw is None:
        return None
    xyz, rgb = raw
    xyz = xyz.astype(np.float32)
    if up_axis == "y":                                     # Y-up splat -> Z-up world
        xyz = np.column_stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]])
    xyz = xyz * scale
    xyz[:, :2] -= np.median(xyz[:, :2], axis=0)            # centre the room
    xyz[:, 2] -= np.percentile(xyz[:, 2], 1.0)             # drop the floor to z=0
    if len(xyz) > max_points:                              # deterministic subsample
        step = len(xyz) // max_points
        xyz, rgb = xyz[::step], rgb[::step]
    ext = xyz.max(0) - xyz.min(0)
    print("  room: %d pts | extent X=%.1f Y=%.1f Z=%.1f m (Z should be the vertical ~2-3m)"
          % (len(xyz), ext[0], ext[1], ext[2]))
    return xyz, rgb


def load_room(path, up_axis="y", scale=1.0, max_points=300_000):
    """Load a room backdrop from .spz (Marble/Niantic) or .ply -> (xyz, rgb), floor at z=0."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        if fh.read(48).startswith(b"version https://git-lfs"):
            print("  room: Git LFS pointer (run `git lfs pull`) -> synthetic")
            return None
    ext = os.path.splitext(path)[1].lower()
    raw = _read_spz(path) if ext == ".spz" else _read_ply(path)
    return _finalize_room(raw, up_axis, scale, max_points)


def synthetic_room(extent=3.2):
    """A believable stand-in until the Marble splat lands: speckled floor + walls."""
    rng = np.random.default_rng(7)
    n = 24_000
    floor = np.column_stack([rng.uniform(-extent, extent, n), rng.uniform(-extent, extent, n),
                             rng.uniform(0.0, 0.02, n)])
    fcol = np.tile(np.array([60, 60, 68], np.uint8), (n, 1))
    walls = []
    for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        m = 9000
        along = rng.uniform(-extent, extent, m)
        h = rng.uniform(0, 2.2, m)
        if sx:
            walls.append(np.column_stack([np.full(m, sx * extent), along, h]))
        else:
            walls.append(np.column_stack([along, np.full(m, sy * extent), h]))
    wall = np.vstack(walls)
    wcol = np.tile(np.array([90, 96, 110], np.uint8), (len(wall), 1))
    xyz = np.vstack([floor, wall]).astype(np.float32)
    rgb = np.vstack([fcol, wcol])
    return xyz, rgb


def log_room(room):
    xyz, rgb = room
    rr.log("world/room", rr.Points3D(xyz, colors=rgb, radii=0.012), static=True)
    # the 4 ArUco floor markers that define the room frame
    corners = np.array([[-1.2, -0.9], [1.2, -0.9], [1.2, 0.9], [-1.2, 0.9]])
    for i, (cx, cy) in zip(TAG_IDS, corners):
        h = TAG_SIZE_M / 2
        sq = np.array([[cx - h, cy - h, 0.006], [cx + h, cy - h, 0.006],
                       [cx + h, cy + h, 0.006], [cx - h, cy + h, 0.006],
                       [cx - h, cy - h, 0.006]])
        rr.log("world/tags/id_%d" % i, rr.LineStrips3D([sq], colors=C_TAG, radii=0.006), static=True)


# --------------------------------------------------------------------------- #
# per-frame scene                                                              #
# --------------------------------------------------------------------------- #
def log_robot(x, y, yaw):
    rr.log("world/robot", rr.Transform3D(translation=[x, y, 0.0], mat3x3=yaw_mat(yaw)))
    rr.log("world/robot/body", rr.Boxes3D(centers=[[0, 0, 0.55]], half_sizes=[[0.16, 0.11, 0.55]],
                                           colors=C_ROBOT))
    rr.log("world/robot/head", rr.Points3D([[0.0, 0.0, HEAD_HEIGHT_M]], colors=C_ROBOT, radii=0.09))
    rr.log("world/robot/heading", rr.Arrows3D(origins=[[0, 0, 0.05]], vectors=[[0.6, 0, 0]],
                                              colors=C_HEADING, radii=0.02))


def log_safety(x, y, r_inner, conf):
    ring_col = _lerp(C_TARGET_SEEK, C_PATH, conf)      # red when unsure -> green when confident
    rr.log("world/safety/keepout", rr.LineStrips3D([circle(x, y, r_inner)], colors=ring_col,
                                                    radii=0.03))
    rr.log("world/safety/barrier", rr.LineStrips3D([circle(x, y, r_inner + BARRIER_MARGIN_M)],
                                                    colors=C_BARRIER, radii=0.02))


def log_nav(pose, phase, goal, path_pts):
    x, y, yaw = pose
    if phase == "far":
        rr.log("world/nav/path", rr.Clear(recursive=False))
        rr.log("world/nav/bearing", rr.Arrows3D(origins=[[x, y, 0.03]],
                                                vectors=[[goal[0] - x, goal[1] - y, 0.0]],
                                                colors=C_BEARING, radii=0.025))
    else:
        rr.log("world/nav/bearing", rr.Clear(recursive=False))
        pts = np.array([[px, py, 0.02] for px, py in path_pts])
        rr.log("world/nav/path", rr.LineStrips3D([pts], colors=C_PATH, radii=0.03))


def log_target(goal, conf):
    lock = conf > 0.6
    col = C_TARGET_LOCK if lock else C_TARGET_SEEK
    rr.log("world/target", rr.Boxes3D(centers=[[goal[0], goal[1], 0.10]],
                                      half_sizes=[[0.10, 0.10, 0.10]], colors=col,
                                      labels=["box  %.0f%%" % (conf * 100)]))


def log_gaze(x, y, yaw, goal, conf):
    head = np.array([x, y, HEAD_HEIGHT_M])
    tgt = np.array([goal[0], goal[1], 0.10])
    col = _lerp(C_TARGET_SEEK, C_GAZE, conf)
    rr.log("world/robot/gaze", rr.Arrows3D(origins=[head], vectors=[tgt - head],
                                           colors=col, radii=0.012))


def log_plots(speed, r_inner, conf):
    rr.log("plot/speed_mps", _scalar(speed))
    rr.log("plot/safety_radius_m", _scalar(r_inner))
    rr.log("plot/detect_confidence", _scalar(conf))


# --------------------------------------------------------------------------- #
# drivers                                                                      #
# --------------------------------------------------------------------------- #
def _smooth_path(p0, p1, n):
    """S-curve from p0 to p1 (ease-in-out), returns Nx2."""
    u = np.linspace(0, 1, n)
    ease = u * u * (3 - 2 * u)
    mid = (np.array(p0) + np.array(p1)) / 2 + np.array([0.0, 1.1])   # bow the path
    a = np.outer(1 - ease, p0) + np.outer(ease, mid)
    b = np.outer(1 - ease, mid) + np.outer(ease, p1)
    return np.outer(1 - ease, [1, 1]) * a + np.outer(ease, [1, 1]) * b


def run_mock(goal=BOX_GOAL, seconds=12.0, fps=20):
    n = int(seconds * fps)
    traj = _smooth_path((-2.3, -1.6), (goal[0] - 0.5, goal[1] - 0.3), n)
    prev = None
    for i in range(n):
        t = i / fps
        _set_time(t)
        x, y = traj[i]
        nx, ny = traj[min(i + 1, n - 1)]
        yaw = math.atan2(ny - y, nx - x)
        speed = 0.0 if prev is None else math.hypot(x - prev[0], y - prev[1]) * fps
        prev = (x, y)

        dist = math.hypot(goal[0] - x, goal[1] - y)
        # confidence: seeking, a mid-run occlusion dip (ring inflates + reddens), then locks
        conf = 0.35 + 0.55 * max(0.0, 1 - dist / 4.0)
        if 4.0 < t < 5.5:
            conf = 0.15
        conf = float(np.clip(conf, 0.1, 0.95))
        phase = "far" if dist > 1.6 else "close"
        r_inner = safety_radius(speed, conf)

        log_robot(x, y, yaw)
        log_safety(x, y, r_inner, conf)
        log_nav((x, y, yaw), phase, goal, traj[i:])
        log_target(goal, conf)
        log_gaze(x, y, yaw, goal, conf)
        log_plots(speed, r_inner, conf)


def _read_json(path, max_age=0.3):
    try:
        d = json.load(open(path))
        if time.time() - float(d.get("t", d.get("timestamp", 0))) > max_age:
            return None
        return d
    except Exception:
        return None


def run_live(pose_path="/run/nero/vive_pose.json", nav_path="/run/nero/nav.json",
             goal=BOX_GOAL, hz=30, duration=None):
    print("  live: reading %s + %s (Ctrl-C to stop)" % (pose_path, nav_path))
    t0 = time.time()
    prev = None
    while True:
        now = time.time()
        _set_time(now - t0)
        p = _read_json(pose_path)
        if p and p.get("tracking_valid", True) and p.get("position"):
            x, y = float(p["position"][0]), float(p["position"][1])
            q = p.get("quaternion_xyzw", [0, 0, 0, 1])
            yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]),
                             1 - 2 * (q[1] ** 2 + q[2] ** 2))
            speed = 0.0 if prev is None else math.hypot(x - prev[0], y - prev[1]) * hz
            prev = (x, y)
            nav = _read_json(nav_path, max_age=0.5) or {}
            conf = float(nav.get("status", {}).get("confidence", 0.8))
            phase = nav.get("phase", "far")
            g = nav.get("goal") or goal
            r_inner = safety_radius(speed, conf)
            log_robot(x, y, yaw)
            log_safety(x, y, r_inner, conf)
            log_nav((x, y, yaw), phase, g, nav.get("path", [[x, y], list(g)]))
            log_target(g, conf)
            log_gaze(x, y, yaw, g, conf)
            log_plots(speed, r_inner, conf)
        if duration and now - t0 > duration:
            return
        time.sleep(1.0 / hz)


def send_blueprint():
    try:
        import rerun.blueprint as rrb
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Digital twin"),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="plot", name="Live control state"),
                ),
                column_shares=[3, 1],
            ),
            rrb.BlueprintPanel(state="collapsed"),
        ))
    except Exception as e:                              # blueprint API varies -> non-fatal
        print("  (blueprint skipped: %s)" % e)


def main():
    ap = argparse.ArgumentParser(description="Rerun digital-twin showcase for the K1 demo")
    ap.add_argument("--mock", action="store_true", help="synthetic run (default)")
    ap.add_argument("--live", action="store_true", help="read /run/nero/*.json on the box")
    ap.add_argument("--room", default=os.environ.get("NERO_ROOM_PLY", ""),
                    help="Marble/splat .ply for the room backdrop")
    ap.add_argument("--up-axis", default="y", choices=["y", "z"], help="splat up-axis")
    ap.add_argument("--room-scale", type=float, default=1.0)
    ap.add_argument("--spawn", action="store_true", help="open the Rerun viewer")
    ap.add_argument("--save", default="", help="write a shareable .rrd instead")
    ap.add_argument("--seconds", type=float, default=12.0, help="mock duration")
    args = ap.parse_args()

    rr.init("nero_digital_twin")
    if args.save:
        rr.save(args.save)
    elif args.spawn:
        rr.spawn()
    else:
        rr.save("nero_twin_demo.rrd")
        args.save = "nero_twin_demo.rrd"

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    send_blueprint()
    room = load_room(args.room, args.up_axis, args.room_scale) or synthetic_room()
    log_room(room)

    if args.live:
        run_live(duration=args.seconds if args.save else None)
    else:
        run_mock(seconds=args.seconds)

    if args.save:
        print("saved -> %s   (open with:  rerun %s )" % (args.save, args.save))


if __name__ == "__main__":
    main()
