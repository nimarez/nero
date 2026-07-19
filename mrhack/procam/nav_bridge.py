#!/usr/bin/env python3
"""
nav_bridge.py -- bridge Nima's onboard SLAM/nav (ROS 2) to the projector's
robot-relative nav overlay.

WHY robot-relative (the whole point of this file):
  The projector renders the floor scene at the robot's *Vive/floor* pose.
  Nima's navigation lives in the *SLAM world* frame. Co-registering two world
  frames (SLAM <-> floor) is fragile and drifts. Instead we express every nav
  target RELATIVE TO THE ROBOT -- bearing, range, and a body-frame path --
  using the robot's own SLAM pose. follow_circle then re-plants that relative
  geometry at the robot's floor pose. Relative geometry is frame-invariant, so
  NO SLAM<->floor calibration is needed. The only shared assumption is that the
  Vive-derived heading and the SLAM heading measure the same physical
  robot-forward, up to one constant (--heading-offset).

Subscribes (ROS 2):
  /nero/slam/pose                  geometry_msgs/PoseStamped   robot in SLAM world
  /nero/navigation/goal_pose       geometry_msgs/PoseStamped   goal in SLAM world
  /nero/navigation/object_position geometry_msgs/PointStamped  far-phase target (YOLO)
  /nero/navigation/plan            nav_msgs/Path               planned path
  /nero/navigation/status          std_msgs/String             JSON {state,target,standoff}

Writes (atomic, at --rate Hz) the contract follow_circle.load_nav_state expects:
  /run/nero/nav.json
    {version, t, phase, bearing_rel, range, goal_rel, path_rel, status}
    phase "far"  -> bearing_rel (rad, body frame) + range (m)
    phase "close"-> goal_rel [dx,dy] (m, body frame) + path_rel [[dx,dy],...]

Run:
  nav_bridge.py --mock       # no ROS; synthetic nav for projector bring-up
  nav_bridge.py              # real ROS 2 bridge (needs rclpy + msg pkgs)
  nav_bridge.py --selftest   # pure-transform tests, no ROS, no numpy
"""

import argparse
import json
import math
import os
import sys
import time

OUT_PATH = os.environ.get("NERO_NAV_PATH", "/run/nero/nav.json")
RATE_HZ = 10.0
FAR_THRESHOLD_M = 2.0        # goal farther than this with no explicit state -> "far"
NAV_VERSION = 1

# status.state strings we map to a phase (case-insensitive). Unknown -> distance fallback.
FAR_STATES = {"far", "search", "searching", "seek", "seeking",
              "approach", "approaching", "bearing", "coarse", "transit"}
CLOSE_STATES = {"close", "near", "align", "aligning", "final", "final_approach",
                "servo", "servoing", "arrived", "plan", "planning", "slam"}


# --------------------------------------------------------------------------- #
# pure transforms (ROS-free, numpy-free -- so --selftest runs anywhere)        #
# --------------------------------------------------------------------------- #
def _wrap(a):
    """wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(qx, qy, qz, qw):
    """yaw about +z from a quaternion (xyzw). Normalises internally."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def world_to_robot_rel(rx, ry, ryaw, tx, ty):
    """Express world target (tx,ty) in the robot body frame: x=forward, y=left."""
    c, s = math.cos(ryaw), math.sin(ryaw)
    ex, ey = tx - rx, ty - ry
    return (c * ex + s * ey, -s * ex + c * ey)


def _rot(dx, dy, th):
    """rotate a body-frame vector by th (folds in the heading offset)."""
    c, s = math.cos(th), math.sin(th)
    return (c * dx - s * dy, s * dx + c * dy)


def bearing_range(dx, dy):
    return (math.atan2(dy, dx), math.hypot(dx, dy))


def phase_from(status, rng_goal, far_threshold):
    """Prefer an explicit status.state; else fall back to distance to goal."""
    if status:
        st = str(status.get("state", "")).strip().lower()
        if st in FAR_STATES:
            return "far"
        if st in CLOSE_STATES:
            return "close"
    if rng_goal is not None:
        return "far" if rng_goal > far_threshold else "close"
    return "far"


def build_nav_state(now, slam_pose, goal, obj, path, status,
                    heading_offset=0.0, far_threshold=FAR_THRESHOLD_M):
    """
    Compose the robot-relative nav.json dict.

    slam_pose : (rx, ry, ryaw) in SLAM world, or None -> returns None
    goal      : (gx, gy) in SLAM world, or None
    obj       : (ox, oy) far-phase target (YOLO), or None -> falls back to goal
    path      : [(x, y), ...] in SLAM world, or None
    status    : {state, target, standoff} dict, or None
    Returns the dict follow_circle consumes, or None if the robot can't be placed.
    """
    if slam_pose is None:
        return None
    rx, ry, ryaw = slam_pose
    out = {"version": NAV_VERSION, "t": now, "status": status or {}}

    rng_goal = None
    if goal is not None:
        _, rng_goal = bearing_range(*world_to_robot_rel(rx, ry, ryaw, goal[0], goal[1]))

    phase = phase_from(status, rng_goal, far_threshold)
    out["phase"] = phase

    if phase == "far":
        tgt = obj if obj is not None else goal
        if tgt is not None:
            dx, dy = _rot(*world_to_robot_rel(rx, ry, ryaw, tgt[0], tgt[1]), heading_offset)
            br, rng = bearing_range(dx, dy)
            out["bearing_rel"] = _wrap(br)
            out["range"] = rng
    else:  # close
        if goal is not None:
            gx, gy = _rot(*world_to_robot_rel(rx, ry, ryaw, goal[0], goal[1]), heading_offset)
            out["goal_rel"] = [gx, gy]
        if path:
            out["path_rel"] = [
                list(_rot(*world_to_robot_rel(rx, ry, ryaw, px, py), heading_offset))
                for px, py in path
            ]
    return out


# --------------------------------------------------------------------------- #
# atomic writer (matches PR #6's vive_pose.json pattern)                       #
# --------------------------------------------------------------------------- #
def write_atomic(path, data):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# real ROS 2 bridge                                                            #
# --------------------------------------------------------------------------- #
def run_ros(out_path, rate_hz, heading_offset, far_threshold):
    import rclpy                                            # noqa: import-outside-toplevel
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, PointStamped
    from nav_msgs.msg import Path
    from std_msgs.msg import String

    class NavBridge(Node):
        def __init__(self):
            super().__init__("nero_nav_bridge")
            self.slam = None
            self.goal = None
            self.obj = None
            self.path = None
            self.status = None
            self.create_subscription(PoseStamped, "/nero/slam/pose", self._slam, 10)
            self.create_subscription(PoseStamped, "/nero/navigation/goal_pose", self._goal, 10)
            self.create_subscription(PointStamped, "/nero/navigation/object_position", self._obj, 10)
            self.create_subscription(Path, "/nero/navigation/plan", self._path, 10)
            self.create_subscription(String, "/nero/navigation/status", self._status, 10)
            self.create_timer(1.0 / rate_hz, self._tick)
            self.get_logger().info("nav_bridge -> %s @ %.0f Hz" % (out_path, rate_hz))

        @staticmethod
        def _pose_xyyaw(msg):
            p, q = msg.pose.position, msg.pose.orientation
            return (p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w))

        def _slam(self, m):
            self.slam = self._pose_xyyaw(m)

        def _goal(self, m):
            self.goal = (m.pose.position.x, m.pose.position.y)

        def _obj(self, m):
            self.obj = (m.point.x, m.point.y)

        def _path(self, m):
            self.path = [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses]

        def _status(self, m):
            try:
                self.status = json.loads(m.data)
            except (ValueError, TypeError):
                self.status = {"state": str(m.data)}

        def _tick(self):
            nav = build_nav_state(time.time(), self.slam, self.goal, self.obj,
                                  self.path, self.status, heading_offset, far_threshold)
            if nav is not None:
                write_atomic(out_path, nav)

    rclpy.init()
    node = NavBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# mock source (no ROS) -- synthetic nav so the projector loop can be exercised #
# --------------------------------------------------------------------------- #
def run_mock(out_path, rate_hz, heading_offset, far_threshold, duration=None):
    """
    Robot sits at the SLAM origin and slowly sweeps its heading; the goal is a
    fixed world point. The phase clock alternates far/close every 6 s so both
    overlays (bearing ray, then multi-arrow path) get exercised.
    """
    t0 = time.time()
    goal = (3.0, 0.5)
    obj = (3.2, 0.6)
    print("nav_bridge MOCK -> %s (Ctrl-C to stop)" % out_path)
    while True:
        now = time.time()
        el = now - t0
        ryaw = 0.4 * math.sin(el * 0.5)                     # gentle heading sweep
        far = (int(el) // 6) % 2 == 0
        status = {"state": "far" if far else "close", "target": "apple", "standoff": 0.6}
        path = None
        if not far:
            # gentle curve from the robot to the goal, in world frame
            path = [(3.0 * u, 0.5 * u + 0.35 * math.sin(math.pi * u))
                    for u in [i / 10.0 for i in range(11)]]
        nav = build_nav_state(now, (0.0, 0.0, ryaw), goal, obj, path,
                              status, heading_offset, far_threshold)
        write_atomic(out_path, nav)
        if duration is not None and el >= duration:
            return 0
        time.sleep(1.0 / rate_hz)


# --------------------------------------------------------------------------- #
# selftest (pure math -- the bridge<->projector contract)                      #
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and bool(cond)

    dx, dy = world_to_robot_rel(0, 0, 0, 2, 0)
    chk("rel: target straight ahead -> (2,0)", abs(dx - 2) < 1e-9 and abs(dy) < 1e-9)

    dx, dy = world_to_robot_rel(0, 0, math.pi / 2, 1, 0)   # facing +y, world +x is on the RIGHT
    chk("rel: yaw90, world +x -> body (0,-1)", abs(dx) < 1e-9 and abs(dy + 1) < 1e-9)

    br, rng = bearing_range(0, 2)
    chk("bearing_range: (0,2) -> (pi/2, 2)", abs(br - math.pi / 2) < 1e-9 and abs(rng - 2) < 1e-9)

    chk("yaw_from_quat: 90deg about z", abs(_wrap(yaw_from_quat(0, 0, 0.70710678, 0.70710678) - math.pi / 2)) < 1e-6)

    nav = build_nav_state(100.0, (0, 0, 0), (3, 0), (3, 0), None, {"state": "far"})
    chk("build far: bearing 0, range 3",
        nav["phase"] == "far" and abs(nav["bearing_rel"]) < 1e-9 and abs(nav["range"] - 3) < 1e-9)

    nav = build_nav_state(100.0, (0, 0, 0), (2, 0), None, [(1, 0), (2, 0)], {"state": "close"})
    chk("build close: goal_rel + path_rel in body frame",
        nav["phase"] == "close" and nav["goal_rel"] == [2.0, 0.0] and nav["path_rel"][1] == [2.0, 0.0])

    chk("phase fallback: far when goal 5m away",
        build_nav_state(100.0, (0, 0, 0), (5, 0), (5, 0), None, None, far_threshold=2.0)["phase"] == "far")
    chk("phase fallback: close when goal 1m away",
        build_nav_state(100.0, (0, 0, 0), (1, 0), None, [(0.5, 0), (1, 0)], None, far_threshold=2.0)["phase"] == "close")

    nav = build_nav_state(100.0, (0, 0, 0), (2, 0), (2, 0), None, {"state": "far"}, heading_offset=math.pi / 2)
    chk("heading offset rotates bearing +90", abs(_wrap(nav["bearing_rel"] - math.pi / 2)) < 1e-9)

    chk("no slam pose -> None", build_nav_state(100.0, None, (2, 0), None, None, None) is None)

    # THE contract invariant: rel geometry re-planted at the SAME pose recovers the
    # world point. follow_circle.nav_rel_to_floor does exactly this re-plant, so as
    # long as the floor heading == the SLAM heading, the goal lands on the real goal.
    rx, ry, ryaw = 1.0, 2.0, 0.7
    gx, gy = 3.5, -0.5
    bdx, bdy = world_to_robot_rel(rx, ry, ryaw, gx, gy)
    c, s = math.cos(ryaw), math.sin(ryaw)
    fx, fy = rx + c * bdx - s * bdy, ry + s * bdx + c * bdy
    chk("round-trip rel->floor recovers world goal", abs(fx - gx) < 1e-9 and abs(fy - gy) < 1e-9)

    # mock writer produces a valid, fresh, parseable file
    tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "nav_bridge_selftest.json")
    run_mock(tmp, rate_hz=1000.0, heading_offset=0.0, far_threshold=2.0, duration=0.0)
    d = json.load(open(tmp))
    chk("mock file has version+t+phase", d.get("version") == NAV_VERSION and "t" in d and d.get("phase") in ("far", "close"))
    try:
        os.remove(tmp)
    except OSError:
        pass

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Bridge ROS nav -> robot-relative /run/nero/nav.json")
    ap.add_argument("--mock", action="store_true", help="synthetic nav, no ROS")
    ap.add_argument("--selftest", action="store_true", help="pure-transform tests, no ROS")
    ap.add_argument("--out", default=OUT_PATH, help="output nav.json path")
    ap.add_argument("--rate", type=float, default=RATE_HZ, help="write rate (Hz)")
    ap.add_argument("--heading-offset", type=float, default=float(os.environ.get("NERO_NAV_HEADING_OFFSET", "0.0")),
                    help="constant radians added between SLAM-forward and floor-forward")
    ap.add_argument("--far-threshold", type=float, default=FAR_THRESHOLD_M,
                    help="goal distance (m) above which phase defaults to 'far'")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.mock:
        return run_mock(args.out, args.rate, args.heading_offset, args.far_threshold)
    return run_ros(args.out, args.rate, args.heading_offset, args.far_threshold)


if __name__ == "__main__":
    sys.exit(main())
