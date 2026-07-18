"""M5 pure pursuit - PURE core (contracts + math + config only, no I/O). The lookahead point
IS the projected setpoint/circle. Built-in closed-loop demo: python -m mrhack.controller.pure_pursuit"""
from __future__ import annotations
import math
from dataclasses import dataclass
from .. import config
from ..contracts import RobotPose, Setpoint, TrajPoint, Trajectory, VelCmd


@dataclass
class PPParams:
    lookahead_m: float = config.LOOKAHEAD_M
    v_max: float = config.V_MAX
    wz_max: float = config.WZ_MAX
    v_deadband: float = config.V_DEADBAND
    wz_deadband: float = config.WZ_DEADBAND
    heading_gain: float = config.HEADING_GAIN
    slow_radius_m: float = config.SLOW_RADIUS_M
    goal_tol_m: float = config.GOAL_TOL_M
    turn_slowdown: float = config.TURN_SLOWDOWN


@dataclass
class PPState:
    traj_id: int = -1
    last_s: float = 0.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _snap(v, db):
    return 0.0 if abs(v) < db else v


def _cum_s(pts):
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y))
    return s


def _point_at_s(pts, s, target):
    for i in range(1, len(s)):
        if s[i] >= target:
            seg = s[i] - s[i - 1]
            f = 0.0 if seg < 1e-9 else (target - s[i - 1]) / seg
            return (pts[i - 1].x + f * (pts[i].x - pts[i - 1].x),
                    pts[i - 1].y + f * (pts[i].y - pts[i - 1].y))
    return (pts[-1].x, pts[-1].y)


def pure_pursuit_step(pose, traj, params, state):
    if traj.traj_id != state.traj_id:
        state = PPState(traj_id=traj.traj_id, last_s=0.0)
    pts = traj.points
    if not pts:
        return (VelCmd(0.0, 0.0, 0.0, pose.t),
                Setpoint(pose.x, pose.y, params.lookahead_m, 0.0, True, pose.t), state)
    s = _cum_s(pts)
    total = s[-1]
    best_i, best_d = None, None
    for i in range(len(pts)):
        if s[i] < state.last_s - 1e-9:
            continue
        d = math.hypot(pts[i].x - pose.x, pts[i].y - pose.y)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    if best_i is None:
        best_i = len(pts) - 1
    s_proj = s[best_i]
    state.last_s = max(state.last_s, s_proj)
    s_la = min(s_proj + params.lookahead_m, total)
    lax, lay = _point_at_s(pts, s, s_la)
    end = pts[-1]
    d_goal = math.hypot(end.x - pose.x, end.y - pose.y)
    if d_goal < params.goal_tol_m:
        return (VelCmd(0.0, 0.0, 0.0, pose.t),
                Setpoint(lax, lay, params.lookahead_m, s_la, True, pose.t), state)
    herr = _wrap(math.atan2(lay - pose.y, lax - pose.x) - pose.yaw)
    wz = max(-params.wz_max, min(params.wz_max, params.heading_gain * herr))
    v = params.v_max
    v *= min(1.0, d_goal / max(params.slow_radius_m, 1e-6))
    v *= max(0.0, 1.0 - params.turn_slowdown * abs(herr) / math.pi)
    return (VelCmd(_snap(v, params.v_deadband), 0.0, _snap(wz, params.wz_deadband), pose.t),
            Setpoint(lax, lay, params.lookahead_m, s_la, False, pose.t), state)


def _demo(start=(0.0, 1.0, 0.0), goal_x=3.0):
    pts = [TrajPoint(x=0.05 * i, y=0.0, heading=0.0) for i in range(int(goal_x / 0.05) + 1)]
    traj = Trajectory(points=pts, traj_id=1, t=0.0)
    params, state = PPParams(), PPState()
    pose = RobotPose(x=start[0], y=start[1], yaw=start[2], t=0.0)
    dt = 1.0 / config.CONTROL_HZ
    steps, done = 0, False
    while steps < 4000:
        cmd, sp, state = pure_pursuit_step(pose, traj, params, state)
        if sp.done:
            done = True
            break
        pose = RobotPose(x=pose.x + cmd.vx * math.cos(pose.yaw) * dt,
                         y=pose.y + cmd.vx * math.sin(pose.yaw) * dt,
                         yaw=_wrap(pose.yaw + cmd.wz * dt), t=pose.t + dt)
        steps += 1
    end = pts[-1]
    err = math.hypot(pose.x - end.x, pose.y - end.y)
    ok = done and err < params.goal_tol_m + 0.05
    verdict = "PASS" if ok else "FAIL"
    print(f"  start={start} -> steps={steps} ({steps * dt:.1f}s) "
          f"final=({pose.x:.2f},{pose.y:.2f}) err={err * 100:.1f}cm  {verdict}")
    return ok


if __name__ == "__main__":
    import sys
    print("PURE-PURSUIT closed-loop convergence demo:")
    results = [_demo((0.0, 1.0, 0.0)), _demo((0.0, -0.8, math.pi)), _demo((-0.5, 0.0, math.pi / 2))]
    print(f"{sum(results)}/{len(results)} cases PASSED")
    sys.exit(0 if all(results) else 1)
