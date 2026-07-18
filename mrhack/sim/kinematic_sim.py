"""Kinematic unicycle sim: vel_cmd -> integrate -> pose. Optional scale/latency replay the
velocity-ID imperfections so gains are tuned against realistic tracking.

__main__ runs the FULL in-process control chain with NO hardware and NO bus:
    plan (M4) -> pure_pursuit (M5) -> KinematicSim -> new pose -> repeat.
Run:  python -m mrhack.sim.kinematic_sim
"""
from __future__ import annotations
import math
from .. import config
from ..contracts import Goal, RobotPose, VelCmd


class KinematicSim:
    def __init__(self, pose, scale=1.0, latency_steps=0):
        self.pose = pose
        self.scale = scale
        self._delay = [VelCmd(0.0, 0.0, 0.0, 0.0)] * max(0, latency_steps)

    def step(self, cmd, dt):
        if self._delay:
            self._delay.append(cmd)
            cmd = self._delay.pop(0)
        vx, vy, wz = cmd.vx * self.scale, cmd.vy * self.scale, cmd.wz * self.scale
        p = self.pose
        nx = p.x + (vx * math.cos(p.yaw) - vy * math.sin(p.yaw)) * dt
        ny = p.y + (vx * math.sin(p.yaw) + vy * math.cos(p.yaw)) * dt
        nyaw = (p.yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        self.pose = RobotPose(x=nx, y=ny, yaw=nyaw, t=p.t + dt)
        return self.pose


def _closed_loop(start=(0.0, 0.0, 0.0), goal_xy=(2.5, 1.5), scale=1.0, latency_steps=0):
    from ..controller.pure_pursuit import PPParams, PPState, pure_pursuit_step
    from ..planner.plan import plan
    sim = KinematicSim(RobotPose(start[0], start[1], start[2], 0.0), scale=scale, latency_steps=latency_steps)
    goal = Goal(goal_xy[0], goal_xy[1], "demo", 0.0)
    traj = plan(sim.pose, goal, traj_id=1)
    params, state = PPParams(), PPState()
    dt = 1.0 / config.CONTROL_HZ
    steps, done = 0, False
    while steps < 4000:
        cmd, sp, state = pure_pursuit_step(sim.pose, traj, params, state)
        if sp.done:
            done = True
            break
        sim.step(cmd, dt)
        steps += 1
    err = math.hypot(sim.pose.x - goal.x, sim.pose.y - goal.y)
    ok = done and err < params.goal_tol_m + 0.05
    print(f"  start={start} goal={goal_xy} scale={scale} lat={latency_steps} -> "
          f"steps={steps} ({steps * dt:.1f}s) final=({sim.pose.x:.2f},{sim.pose.y:.2f}) "
          f"err={err * 100:.1f}cm  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    print("FULL closed loop: plan (M4) -> pure-pursuit (M5) -> kinematic sim (no hardware, no bus):")
    results = [
        _closed_loop((0.0, 0.0, 0.0), (2.5, 1.5)),
        _closed_loop((0.0, 0.0, math.pi), (-1.5, 2.0)),          # start facing away
        _closed_loop((0.0, 0.0, 0.0), (2.0, 0.0), scale=0.8),     # velocity under-tracking
        _closed_loop((0.0, 0.0, 0.0), (2.0, 1.0), latency_steps=4),  # ~130 ms actuation latency
    ]
    print(f"{sum(results)}/{len(results)} closed-loop cases PASSED")
    sys.exit(0 if all(results) else 1)
