#!/usr/bin/env python
"""Render the mrhack closed-loop sim to an animated GIF (top-down floor view).
Drives the REAL verified loop (plan -> pure-pursuit -> kinematic sim) and draws what the
projector/Rerun overlay will show: the robot + heading, a safety ring around it, the
planned path, and the moving lookahead setpoint (the 'projected circle').

Run:  uv run --with matplotlib --with pillow python starters/sim_viz.py --out sim.gif
"""
from __future__ import annotations
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -> import mrhack

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from mrhack.contracts import Goal, RobotPose
from mrhack.controller.pure_pursuit import PPParams, PPState, pure_pursuit_step
from mrhack.planner.plan import plan
from mrhack.sim.kinematic_sim import KinematicSim


def record(start=(0.0, 1.2, 0.0), goal_xy=(3.0, 0.0)):
    sim = KinematicSim(RobotPose(start[0], start[1], start[2], 0.0))
    goal = Goal(goal_xy[0], goal_xy[1], "wrench", 0.0)
    traj = plan(sim.pose, goal, traj_id=1)
    params, state = PPParams(), PPState()
    dt = 1.0 / 30.0
    frames = []
    for _ in range(2000):
        cmd, sp, state = pure_pursuit_step(sim.pose, traj, params, state)
        frames.append((sim.pose.x, sim.pose.y, sim.pose.yaw, sp.x, sp.y, sp.done))
        if sp.done:
            break
        sim.step(cmd, dt)
    return traj, goal, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim.gif")
    a = ap.parse_args()

    traj, goal, frames = record()
    tx = [p.x for p in traj.points]
    ty = [p.y for p in traj.points]
    step = max(1, len(frames) // 60)
    frames = frames[::step]

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=115)
    fig.patch.set_facecolor("#0e1014")
    ax.set_facecolor("#0e1014")
    allx = tx + [f[0] for f in frames]
    ally = ty + [f[1] for f in frames]
    ax.set_xlim(min(allx) - 0.7, max(allx) + 0.7)
    ax.set_ylim(min(ally) - 0.9, max(ally) + 0.9)
    ax.set_aspect("equal")
    ax.set_title("mrhack sim  ·  plan → pure-pursuit → walk  (top-down floor)", color="w", fontsize=10)
    ax.tick_params(colors="#666")
    for s in ax.spines.values():
        s.set_color("#2a2f3a")

    ax.plot(tx, ty, "--", color="#3a7bd5", lw=1.6, alpha=0.55, label="planned path")
    ax.plot(goal.x, goal.y, "*", color="#2ecc71", ms=20, label="goal (wrench)")

    trail, = ax.plot([], [], "-", color="#5aa9ff", lw=2.2, alpha=0.8)
    robot_dot, = ax.plot([], [], "o", color="#5aa9ff", ms=9)
    heading, = ax.plot([], [], "-", color="#5aa9ff", lw=3)
    safety = Circle((0, 0), 0.42, fill=False, color="#ff8c2a", lw=2.0, alpha=0.9)
    ax.add_patch(safety)
    setpoint_dot, = ax.plot([], [], "o", color="#ffd166", ms=8, label="lookahead (the projected circle)")
    lookline, = ax.plot([], [], ":", color="#ffd166", lw=1.3, alpha=0.85)
    ax.legend(loc="upper left", facecolor="#161a22", edgecolor="#2a2f3a", labelcolor="w", fontsize=8)

    trx, trry = [], []

    def upd(i):
        x, y, yaw, sx, sy, done = frames[i]
        trx.append(x); trry.append(y)
        trail.set_data(trx, trry)
        robot_dot.set_data([x], [y])
        heading.set_data([x, x + 0.34 * math.cos(yaw)], [y, y + 0.34 * math.sin(yaw)])
        safety.center = (x, y)
        setpoint_dot.set_data([sx], [sy])
        lookline.set_data([x, sx], [y, sy])
        return trail, robot_dot, heading, safety, setpoint_dot, lookline

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=60, blit=False)
    anim.save(a.out, writer=PillowWriter(fps=18))
    print(f"wrote {a.out}  ({len(frames)} frames, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
