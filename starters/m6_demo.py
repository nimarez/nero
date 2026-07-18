#!/usr/bin/env python
"""Render the sim THROUGH the M6 projector renderer -> a GIF of the floor overlay
(exactly what the projector paints on the floor: path, goal, safety ring, heading, moving circle).

Run:  uv run --with opencv-python --with numpy --with pillow python starters/m6_demo.py --out m6_floor.gif
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from mrhack.contracts import Goal, RobotPose
from mrhack.controller.pure_pursuit import PPParams, PPState, pure_pursuit_step
from mrhack.planner.plan import plan
from mrhack.projector.renderer import render_floor_canvas
from mrhack.sim.kinematic_sim import KinematicSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m6_floor.gif")
    a = ap.parse_args()

    sim = KinematicSim(RobotPose(0.0, 1.2, 0.0, 0.0))
    goal = Goal(3.0, 0.0, "wrench", 0.0)
    traj = plan(sim.pose, goal, traj_id=1)
    params, state = PPParams(), PPState()
    dt = 1.0 / 30.0
    steps = []
    for _ in range(2000):
        cmd, sp, state = pure_pursuit_step(sim.pose, traj, params, state)
        steps.append((sim.pose, sp))
        if sp.done:
            break
        sim.step(cmd, dt)

    xs = [p.x for p in traj.points] + [s[0].x for s in steps]
    ys = [p.y for p in traj.points] + [s[0].y for s in steps]
    m = 0.6
    bounds = [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]

    steps = steps[:: max(1, len(steps) // 68)]
    imgs = []
    for pose, sp in steps:
        canvas = render_floor_canvas(pose, traj, sp, bounds, ppm=170)   # BGR
        imgs.append(Image.fromarray(canvas[:, :, ::-1]))               # BGR -> RGB
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=55, loop=0)
    print(f"wrote {a.out}  ({len(imgs)} frames, {imgs[0].size}, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
