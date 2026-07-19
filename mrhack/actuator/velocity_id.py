"""FIRST RUNNABLE - open-loop velocity ID. Drives via the repo Booster interface
(nero.robot.RobotInterface.set_velocity) through K1Gate. De-risks the two scariest unknowns:
commanded->actual velocity + latency. Runs mock (no robot) or real.

The K1 must ALREADY be in walking mode (2) for the real path - K1Gate/RobotInterface never change mode.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import time

from .. import config
from ..contracts import VelCmd
from .state_machine import K1Gate

log = logging.getLogger("velocity_id")


def build_robot(ip, mock):
    if mock:
        from .mock_robot import MockRobot

        return MockRobot(ip)
    from .state_machine import make_robot

    return make_robot()   # real nero.robot.RobotInterface (needs the robot's ROS 2 env + SDK)


def _hold(gate, f, vx, wz, dur, phase):
    dt = 1.0 / config.ACTUATOR_HZ
    t_end = time.time() + dur
    n = 0
    while time.time() < t_end:
        t = time.time()
        gate.command(VelCmd(vx=vx, vy=0.0, wz=wz, t=t))
        f.write(json.dumps({"t": t, "phase": phase, "vx_cmd": vx, "wz_cmd": wz}) + "\n")
        n += 1
        time.sleep(dt)
    return n


def run(ip, matrix, out_path, mock=False, settle_s=None, cmd_s=None):
    settle_s = config.VELID_SETTLE_S if settle_s is None else settle_s
    cmd_s = config.VELID_CMD_S if cmd_s is None else cmd_s
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    robot = build_robot(ip, mock)
    ticks = 0
    with K1Gate(robot) as gate:          # __enter__ -> start() -> robot.initialize(), arms at zero
        with open(out_path, "w") as f:
            for (vx, wz) in matrix:
                log.info("=== test point vx=%.2f wz=%.2f ===", vx, wz)
                _hold(gate, f, 0.0, 0.0, settle_s, "settle")
                ticks += _hold(gate, f, vx, wz, cmd_s, "cmd")
            _hold(gate, f, 0.0, 0.0, settle_s, "settle")
    log.info("velocity-ID done: %d command ticks -> %s", ticks, out_path)
    if hasattr(robot, "velocity_calls"):
        vels = robot.velocity_calls()
        log.info("MOCK SUMMARY: %d set_velocity calls; last = %s", len(vels), vels[-1] if vels else None)
        assert robot.initialized, "robot.initialize() was not called"
        assert robot.stopped, "robot was not stopped on shutdown"
        assert vels and vels[-1] == (0.0, 0.0, 0.0), f"did not zero on shutdown: {vels[-1] if vels else None}"
        assert all(abs(vx) <= config.LIMITS.vx_max + 1e-9 for (vx, _, _) in vels), "vx exceeded clamp"
        log.info("MOCK CHECKS PASSED: initialized, clamped, zeroed + stopped on exit.")
    return out_path


def _parse_matrix(s):
    return [tuple(float(x) for x in pair.split(",")) for pair in s.split(";") if pair.strip()]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--matrix", default="default")
    ap.add_argument("--settle", type=float, default=None)
    ap.add_argument("--cmd", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    matrix = config.VELID_MATRIX if a.matrix == "default" else _parse_matrix(a.matrix)
    out = a.out or os.path.join("mrhack", "runs", "velid_%d.jsonl" % int(time.time()))
    run(a.ip, matrix, out, mock=a.mock, settle_s=a.settle, cmd_s=a.cmd)


if __name__ == "__main__":
    main()
