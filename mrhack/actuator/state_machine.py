"""K1Gate - the ONLY code that drives the robot: a thin SAFETY wrapper around the repo's Booster
adapter (nero.robot.RobotAdapter / RobotInterface).

It does NOT touch the SDK directly and does NOT change robot mode: RobotInterface owns the Booster
B1LocoClient and REQUIRES the K1 to already be in walking mode (2) - it refuses to change mode. This
gate adds the safety layer on top of `set_velocity`: clamp to Limits, deadband, a deadman watchdog
that zeroes on comms loss, and a GUARANTEED stop on any exit (atexit / signals / context manager).

    from mrhack.actuator.state_machine import K1Gate, make_robot
    with K1Gate(make_robot()) as gate:      # real K1 (needs the robot's ROS 2 env + booster SDK)
        gate.command(vel)                    # clamp -> RobotInterface.set_velocity -> B1LocoClient.Move

Test with any object exposing initialize/set_velocity/stop (no hardware, no nero package):
    uv run python -m mrhack.actuator.state_machine --selftest
"""
from __future__ import annotations
import atexit
import logging
import signal
import threading
import time
from typing import Protocol

from .. import config
from ..contracts import VelCmd

log = logging.getLogger("K1Gate")


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _snap(v, db):
    return 0.0 if abs(v) < db else v


class RobotAdapter(Protocol):
    """Structural match for nero.robot.RobotAdapter - the repo's real Booster interface."""

    def initialize(self) -> None: ...
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None: ...
    def stop(self) -> None: ...


class K1Gate:
    def __init__(self, robot, limits=None, deadman_s=config.DEADMAN_S, clock=time.monotonic):
        self.robot = robot
        self.limits = limits or config.LIMITS
        self.deadman_s = deadman_s
        self._clock = clock
        self._last_cmd_t = None
        self._started = False
        self._zeroed = False
        self._shutdown_done = False
        self._guards = False
        self._watch = None
        self._stop_watch = threading.Event()

    # ---- lifecycle ----
    def start(self, watchdog=True):
        """Preflight through the adapter (RobotInterface verifies sensors + walk mode, arms at zero).
        No mode change here - the K1 must ALREADY be in walking mode (2)."""
        self.robot.initialize()
        self._started = True
        self._last_cmd_t = self._clock()
        self.install_guards()
        if watchdog:
            self._watch = threading.Thread(target=self._watchdog, name="k1-deadman", daemon=True)
            self._watch.start()
        return self

    def command(self, cmd):
        """Clamp + deadband a VelCmd and stream it to the Booster adapter."""
        if not self._started:
            raise RuntimeError("call start() before command()")
        vx = _snap(_clamp(cmd.vx, -self.limits.vx_max, self.limits.vx_max), config.V_DEADBAND)
        vy = _snap(_clamp(cmd.vy, -self.limits.vy_max, self.limits.vy_max), config.V_DEADBAND)
        wz = _snap(_clamp(cmd.wz, -self.limits.wz_max, self.limits.wz_max), config.WZ_DEADBAND)
        self.robot.set_velocity(vx, vy, wz)
        self._last_cmd_t = self._clock()
        self._zeroed = vx == 0.0 and vy == 0.0 and wz == 0.0
        return vx, vy, wz

    def zero(self):
        if self._started:
            self.robot.set_velocity(0.0, 0.0, 0.0)
            self._zeroed = True

    # ---- deadman ----
    def _deadman_expired(self):
        return (
            self._started
            and self._last_cmd_t is not None
            and self._clock() - self._last_cmd_t > self.deadman_s
        )

    def _watchdog_step(self):
        """One deadman check: zero exactly once if commands go stale (comms loss / dead loop)."""
        if self._deadman_expired() and not self._zeroed:
            log.warning("deadman: no command in %.2fs -> zero", self.deadman_s)
            self.zero()

    def _watchdog(self):
        period = 1.0 / config.ACTUATOR_HZ
        while not self._stop_watch.wait(period):
            try:
                self._watchdog_step()
            except Exception:
                log.exception("watchdog step failed")

    # ---- shutdown (guaranteed) ----
    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop_watch.set()
        try:
            self.robot.stop()
            log.info("shutdown: robot stopped")
        except Exception:
            log.exception("shutdown FAILED - robot may still be moving!")
            try:
                self.robot.set_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass

    def install_guards(self):
        if self._guards:
            return
        self._guards = True
        atexit.register(self.shutdown)
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass  # not in the main thread

    def _on_signal(self, signum, frame):
        log.warning("signal %s -> shutdown", signum)
        self.shutdown()
        raise SystemExit(0)

    def __enter__(self):
        if not self._started:
            self.start()
        return self

    def __exit__(self, et, e, tb):
        self.shutdown()
        return False


def make_robot(**kwargs):
    """Construct the real repo Booster adapter (nero.robot.RobotInterface). Lazy import: it needs the
    robot's ROS 2 env + booster-robotics-sdk-python, so it only resolves on the K1 / robot box."""
    from nero.robot import RobotInterface

    return RobotInterface(**kwargs)


class _MockAdapter:
    """Minimal RobotAdapter for the self-test - records set_velocity calls, no hardware."""

    def __init__(self):
        self.calls, self.initialized, self.stopped = [], False, False

    def initialize(self):
        self.initialized = True

    def set_velocity(self, vx, vy, vyaw):
        self.calls.append((round(vx, 4), round(vy, 4), round(vyaw, 4)))

    def stop(self):
        self.stopped = True


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    t = [0.0]
    m = _MockAdapter()
    g = K1Gate(m, limits=config.Limits(vx_max=0.3, vy_max=0.0, wz_max=0.6),
               deadman_s=0.3, clock=lambda: t[0])
    g.start(watchdog=False)                          # drive _watchdog_step directly (no thread in test)
    check("start() initializes the adapter (no mode change)", m.initialized)

    g.command(VelCmd(0.9, 0.5, 0.4, 0.0))            # over-limit vx, lateral vy, in-range wz
    check("clamp vx->0.3, vy->0 (K1 no lateral), wz kept", m.calls[-1] == (0.3, 0.0, 0.4))

    g.command(VelCmd(0.2, 0.0, 0.01, 0.0))           # tiny wz below deadband
    check("deadband zeroes tiny wz", m.calls[-1] == (0.2, 0.0, 0.0))

    t[0] = 0.1
    g._watchdog_step()
    check("no premature deadman (0.1s < 0.3s)", m.calls[-1] == (0.2, 0.0, 0.0))
    t[0] = 0.5
    g._watchdog_step()
    check("deadman zeroes on comms loss (0.5s > 0.3s)", m.calls[-1] == (0.0, 0.0, 0.0))
    n = len(m.calls)
    g._watchdog_step()
    check("deadman fires once, not spammed", len(m.calls) == n)

    g.shutdown()
    check("shutdown stops the robot", m.stopped)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(0 if _selftest() else 1)
    print("K1Gate(make_robot()) drives the real K1 via nero.robot.RobotInterface; run --selftest for the mock.")


if __name__ == "__main__":
    main()
