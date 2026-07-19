"""MockRobot - duck-typed stand-in for nero.robot.RobotAdapter (initialize / set_velocity / stop).
Records set_velocity calls so all control code runs with no hardware and no nero package.
Mirrors the REAL Booster surface: RobotInterface.set_velocity(vx, vy, vyaw) -> B1LocoClient.Move."""
from __future__ import annotations
import logging
import time

log = logging.getLogger("mock_robot")


class MockRobot:
    """Matches nero.robot.RobotAdapter: initialize / set_velocity / stop. No mode changing."""

    def __init__(self, ip="mock"):
        self.ip = ip
        self.initialized = False
        self.stopped = False
        self.calls: list[tuple[float, float, float, float]] = []   # (t, vx, vy, vyaw)

    def initialize(self) -> None:
        self.initialized = True
        log.debug("[mock] initialize (assumes robot already in walking mode)")

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        self.calls.append((time.time(), vx, vy, vyaw))
        log.debug("[mock] set_velocity %.3f %.3f %.3f", vx, vy, vyaw)

    def stop(self) -> None:
        self.stopped = True
        self.set_velocity(0.0, 0.0, 0.0)

    def velocity_calls(self) -> list[tuple[float, float, float]]:
        return [(vx, vy, vyaw) for (_, vx, vy, vyaw) in self.calls]
