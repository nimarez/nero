"""MockClient - duck-typed stand-in for the robot SDK client. Prints/records instead of moving.
The REAL SDK surface (boosteros set_mode/set_velocity vs booster_robotics_sdk change_mode/walk)
is VERIFY-ON-ROBOT; MockClient is a plausible twin so all control code runs with no hardware."""
from __future__ import annotations
import logging, time

log = logging.getLogger("mock_robot")


class MockClient:
    def __init__(self, ip="mock"):
        self.ip = ip
        self.mode = "DAMP"
        self.calls = []

    def _rec(self, method, *args):
        self.calls.append((time.time(), method, args))

    def change_mode(self, mode):
        self._rec("change_mode", mode)
        self.mode = mode
        log.debug("[mock] mode -> %s", mode)

    def walk(self, vx, vy, wz):
        self._rec("walk", vx, vy, wz)
        log.debug("[mock] walk %.3f %.3f %.3f (%s)", vx, vy, wz, self.mode)

    def get_mode(self):
        return self.mode

    def mode_sequence(self):
        return [a[0] for (_, m, a) in self.calls if m == "change_mode"]

    def walk_calls(self):
        return [a for (_, m, a) in self.calls if m == "walk"]
