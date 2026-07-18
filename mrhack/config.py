"""All tunables in one place. velocity-ID overwrites the control numbers on the robot."""
from __future__ import annotations
import os
from dataclasses import dataclass

MRHACK_HOST = os.environ.get("MRHACK_HOST", "127.0.0.1")
PUB_PORT = 5555
SUB_PORT = 5556

CONTROL_HZ = 30.0
ACTUATOR_HZ = 50.0
DEADMAN_S = 0.3
POSE_STALE_S = 0.3
PREP_WAIT_S = 3.0


@dataclass
class Limits:
    vx_max: float = 0.3
    vy_max: float = 0.0
    wz_max: float = 0.6


LIMITS = Limits()

LOOKAHEAD_M = 0.5
V_MAX = 0.3
WZ_MAX = 0.6
V_DEADBAND = 0.05
WZ_DEADBAND = 0.05
HEADING_GAIN = 1.5
SLOW_RADIUS_M = 0.5
GOAL_TOL_M = 0.15
TURN_SLOWDOWN = 1.0

VELID_MATRIX = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.0, 0.3), (0.0, 0.6), (0.2, 0.3)]
VELID_SETTLE_S = 2.0
VELID_CMD_S = 3.0
