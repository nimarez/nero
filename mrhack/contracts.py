"""Shared contracts - single source of truth. stdlib only.
Units: meters, radians, seconds, m/s, rad/s. Floor frame from the origin ArUco marker, yaw CCW.
Body frame: vx forward, vy left, wz CCW. Every message stamps t (epoch s)."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1


@dataclass
class RobotPose:
    x: float; y: float; yaw: float; t: float


@dataclass
class Goal:
    x: float; y: float; label: str; t: float


@dataclass
class TrajPoint:
    x: float; y: float; heading: float


@dataclass
class Trajectory:
    points: list; traj_id: int; t: float


@dataclass
class Setpoint:
    x: float; y: float; radius: float; s: float; done: bool; t: float


@dataclass
class VelCmd:
    vx: float; vy: float; wz: float; t: float


@dataclass
class RobotState:
    mode: str; vx: float; vy: float; wz: float; t: float


@dataclass
class Estop:
    reason: str; t: float


@dataclass
class CalibConfig:
    H_cam2floor: list; H_proj2floor: list; floor_bounds: list
    origin_marker_id: int; marker_size_m: float; camera_index: int
    camera_resolution: list; reproj_error_px: float; calib_time: float

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def sha(self):
        payload = json.dumps([self.H_cam2floor, self.H_proj2floor], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]


TOPICS = {"pose": RobotPose, "goal": Goal, "traj": Trajectory, "setpoint": Setpoint,
          "vel_cmd": VelCmd, "robot_state": RobotState, "estop": Estop}
_TYPE_TOPIC = {v: k for k, v in TOPICS.items()}


def to_wire(msg):
    topic = _TYPE_TOPIC[type(msg)]
    return topic.encode("utf-8"), json.dumps(asdict(msg)).encode("utf-8")


def from_wire(topic, payload):
    t = topic.decode("utf-8") if isinstance(topic, (bytes, bytearray)) else topic
    cls = TOPICS[t]
    d = json.loads(payload)
    if cls is Trajectory:
        d["points"] = [TrajPoint(**p) for p in d["points"]]
    return cls(**d)
