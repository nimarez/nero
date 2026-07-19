"""Lightweight object navigation using only camera-frame pure pursuit.

This keeps the human-facing behavior of ``nero-orb-slam`` but intentionally
does not build a map or estimate a global pose.  It detects the requested
object in live RGB-D, curves toward it, and stops at the configured stand-off.

Usage:
    uv run nero-pure-pursuit --no-display
"""

from __future__ import annotations

import argparse
import enum
import logging
import math
import os
import signal
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from nero.interaction import (
    K1VoiceCommandSource,
    NavigationTargetListener,
    TerminalCommandSource,
    UnixSocketCommandSource,
    safe_stand_off_distance,
)
from nero.navigation.controller import VelocityCommand
from nero.navigation.pure_pursuit import PurePursuitConfig, PurePursuitController
from nero.navigation.runtime import SensorFrame, read_sensor_frame, send_velocity
from nero.navigation.safety import SafetyMonitor, SafetyStatus
from nero.observability import RosObservabilityPublisher
from nero.perception.depth_processor import DepthProcessor
from nero.perception.detector_factory import create_object_detector
from nero.perception.object_detector import (
    ObjectDetection,
    ObjectDetector,
    configure_qualcomm_cpu_partition,
)
from nero.robot import K1_HEAD_PITCH_LIMITS, K1_HEAD_YAW_LIMITS, RobotInterface
from nero.utils.visualization import Visualization

logger = logging.getLogger(__name__)


class PursuitState(enum.Enum):
    IDLE = "idle"
    WAITING_FOR_OBJECT = "waiting_for_object"
    DETECTING = "detecting"
    EXPLORING = "exploring"
    RELOCATING = "relocating"
    ALIGNING = "aligning"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    LOST = "lost"
    ERROR = "error"


DEFAULT_HEAD_SCAN_POSES = ((0.0, 0.0),)
RELOCATION_MANEUVER_PATTERN = (
    "advance",
    "sidestep_left",
    "turn_around_then_advance",
    "sidestep_right",
    "advance",
)
_REVISION_UNSET = object()


@dataclass(frozen=True)
class HeadScanConfig:
    """Fixed-forward camera observation timing; active head motion is disabled."""

    poses: tuple[tuple[float, float], ...] = DEFAULT_HEAD_SCAN_POSES
    move_duration: float = 0.35
    settle_time: float = 0.15

    def __post_init__(self) -> None:
        if not self.poses:
            raise ValueError("head scan must contain at least one pose")
        if not math.isfinite(self.move_duration) or self.move_duration <= 0:
            raise ValueError("head scan move_duration must be positive and finite")
        if not math.isfinite(self.settle_time) or self.settle_time < 0:
            raise ValueError("head scan settle_time must be non-negative and finite")
        for pitch, yaw in self.poses:
            if not math.isfinite(pitch) or not math.isfinite(yaw):
                raise ValueError("head scan poses must be finite")
            if not K1_HEAD_PITCH_LIMITS[0] <= pitch <= K1_HEAD_PITCH_LIMITS[1]:
                raise ValueError("head scan pitch must stay within K1 limits")
            if not K1_HEAD_YAW_LIMITS[0] <= yaw <= K1_HEAD_YAW_LIMITS[1]:
                raise ValueError("head scan yaw must stay within K1 limits")
            if pitch != 0.0 or yaw != 0.0:
                raise ValueError(
                    "active K1 head motion is disabled; only neutral pose is supported"
                )


@dataclass(frozen=True)
class RelocationConfig:
    """Bounded base motion between fixed-forward camera observations."""

    distance: float = 0.5
    linear_velocity: float = 0.12
    lateral_velocity: float = 0.10
    angular_velocity: float = 0.35
    turnaround_angle: float = math.pi
    max_relocations: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("relocation distance must be positive and finite")
        if not math.isfinite(self.linear_velocity) or self.linear_velocity <= 0:
            raise ValueError("relocation linear_velocity must be positive and finite")
        if not math.isfinite(self.lateral_velocity) or self.lateral_velocity <= 0:
            raise ValueError("relocation lateral_velocity must be positive and finite")
        if not math.isfinite(self.angular_velocity) or self.angular_velocity <= 0:
            raise ValueError("relocation angular_velocity must be positive and finite")
        if not math.isfinite(self.turnaround_angle) or not 0 < self.turnaround_angle <= math.pi:
            raise ValueError("relocation turnaround_angle must be in (0, pi]")
        if self.max_relocations < 0:
            raise ValueError("max_relocations must be non-negative")


@dataclass
class PursuitStatus:
    state: PursuitState
    message: str
    velocity_command: VelocityCommand = field(default_factory=VelocityCommand)
    detections: list[ObjectDetection] = field(default_factory=list)
    safety_status: SafetyStatus | None = None
    safety_enforced: bool = True
    target: str | None = None
    stand_off_distance: float | None = None
    stand_off_tolerance: float = 0.0
    target_position_camera: list[float] | None = None
    obstacle_info: dict | None = None
    detector_metrics: dict | None = None
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    exploration_step: int | None = None
    exploration_steps: int = 0
    relocation_count: int = 0
    relocation_limit: int = 0
    relocation_phase: str | None = None
    relocation_maneuver: str | None = None
    relocation_progress: float = 0.0
    relocation_distance: float = 0.0


class DirectPursuitPolicy:
    """Minimal RGB-D object follower with no localization dependency."""

    def __init__(
        self,
        robot,
        *,
        object_detector=None,
        controller=None,
        depth_processor=None,
        safety=None,
        head_scan: HeadScanConfig | None = None,
        relocation: RelocationConfig | None = None,
        target_timeout: float = 3.0,
        acquisition_timeout: float = 60.0,
        stand_off_distance: float | None = None,
        safety_enforced: bool = True,
    ) -> None:
        if target_timeout <= 0:
            raise ValueError("target_timeout must be positive")
        if acquisition_timeout <= 0:
            raise ValueError("acquisition_timeout must be positive")
        if stand_off_distance is not None and (
            not math.isfinite(stand_off_distance) or stand_off_distance <= 0
        ):
            raise ValueError("stand_off_distance must be positive and finite")
        self.robot = robot
        self.detector = object_detector or ObjectDetector()
        self.controller = controller or PurePursuitController()
        self.depth = depth_processor or DepthProcessor()
        self.safety = safety or SafetyMonitor()
        self.safety_enforced = bool(safety_enforced)
        self.head_scan = head_scan or HeadScanConfig()
        self.relocation = relocation or RelocationConfig()
        self.target_timeout = target_timeout
        self.acquisition_timeout = acquisition_timeout
        self._stand_off_override = stand_off_distance
        self.state = PursuitState.IDLE
        self.target: str | None = None
        self.stand_off = 0.8
        self.last_sensor: SensorFrame | None = None
        self._last_seen: float | None = None
        self._exploration_started: float | None = None
        self._last_detection_revision: int | None = None
        self._confirmation_started: float | None = None
        self._confirmation_revision: int | None = None
        self._confirmation_requires_alignment = False
        self._scan_index = 0
        self._scan_pose_ready_at: float | None = None
        self._scan_settle_revision = _REVISION_UNSET
        self._head_pose = (0.0, 0.0)
        self._alignment_yaw = 0.0
        self._relocation_count = 0
        self._relocation_phase: str | None = None
        self._relocation_plan: str | None = None
        self._relocation_maneuver: str | None = None
        self._relocation_origin: np.ndarray | None = None
        self._relocation_target_yaw = 0.0
        self._relocation_turn_complete = False
        self._relocation_progress = 0.0
        self._running = False
        self._last_obstacle_info: dict | None = None

    def start(self) -> PursuitStatus:
        if not self.safety_enforced:
            logger.warning("SAFETY ENFORCEMENT IS DISABLED; hazard checks are diagnostic only")
        logger.warning(
            "Active K1 head motion is disabled; verify the camera is physically neutral-forward"
        )
        try:
            self.robot.initialize()
            if not self.detector.initialize():
                raise RuntimeError("No live object detector is available")
            self.safety.reset()
        except Exception:
            try:
                self.detector.close()
            except Exception:
                logger.exception("Detector cleanup failed during startup")
            try:
                close = getattr(self.robot, "close", self.robot.stop)
                close()
            except Exception:
                logger.exception("Robot cleanup failed during startup")
            raise
        self._running = True
        self.state = PursuitState.WAITING_FOR_OBJECT
        return self._status("Ready to receive object name")

    def supports_target(self, name: str) -> bool:
        return self.detector.supports_target(name)

    def set_target(self, name: str) -> PursuitStatus:
        resolved = self.detector.resolve_target(name)
        if resolved is None:
            return self._status(f"Object class '{name}' is not supported")
        self.detector.set_target(resolved)
        self.target = resolved
        self.stand_off = (
            self._stand_off_override
            if self._stand_off_override is not None
            else safe_stand_off_distance(resolved)
        )
        self._last_seen = None
        self._begin_exploration(time.monotonic(), reset_scan=True, restart_search=True)
        self._last_detection_revision = None
        self._confirmation_started = None
        self._confirmation_revision = None
        self._confirmation_requires_alignment = False
        stop_failure = self._stop_required([], None)
        if stop_failure is not None:
            return stop_failure
        return self._status(f"Exploring for '{resolved}' with the fixed forward camera")

    def step(self) -> PursuitStatus:
        if not self._running:
            return self._status("Policy not running")
        if self.state == PursuitState.ERROR:
            return self._status("Policy stopped after a command or sensor error")
        try:
            sensor = read_sensor_frame(self.robot)
            self.last_sensor = sensor
            depth_m = self.depth.preprocess(sensor.depth)
            obstacles = self.depth.detect_obstacles(depth_m)
            self._last_obstacle_info = obstacles
            safety = self.safety.check_safety(
                imu_rpy=sensor.imu_rpy,
                obstacle_distance=float(obstacles["min_distance"]),
                battery_level=getattr(sensor.raw_state, "battery_level", None),
                depth_sensor_blind=bool(obstacles.get("sensor_blind", False)),
            )
        except Exception as exc:
            logger.exception("Direct pursuit sensor failure")
            self.state = PursuitState.ERROR
            self._stop_robot()
            return self._status(f"Sensor failure: {exc}")

        if self.target is None:
            self.state = PursuitState.WAITING_FOR_OBJECT
            self._stop_robot()
            return self._status("Waiting for object name", safety_status=safety)

        detections = self.detector.detect(sensor.rgb, sensor.depth, sensor.camera_info)
        revision = getattr(self.detector, "result_revision", None)
        new_detection_result = revision is None or revision != self._last_detection_revision
        if new_detection_result:
            self._last_detection_revision = revision
        target = self.detector.find_object(detections, self.target)
        now = time.monotonic()

        if self.state == PursuitState.RELOCATING:
            if new_detection_result and target is not None:
                return self._begin_confirmation(detections, safety, revision, now)
            return self._relocate(sensor, detections, safety, obstacles, now)

        if self.state == PursuitState.EXPLORING:
            observation_ready = self._scan_observation_ready(now, revision)
            if observation_ready and new_detection_result and target is not None:
                return self._begin_confirmation(detections, safety, revision, now)
            return self._explore(
                detections,
                safety,
                revision,
                now,
                observation_ready=observation_ready,
            )

        if self.state == PursuitState.ALIGNING:
            return self._align_to_discovery(sensor, detections, safety, revision, now)

        if self._confirmation_started is not None:
            return self._confirm_detection(
                target,
                detections,
                safety,
                obstacles,
                revision,
                now,
                new_detection_result,
            )

        if target is None or target.position_3d is None:
            self._begin_exploration(now, reset_scan=True, restart_search=True)
            return self._explore(detections, safety, revision, now)

        if new_detection_result:
            self._last_seen = now
        elif self._last_seen is None or now - self._last_seen > self.target_timeout:
            self._begin_exploration(now, reset_scan=True, restart_search=True)
            return self._explore(detections, safety, revision, now)
        return self._pursue(target, detections, safety, obstacles)

    def _pursue(self, target, detections, safety, obstacles) -> PursuitStatus:
        if self.safety_enforced and not safety.is_safe:
            stop_failure = self._stop_required(detections, safety)
            if stop_failure is not None:
                return stop_failure
            return self._status(
                f"Motion blocked: {safety.reason}",
                detections=detections,
                safety_status=safety,
                target_position=target.position_3d,
            )
        self.state = PursuitState.NAVIGATING
        try:
            arrived = self.controller.has_arrived(target.position_3d, self.stand_off)
            command = (
                VelocityCommand()
                if arrived
                else self.controller.compute_command(target.position_3d, self.stand_off)
            )
        except ValueError as exc:
            self._stop_robot()
            return self._status(
                f"Invalid target depth: {exc}",
                detections=detections,
                safety_status=safety,
                target_position=target.position_3d,
            )

        if arrived:
            self.state = PursuitState.ARRIVED
            self._stop_robot()
            return self._status(
                f"Holding stand-off from '{self.target}'",
                detections=detections,
                safety_status=safety,
                target_position=target.position_3d,
            )

        # Never translate into a blocked center corridor. Turning remains
        # allowed so the live target can be reacquired around an obstacle.
        if self.safety_enforced and not obstacles.get("center_clear", False):
            command = VelocityCommand(angular_z=command.angular_z)
        try:
            send_velocity(self.robot, command)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return self._status(
            f"Pursuing '{self.target}'",
            command=command,
            detections=detections,
            safety_status=safety,
            target_position=target.position_3d,
        )

    def _begin_exploration(
        self,
        now: float,
        *,
        reset_scan: bool,
        restart_search: bool,
    ) -> None:
        self.state = PursuitState.EXPLORING
        self._confirmation_started = None
        self._confirmation_revision = None
        self._confirmation_requires_alignment = False
        if restart_search or self._exploration_started is None:
            self._exploration_started = now
        if restart_search:
            self._relocation_count = 0
            self._clear_relocation()
        if reset_scan:
            self._scan_index = 0
            self._scan_pose_ready_at = None
            self._scan_settle_revision = _REVISION_UNSET

    def _scan_observation_ready(self, now: float, revision: int | None) -> bool:
        if self._scan_pose_ready_at is None or now < self._scan_pose_ready_at:
            return False
        if revision is None:
            return True
        if self._scan_settle_revision is _REVISION_UNSET:
            self._scan_settle_revision = revision
            return False
        return revision != self._scan_settle_revision

    def _explore(
        self,
        detections,
        safety,
        revision,
        now: float,
        *,
        observation_ready: bool = False,
    ) -> PursuitStatus:
        self.state = PursuitState.EXPLORING
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        started = now if self._exploration_started is None else self._exploration_started
        self._exploration_started = started
        if now - started > self.acquisition_timeout:
            return self._target_lost(detections, safety)
        if observation_ready:
            if self._scan_index + 1 >= len(self.head_scan.poses):
                return self._begin_relocation(detections, safety, now)
            self._scan_index += 1
            self._scan_pose_ready_at = None
            self._scan_settle_revision = _REVISION_UNSET
        if self._scan_pose_ready_at is None:
            self._head_pose = (0.0, 0.0)
            self._scan_pose_ready_at = now + self.head_scan.settle_time
            self._scan_settle_revision = _REVISION_UNSET
        pitch, yaw = self._head_pose
        return self._status(
            f"Exploring for '{self.target}' with fixed forward camera "
            f"{self._scan_index + 1}/{len(self.head_scan.poses)} "
            f"(pitch={pitch:+.2f}, yaw={yaw:+.2f})",
            detections=detections,
            safety_status=safety,
        )

    def _begin_relocation(self, detections, safety, now: float) -> PursuitStatus:
        if self._relocation_count >= self.relocation.max_relocations:
            return self._target_lost(detections, safety)
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        self.state = PursuitState.RELOCATING
        self._head_pose = (0.0, 0.0)
        self._scan_pose_ready_at = None
        self._relocation_phase = "selecting_path"
        self._relocation_plan = RELOCATION_MANEUVER_PATTERN[
            self._relocation_count % len(RELOCATION_MANEUVER_PATTERN)
        ]
        self._relocation_maneuver = self._relocation_plan
        self._relocation_turn_complete = False
        self._relocation_origin = np.asarray(self.last_sensor.odometry[:2], dtype=float).copy()
        self._relocation_progress = 0.0
        return self._status(
            "Forward observation complete; selecting a relocation path",
            detections=detections,
            safety_status=safety,
        )

    def _relocate(self, sensor, detections, safety, obstacles, now: float) -> PursuitStatus:
        """Move to a new observation point without entering pursuit."""
        self.state = PursuitState.RELOCATING
        started = now if self._exploration_started is None else self._exploration_started
        self._exploration_started = started
        if now - started > self.acquisition_timeout:
            return self._target_lost(detections, safety)

        if self._relocation_origin is None:
            self._relocation_origin = np.asarray(sensor.odometry[:2], dtype=float).copy()
        displacement = np.asarray(sensor.odometry[:2], dtype=float) - self._relocation_origin
        self._relocation_progress = float(np.linalg.norm(displacement))
        if self._relocation_progress >= self.relocation.distance:
            return self._complete_relocation(detections, safety, now)

        if self.safety_enforced and not safety.is_safe:
            stop_failure = self._stop_required(detections, safety)
            if stop_failure is not None:
                return stop_failure
            return self._status(
                f"Relocation blocked: {safety.reason}",
                detections=detections,
                safety_status=safety,
            )

        if self._relocation_phase == "turning":
            current_yaw = float(sensor.odometry[2])
            yaw_error = self._normalize_angle(self._relocation_target_yaw - current_yaw)
            if abs(yaw_error) <= self.controller.config.bearing_tolerance:
                stop_failure = self._stop_required(detections, safety)
                if stop_failure is not None:
                    return stop_failure
                self._relocation_phase = "selecting_path"
                self._relocation_turn_complete = True
                self._relocation_maneuver = "advance_after_turn"
                return self._status(
                    "Relocation turn complete; checking the forward path",
                    detections=detections,
                    safety_status=safety,
                )
            command = VelocityCommand(
                angular_z=math.copysign(self.relocation.angular_velocity, yaw_error)
            )
            return self._send_relocation_command(
                command,
                "Turning around to explore a new heading",
                detections,
                safety,
            )

        sensor_blind = bool(obstacles.get("sensor_blind", False))
        maneuver_clear = self._maneuver_path_clear(obstacles)
        if self._relocation_phase == "advancing" and not maneuver_clear and self.safety_enforced:
            stop_failure = self._stop_required(detections, safety)
            if stop_failure is not None:
                return stop_failure
            self._relocation_phase = "selecting_path"

        if self._relocation_phase == "selecting_path":
            if (
                self._relocation_plan == "turn_around_then_advance"
                and not self._relocation_turn_complete
            ):
                self._relocation_target_yaw = self._normalize_angle(
                    float(sensor.odometry[2]) + self.relocation.turnaround_angle
                )
                self._relocation_phase = "turning"
                yaw_error = self._normalize_angle(
                    self._relocation_target_yaw - float(sensor.odometry[2])
                )
                command = VelocityCommand(
                    angular_z=math.copysign(self.relocation.angular_velocity, yaw_error)
                )
                return self._send_relocation_command(
                    command,
                    "Turning around to explore a new heading",
                    detections,
                    safety,
                )

            selected = self._select_translation_maneuver(obstacles)
            if selected is None and self.safety_enforced:
                stop_failure = self._stop_required(detections, safety)
                if stop_failure is not None:
                    return stop_failure
                reason = "depth is unavailable" if sensor_blind else "no clear sector"
                return self._status(
                    f"Relocation blocked: {reason}",
                    detections=detections,
                    safety_status=safety,
                )
            if selected is not None:
                self._relocation_maneuver = selected
            self._relocation_phase = "advancing"

        command = self._relocation_velocity_command()
        maneuver_name = self._relocation_maneuver.replace("_", " ")
        return self._send_relocation_command(
            command,
            f"Relocating via {maneuver_name} to observation point {self._relocation_count + 2}",
            detections,
            safety,
        )

    def _select_translation_maneuver(self, obstacles: dict) -> str | None:
        if not self.safety_enforced:
            if self._relocation_plan == "turn_around_then_advance":
                return "advance_after_turn"
            return self._relocation_plan

        preferences = {
            "advance": ("advance", "sidestep_left", "sidestep_right"),
            "sidestep_left": ("sidestep_left", "sidestep_right", "advance"),
            "sidestep_right": ("sidestep_right", "sidestep_left", "advance"),
            "turn_around_then_advance": (
                "advance_after_turn",
                "sidestep_left",
                "sidestep_right",
            ),
        }
        for maneuver in preferences[self._relocation_plan]:
            if self._maneuver_path_clear(obstacles, maneuver=maneuver):
                return maneuver
        return None

    def _maneuver_path_clear(self, obstacles: dict, *, maneuver: str | None = None) -> bool:
        if bool(obstacles.get("sensor_blind", False)):
            return False
        maneuver = maneuver or self._relocation_maneuver
        if maneuver == "sidestep_left":
            return bool(obstacles.get("left_clear", False))
        if maneuver == "sidestep_right":
            return bool(obstacles.get("right_clear", False))
        return bool(obstacles.get("center_clear", False))

    def _relocation_velocity_command(self) -> VelocityCommand:
        if self._relocation_maneuver == "sidestep_left":
            return VelocityCommand(linear_y=self.relocation.lateral_velocity)
        if self._relocation_maneuver == "sidestep_right":
            return VelocityCommand(linear_y=-self.relocation.lateral_velocity)
        return VelocityCommand(linear_x=self.relocation.linear_velocity)

    def _send_relocation_command(
        self,
        command: VelocityCommand,
        message: str,
        detections,
        safety,
    ) -> PursuitStatus:
        try:
            send_velocity(self.robot, command)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return self._status(
            message,
            command=command,
            detections=detections,
            safety_status=safety,
        )

    def _complete_relocation(self, detections, safety, now: float) -> PursuitStatus:
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        self._relocation_count += 1
        self._clear_relocation()
        self._begin_exploration(now, reset_scan=True, restart_search=False)
        return self._status(
            f"Reached observation point {self._relocation_count + 1}; starting a new scan",
            detections=detections,
            safety_status=safety,
        )

    def _clear_relocation(self) -> None:
        self._relocation_phase = None
        self._relocation_plan = None
        self._relocation_maneuver = None
        self._relocation_origin = None
        self._relocation_target_yaw = 0.0
        self._relocation_turn_complete = False
        self._relocation_progress = 0.0

    def _begin_confirmation(
        self,
        detections,
        safety,
        revision: int | None,
        now: float,
    ) -> PursuitStatus:
        self.state = PursuitState.DETECTING
        self._confirmation_started = now
        self._confirmation_revision = revision
        self._confirmation_requires_alignment = True
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        return self._status(
            f"Confirming '{self.target}' in the fixed forward camera",
            detections=detections,
            safety_status=safety,
        )

    def _confirm_detection(
        self,
        target,
        detections,
        safety,
        obstacles,
        revision: int | None,
        now: float,
        new_detection_result: bool,
    ) -> PursuitStatus:
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        if now - self._confirmation_started > self.target_timeout:
            if self._resume_exploration(now, advance_scan=True):
                return self._begin_relocation(detections, safety, now)
            return self._explore(detections, safety, revision, now)
        fresh = revision is None or (
            new_detection_result and revision != self._confirmation_revision
        )
        if not fresh:
            return self._status(
                f"Waiting for a fresh '{self.target}' detection",
                detections=detections,
                safety_status=safety,
            )
        if target is None:
            if self._resume_exploration(now, advance_scan=True):
                return self._begin_relocation(detections, safety, now)
            return self._explore(detections, safety, revision, now)
        if target.position_3d is None:
            return self._status(
                f"Detected '{self.target}' in 2D; waiting for valid depth",
                detections=detections,
                safety_status=safety,
            )
        self._last_seen = now
        self._confirmation_started = None
        self._confirmation_revision = None
        requires_alignment = self._confirmation_requires_alignment
        self._confirmation_requires_alignment = False
        if requires_alignment:
            return self._begin_alignment(target, detections, safety, now)
        self._exploration_started = None
        return self._pursue(target, detections, safety, obstacles)

    def _begin_alignment(self, target, detections, safety, now: float) -> PursuitStatus:
        stop_failure = self._stop_required(detections, safety)
        if stop_failure is not None:
            return stop_failure
        try:
            target_base = self._target_at_base_heading(target.position_3d)
            lateral = -float(target_base[0])
            forward = float(target_base[2])
            bearing = math.atan2(lateral, forward)
            current_yaw = float(self.last_sensor.odometry[2])
            self._alignment_yaw = self._normalize_angle(current_yaw + bearing)
        except (RuntimeError, ValueError, IndexError) as exc:
            self.state = PursuitState.ERROR
            self._stop_robot()
            return self._status(
                f"Body alignment setup failed: {exc}",
                detections=detections,
                safety_status=safety,
            )
        self._head_pose = (0.0, 0.0)
        self._scan_pose_ready_at = None
        self.state = PursuitState.ALIGNING
        self._stop_robot()
        return self._status(
            f"Aligning the body before pursuing '{self.target}'",
            detections=detections,
            safety_status=safety,
            target_position=target.position_3d,
        )

    def _align_to_discovery(
        self,
        sensor,
        detections,
        safety,
        revision: int | None,
        now: float,
    ) -> PursuitStatus:
        if (
            self._exploration_started is not None
            and now - self._exploration_started > self.acquisition_timeout
        ):
            return self._target_lost(detections, safety)
        if self.safety_enforced and not safety.is_safe:
            stop_failure = self._stop_required(detections, safety)
            if stop_failure is not None:
                return stop_failure
            return self._status(
                f"Alignment blocked: {safety.reason}",
                detections=detections,
                safety_status=safety,
            )
        current_yaw = float(sensor.odometry[2])
        yaw_error = self._normalize_angle(self._alignment_yaw - current_yaw)
        if abs(yaw_error) <= self.controller.config.bearing_tolerance:
            self.state = PursuitState.DETECTING
            self._confirmation_started = now
            self._confirmation_revision = revision
            self._confirmation_requires_alignment = False
            stop_failure = self._stop_required(detections, safety)
            if stop_failure is not None:
                return stop_failure
            return self._status(
                f"Confirming forward view of '{self.target}'",
                detections=detections,
                safety_status=safety,
            )
        angular = float(
            np.clip(
                2.0 * yaw_error,
                -self.controller.config.max_angular_velocity,
                self.controller.config.max_angular_velocity,
            )
        )
        command = VelocityCommand(angular_z=angular)
        try:
            send_velocity(self.robot, command)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return self._status(
            f"Aligning the body with '{self.target}'",
            command=command,
            detections=detections,
            safety_status=safety,
        )

    def _resume_exploration(self, now: float, *, advance_scan: bool) -> bool:
        self.state = PursuitState.EXPLORING
        self._confirmation_started = None
        self._confirmation_revision = None
        self._confirmation_requires_alignment = False
        if advance_scan:
            if self._scan_index + 1 >= len(self.head_scan.poses):
                return True
            self._scan_index += 1
        self._scan_pose_ready_at = None
        self._scan_settle_revision = _REVISION_UNSET
        if self._exploration_started is None:
            self._exploration_started = now
        return False

    def _target_at_base_heading(self, target_camera) -> np.ndarray:
        target = np.asarray(target_camera, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_camera must be a finite [x, y, z] vector")
        pitch, yaw = self._head_pose
        x_camera, y_camera, z_camera = map(float, target)
        forward_at_yaw = z_camera * math.cos(pitch) - y_camera * math.sin(pitch)
        x_base_camera = x_camera * math.cos(yaw) - forward_at_yaw * math.sin(yaw)
        forward_base = forward_at_yaw * math.cos(yaw) + x_camera * math.sin(yaw)
        if forward_base <= 0:
            raise ValueError("discovered target is behind the robot")
        return np.array([x_base_camera, 0.0, forward_base], dtype=float)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _target_lost(self, detections, safety) -> PursuitStatus:
        self.state = PursuitState.LOST
        self._stop_robot()
        self._center_head_best_effort()
        return self._status(
            f"Could not find '{self.target}'",
            detections=detections,
            safety_status=safety,
        )

    def reset(self) -> PursuitStatus:
        self._stop_robot()
        self._center_head_best_effort()
        self.target = None
        self._last_seen = None
        self._exploration_started = None
        self._last_detection_revision = None
        self._confirmation_started = None
        self._confirmation_revision = None
        self._confirmation_requires_alignment = False
        self._relocation_count = 0
        self._clear_relocation()
        self._scan_index = 0
        self._scan_pose_ready_at = None
        self._scan_settle_revision = _REVISION_UNSET
        self.state = PursuitState.WAITING_FOR_OBJECT
        return self._status("Ready for another object command")

    def stop(self) -> None:
        self._running = False
        try:
            self._stop_robot()
            self._center_head_best_effort()
        finally:
            try:
                self.detector.close()
            finally:
                self.state = PursuitState.IDLE

    def _stop_robot(self) -> None:
        """Hold position even if Booster already dropped walking control."""
        try:
            send_velocity(self.robot)
        except RuntimeError:
            logger.exception("Robot locomotion controller rejected the stop command")

    def _stop_required(self, detections, safety) -> PursuitStatus | None:
        """Stop the base or fail closed before any stationary head behavior."""
        try:
            send_velocity(self.robot)
        except RuntimeError as exc:
            return self._locomotion_error(exc, detections, safety)
        return None

    def _locomotion_error(self, exc, detections, safety) -> PursuitStatus:
        self.state = PursuitState.ERROR
        self._stop_robot()
        return self._status(
            f"Locomotion command failed: {exc}",
            detections=detections,
            safety_status=safety,
        )

    def _center_head_best_effort(self) -> None:
        self._head_pose = (0.0, 0.0)

    def _status(
        self,
        message: str,
        *,
        command: VelocityCommand | None = None,
        detections: list[ObjectDetection] | None = None,
        safety_status: SafetyStatus | None = None,
        target_position=None,
    ) -> PursuitStatus:
        telemetry = getattr(self.detector, "telemetry", None)
        return PursuitStatus(
            state=self.state,
            message=message,
            velocity_command=command or VelocityCommand(),
            detections=detections or [],
            safety_status=safety_status,
            safety_enforced=self.safety_enforced,
            target=self.target,
            stand_off_distance=self.stand_off if self.target is not None else None,
            stand_off_tolerance=self.controller.config.position_tolerance,
            target_position_camera=(
                None if target_position is None else [float(value) for value in target_position]
            ),
            obstacle_info=self._last_obstacle_info,
            detector_metrics=telemetry() if callable(telemetry) else None,
            head_pitch=self._head_pose[0],
            head_yaw=self._head_pose[1],
            exploration_step=(
                self._scan_index + 1 if self.state == PursuitState.EXPLORING else None
            ),
            exploration_steps=len(self.head_scan.poses),
            relocation_count=self._relocation_count,
            relocation_limit=self.relocation.max_relocations,
            relocation_phase=self._relocation_phase,
            relocation_maneuver=self._relocation_maneuver,
            relocation_progress=self._relocation_progress,
            relocation_distance=self.relocation.distance,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nero direct RGB-D pure-pursuit object agent")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--command-source",
        choices=("socket", "terminal", "voice"),
        default="socket",
    )
    parser.add_argument("--command-socket", default="/tmp/nero-navigation.sock")
    parser.add_argument(
        "--no-ros-observability",
        action="store_true",
        help="Disable normalized /nero ROS 2 telemetry topics",
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Disable motion safety enforcement while retaining diagnostics (dangerous)",
    )
    parser.add_argument(
        "--object-backend",
        help="Detector backend (default: NERO_OBJECT_BACKEND or K1 QNN)",
    )
    parser.add_argument(
        "--aruco-map",
        help="JSON marker-ID to object-name mapping (or set NERO_ARUCO_MAP)",
    )
    parser.add_argument(
        "--aruco-dictionary",
        help="OpenCV ArUco dictionary (default: DICT_4X4_50)",
    )
    parser.add_argument("--max-velocity", type=float, default=0.25)
    parser.add_argument("--max-angular-velocity", type=float, default=0.7)
    parser.add_argument("--target-timeout", type=float, default=3.0)
    parser.add_argument("--acquisition-timeout", type=float, default=60.0)
    parser.add_argument(
        "--stand-off-distance",
        type=float,
        help="Override the class-aware target stand-off distance in meters",
    )
    parser.add_argument(
        "--observation-settle-time",
        "--head-scan-settle-time",
        dest="observation_settle_time",
        type=float,
        default=0.15,
        help="Seconds to wait before accepting a fresh fixed-camera detection",
    )
    parser.add_argument(
        "--relocation-distance",
        type=float,
        default=0.5,
        help="Meters to walk between fixed-forward camera observations",
    )
    parser.add_argument(
        "--relocation-velocity",
        type=float,
        default=0.12,
        help="Forward velocity in m/s while relocating between scans",
    )
    parser.add_argument(
        "--relocation-lateral-velocity",
        type=float,
        default=0.10,
        help="Side-step velocity in m/s while relocating between scans",
    )
    parser.add_argument(
        "--relocation-angular-velocity",
        "--search-angular-velocity",
        dest="relocation_angular_velocity",
        type=float,
        default=0.35,
        help="Angular velocity in rad/s during exploratory body turns",
    )
    parser.add_argument(
        "--exploration-turn-angle",
        type=float,
        default=180.0,
        help="Body turnaround angle in degrees during the exploration pattern",
    )
    parser.add_argument(
        "--max-relocations",
        type=int,
        default=3,
        help="Maximum number of new observation points before reporting target lost",
    )
    web = parser.add_mutually_exclusive_group()
    web.add_argument(
        "--web-rerun",
        dest="web_rerun",
        action="store_true",
        help="Serve Rerun in a browser (automatic with terminal commands)",
    )
    web.add_argument(
        "--no-web-rerun",
        dest="web_rerun",
        action="store_false",
        help="Do not start the robot-hosted Rerun viewer",
    )
    parser.set_defaults(web_rerun=None)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-path", default="/rerun")
    parser.add_argument("--viewer-port", type=int, default=8081)
    parser.add_argument("--websocket-port", type=int, default=9877)
    parser.add_argument("--server-memory-limit", default="256MB")
    parser.add_argument(
        "--advertise-host",
        default=os.getenv("NERO_ROBOT_HOST", "10.2.1.130"),
        help="Robot hostname or IP printed for the browser URL",
    )
    args = parser.parse_args()
    if args.stand_off_distance is not None and (
        not math.isfinite(args.stand_off_distance) or args.stand_off_distance <= 0
    ):
        parser.error("--stand-off-distance must be positive and finite")
    for name in ("web_port", "viewer_port", "websocket_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if len({args.web_port, args.viewer_port, args.websocket_port}) != 3:
        parser.error("web, viewer, and WebSocket ports must be different")
    if not args.web_path.startswith("/") or "?" in args.web_path or "#" in args.web_path:
        parser.error("--web-path must be an absolute URL path without query or fragment")
    if args.web_path.rstrip("/") in {"", "/"}:
        parser.error("--web-path must name a path such as /rerun")
    if _should_start_web_rerun(args) and args.no_ros_observability:
        parser.error("browser Rerun requires ROS observability; use --no-web-rerun")
    return args


def _should_start_web_rerun(args: argparse.Namespace) -> bool:
    requested = getattr(args, "web_rerun", None)
    return args.command_source == "terminal" if requested is None else bool(requested)


def run_agent(robot, args, *, object_detector=None, command_source=None) -> None:
    controller = PurePursuitController(
        PurePursuitConfig(
            max_linear_velocity=args.max_velocity,
            max_angular_velocity=args.max_angular_velocity,
        )
    )
    policy = DirectPursuitPolicy(
        robot,
        object_detector=object_detector,
        controller=controller,
        target_timeout=args.target_timeout,
        acquisition_timeout=args.acquisition_timeout,
        stand_off_distance=getattr(args, "stand_off_distance", None),
        head_scan=HeadScanConfig(
            settle_time=getattr(args, "observation_settle_time", 0.15),
        ),
        relocation=RelocationConfig(
            distance=getattr(args, "relocation_distance", 0.5),
            linear_velocity=getattr(args, "relocation_velocity", 0.12),
            lateral_velocity=getattr(args, "relocation_lateral_velocity", 0.10),
            angular_velocity=getattr(args, "relocation_angular_velocity", 0.35),
            turnaround_angle=math.radians(getattr(args, "exploration_turn_angle", 180.0)),
            max_relocations=getattr(args, "max_relocations", 3),
        ),
        safety_enforced=not getattr(args, "disable_safety", False),
    )
    shutdown = False
    policy_started = False
    listener = None
    telemetry = None

    def handle_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    try:
        policy.start()
        policy_started = True
        telemetry = RosObservabilityPublisher.try_create(
            enabled=not getattr(args, "no_ros_observability", False)
        )
        signal.signal(signal.SIGTERM, handle_signal)
        commands = command_source or TerminalCommandSource()
        listener = NavigationTargetListener(
            robot,
            commands,
            cancelled=lambda: shutdown,
            target_validator=policy.supports_target,
        )
        listener.start()
        viz = Visualization()
        target_name = None
        announced_arrival = False
        announced_failure = False
        announced_exploration = False
        explored_target = False

        while not shutdown:
            started = time.monotonic()
            if target_name is None:
                target_name = listener.poll()
                if target_name is not None:
                    policy.set_target(target_name)
                    announced_arrival = False
                    announced_failure = False
                    announced_exploration = False
                    explored_target = False

            status = policy.step()
            sensor = policy.last_sensor
            if sensor is not None and telemetry is not None:
                if sensor.raw_state is not None:
                    telemetry.publish_robot_state(sensor.raw_state, robot)
                telemetry.publish_policy(status, sensor.timestamp)
            if status.state == PursuitState.ERROR:
                logger.error("Direct pursuit stopped: %s", status.message)
                break
            if sensor is not None and not args.no_display:
                frame = viz.draw_navigation_info(
                    sensor.rgb,
                    state=status.state.value,
                    message=status.message,
                    fps=20.0,
                    velocity=(
                        status.velocity_command.linear_x,
                        status.velocity_command.angular_z,
                    ),
                )
                key = viz.show_stream(frame, "Nero Pure Pursuit Agent", 20.0)
                if key == ord("q"):
                    shutdown = True
                elif key == ord("r"):
                    policy.reset()
                    target_name = None
                    announced_arrival = False
                    announced_failure = False
                    announced_exploration = False
                    explored_target = False
                    listener.start()

            if (
                status.state == PursuitState.EXPLORING
                and target_name is not None
                and not announced_exploration
            ):
                try:
                    robot.speak(f"Exploring for the {target_name}.")
                except RuntimeError as exc:
                    logger.warning("Could not announce exploration: %s", exc)
                announced_exploration = True
                explored_target = True

            if (
                status.state in {PursuitState.NAVIGATING, PursuitState.ARRIVED}
                and target_name is not None
                and explored_target
            ):
                try:
                    robot.speak(f"Found the {target_name}.")
                except RuntimeError as exc:
                    logger.warning("Could not announce target discovery: %s", exc)
                explored_target = False
                announced_exploration = False

            if status.state == PursuitState.ARRIVED and not announced_arrival:
                try:
                    robot.speak(f"Arrived at {target_name}.")
                except RuntimeError as exc:
                    logger.warning("Could not announce arrival: %s", exc)
                announced_arrival = True
                if args.no_display:
                    policy.reset()
                    target_name = None
                    announced_arrival = False
                    announced_failure = False
                    announced_exploration = False
                    explored_target = False
                    listener.start()
            elif status.state == PursuitState.LOST:
                if not announced_failure and target_name is not None:
                    try:
                        robot.speak(f"I could not detect the {target_name}.")
                    except RuntimeError as exc:
                        logger.warning("Could not announce missing object: %s", exc)
                    announced_failure = True
                policy.reset()
                target_name = None
                announced_arrival = False
                announced_exploration = False
                explored_target = False
                listener.start()

            elapsed = time.monotonic() - started
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    except (KeyboardInterrupt, EOFError, InterruptedError):
        logger.info("Stopping direct pursuit agent")
    finally:
        if policy_started:
            try:
                policy.stop()
            except Exception:
                logger.exception("Direct pursuit policy cleanup failed")
        try:
            close = getattr(robot, "close", robot.stop)
            close()
        except Exception:
            logger.exception("Robot cleanup failed")
        if listener is not None:
            try:
                listener.close()
            except Exception:
                logger.exception("Command listener cleanup failed")
        if telemetry is not None:
            telemetry.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    configure_qualcomm_cpu_partition(args.object_backend)
    try:
        object_detector = create_object_detector(
            backend=args.object_backend,
            aruco_map=args.aruco_map,
            aruco_dictionary=args.aruco_dictionary,
        )
    except ValueError as exc:
        logger.error("Invalid object detector configuration: %s", exc)
        raise SystemExit(2) from exc
    web_bridge = None
    try:
        if _should_start_web_rerun(args):
            from nero.robot_web import start_rerun_web_bridge

            try:
                web_bridge = start_rerun_web_bridge(
                    web_port=args.web_port,
                    web_path=args.web_path,
                    viewer_port=args.viewer_port,
                    websocket_port=args.websocket_port,
                    server_memory_limit=args.server_memory_limit,
                    debug=args.debug,
                    ensure_viz_extra=True,
                )
            except (OSError, RuntimeError) as exc:
                logger.error("Could not start robot-hosted Rerun: %s", exc)
                raise SystemExit(1) from exc
            print(
                f"Rerun: http://{args.advertise_host}:{args.web_port}{args.web_path}",
                flush=True,
            )

        robot = RobotInterface()
        if args.command_source == "socket":
            command_source = UnixSocketCommandSource(args.command_socket)
        elif args.command_source == "terminal":
            command_source = TerminalCommandSource()
        else:
            command_source = K1VoiceCommandSource()
            command_source.start_listening()
            command_source.stop_listening()
        run_agent(
            robot,
            args,
            object_detector=object_detector,
            command_source=command_source,
        )
    finally:
        if web_bridge is not None:
            from nero.robot_web import _stop_process

            _stop_process(web_bridge)


if __name__ == "__main__":
    main()
