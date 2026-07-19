"""Main navigation policy for object-following behavior.

This module implements the agent/policy loop:
1. Read the K1's built-in RGB-D camera stream
2. Accept a human direction and retain only matching live detections
3. Project the track into the SLAM world frame
4. Follow a dynamic stand-off pose while avoiding obstacles
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nero.perception.object_detector import ObjectDetector, ObjectDetection
from nero.navigation.controller import VelocityController, VelocityCommand
from nero.navigation.map_policy import MapNavConfig, MapNavigator, MapRouteResult
from nero.navigation.object_goal import (
    approach_pose,
    blend_world_position,
    body_point_to_world,
    camera_point_to_world,
    planar_detection_to_world,
)
from nero.interaction import safe_stand_off_distance

from nero.perception.depth_processor import DepthProcessor
from nero.slam.orb_slam3_node import ORBSLAM3Node
from nero.slam.pose_estimator import PoseEstimator, FusedPose
from nero.navigation.safety import SafetyMonitor, SafetyStatus
from nero.navigation.runtime import (
    LocalizedFrame,
    SensorFrame,
    initialize_sensor_navigation,
    localize_sensor_frame,
    read_sensor_frame,
    send_velocity,
)

logger = logging.getLogger(__name__)


class PolicyState(enum.Enum):
    """States in the navigation policy loop."""

    IDLE = "idle"
    SHOWING_CAMERA = "showing_camera"
    WAITING_FOR_OBJECT = "waiting_for_object"
    LOCALIZING = "localizing"
    DETECTING = "detecting"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    TRACKING_OBJECT = "tracking_object"
    ARRIVED = "arrived"
    LOST = "lost"
    ERROR = "error"


@dataclass
class NavigationGoal:
    """A navigation goal."""

    object_name: str
    kind: str = "object"
    object_position_world: Optional[np.ndarray] = None
    approach_pose: Optional[np.ndarray] = None  # [x, y, yaw] in world frame
    detection: Optional[ObjectDetection] = None
    stand_off_distance: float = 0.8
    last_observed_monotonic: Optional[float] = None


@dataclass
class PolicyStatus:
    """Current status of the navigation policy."""

    state: PolicyState = PolicyState.IDLE
    current_goal: Optional[NavigationGoal] = None
    current_pose: Optional["FusedPose"] = None
    safety_status: Optional["SafetyStatus"] = None
    velocity_command: Optional[VelocityCommand] = None
    message: str = ""
    detections: list[ObjectDetection] = field(default_factory=list)
    robot_pose: Optional[np.ndarray] = None  # [x, y, yaw] for sim mode
    planned_path: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )  # Nx3 active-frame path for observability
    obstacle_info: dict | None = None


class NavigationPolicy:
    """Main policy loop for object-following navigation.

    Supports both real robot (via RobotInterface) and simulation (via SimEnvironment).

    State machine:
    IDLE -> SHOWING_CAMERA -> WAITING_FOR_OBJECT -> DETECTING -> NAVIGATING -> ARRIVED
    An optional fixed map inserts LOCALIZING and PLANNING without changing the
    sensor, safety, object tracking, command, or actuator loop.

    Transitions:
    - IDLE: start() -> SHOWING_CAMERA
    - SHOWING_CAMERA: (auto) -> WAITING_FOR_OBJECT
    - WAITING_FOR_OBJECT: set_target(name) -> DETECTING
    - DETECTING: object_found -> NAVIGATING, not_found -> WAITING_FOR_OBJECT
    - NAVIGATING: arrived -> ARRIVED, lost -> LOST
    - TRACKING_OBJECT: arrived -> ARRIVED, lost -> LOST
    - Any state: stop() -> IDLE, safety_violation -> ERROR
    """

    def __init__(
        self,
        robot=None,  # RobotInterface (real robot)
        sim_env=None,  # SimEnvironment (simulation)
        slam_config=None,
        slam_options=None,
        object_detector=None,
        navigation_config=None,
        safety_config=None,
        object_track_timeout: float = 1.0,
        object_position_filter: float = 0.35,
        goal_yaw_tolerance: float = 0.15,
        slam_bootstrap_angular_velocity: float = 0.18,
        slam_bootstrap_leg_seconds: float = 2.0,
        slam_bootstrap_timeout: float = 20.0,
        object_search_angular_velocity: float = 0.08,
        map_config: MapNavConfig | None = None,
        enable_object_detection: bool = True,
    ):
        # Environment: real robot or simulation
        self.robot = robot
        self.sim_env = sim_env
        self._is_sim = sim_env is not None

        # Components are required for every sensor-backed robot adapter. The
        # lightweight in-process SimEnvironment has its own synthetic path.
        if not self._is_sim:
            options = dict(slam_options or {})
            self.slam = ORBSLAM3Node(config=slam_config, **options)
            self.pose_estimator = PoseEstimator()
            self.depth_processor = DepthProcessor()
            self.safety = SafetyMonitor(**(safety_config or {}))
        else:
            self.slam = None
            self.pose_estimator = None
            self.depth_processor = None
            self.safety = None

        self.object_detector = None
        if enable_object_detection:
            self.object_detector = object_detector or ObjectDetector(
                backend="yolo-world" if self._is_sim else None
            )
        self.map_navigator = (
            MapNavigator(map_config) if map_config is not None else None
        )
        controller_config = dict(navigation_config or {})
        if map_config is not None:
            controller_config.setdefault(
                "max_linear_velocity", map_config.max_linear_vel
            )
            controller_config.setdefault(
                "max_angular_velocity", map_config.max_angular_vel
            )
            controller_config.setdefault("goal_threshold", map_config.goal_threshold)
            goal_yaw_tolerance = map_config.goal_yaw_tolerance
        self.controller = VelocityController(**controller_config)

        # State
        self._state = PolicyState.IDLE
        self._goal: Optional[NavigationGoal] = None
        self._status = PolicyStatus()
        self._running = False
        self._start_time: Optional[float] = None
        self._last_sensor: SensorFrame | None = None
        self._last_obstacle_info: dict | None = None

        # Tracking
        self._object_not_found_count = 0
        self._max_object_not_found = 20  # completed detector results before giving up
        self._navigation_timeout = 120.0  # seconds (longer for sim)
        self._last_detection: Optional[ObjectDetection] = None
        self._last_detection_revision: int | None = None
        self._slam_ever_tracked = False
        self._slam_bootstrap_started: float | None = None
        self._object_search_started: float | None = None
        if object_track_timeout <= 0:
            raise ValueError("object_track_timeout must be positive")
        if not 0 < object_position_filter <= 1:
            raise ValueError("object_position_filter must be in (0, 1]")
        if goal_yaw_tolerance <= 0:
            raise ValueError("goal_yaw_tolerance must be positive")
        if not (
            0 < slam_bootstrap_angular_velocity <= self.controller.max_angular_velocity
        ):
            raise ValueError(
                "slam_bootstrap_angular_velocity must be positive and within the controller limit"
            )
        if slam_bootstrap_leg_seconds <= 0 or slam_bootstrap_timeout <= 0:
            raise ValueError("SLAM bootstrap durations must be positive")
        if not (
            0 < object_search_angular_velocity <= self.controller.max_angular_velocity
        ):
            raise ValueError(
                "object_search_angular_velocity must be positive and within the controller limit"
            )
        self._object_track_timeout = object_track_timeout
        self._object_position_filter = object_position_filter
        self._goal_yaw_tolerance = goal_yaw_tolerance
        self._slam_bootstrap_angular_velocity = slam_bootstrap_angular_velocity
        self._slam_bootstrap_leg_seconds = slam_bootstrap_leg_seconds
        self._slam_bootstrap_timeout = slam_bootstrap_timeout
        self._object_search_angular_velocity = object_search_angular_velocity

    def start(self) -> PolicyStatus:
        """Start the navigation policy.

        Initializes all components and transitions to SHOWING_CAMERA state.
        """
        logger.info(f"Starting navigation policy (sim={self._is_sim})")
        self._running = False
        self._start_time = time.time()
        self._slam_ever_tracked = False
        self._slam_bootstrap_started = None
        self._object_search_started = None

        try:
            if self.map_navigator is not None and self.map_navigator.grid is None:
                self.map_navigator.load_map()
            if self._is_sim:
                self.sim_env.initialize()
            elif self.robot:
                initialize_sensor_navigation(
                    self.robot, self.slam, self.pose_estimator, self.safety
                )
            elif self.slam:
                self.slam.initialize()

            if (
                not self._is_sim
                and self.object_detector is not None
                and not self.object_detector.initialize()
            ):
                raise RuntimeError(
                    "No live object detector is available; install the configured model"
                )
            if self.safety:
                self.safety.reset()
            if self.pose_estimator:
                self.pose_estimator.reset()
        except Exception:
            self._cleanup_components()
            raise

        self._running = True
        self._state = PolicyState.SHOWING_CAMERA
        self._update_status(message="Camera stream ready. Waiting for object name...")
        return self._status

    def supports_target(self, object_name: str) -> bool:
        """Whether the configured detector can admit this object command."""
        if self.object_detector is None:
            return False
        supports = getattr(self.object_detector, "supports_target", None)
        return bool(supports(object_name)) if callable(supports) else True

    def set_target(self, object_name: str) -> PolicyStatus:
        """Set the target object to navigate to.

        Args:
            object_name: Name of the object to find

        Returns:
            Updated PolicyStatus
        """
        if self._state not in (
            PolicyState.SHOWING_CAMERA,
            PolicyState.WAITING_FOR_OBJECT,
            PolicyState.LOCALIZING,
        ):
            self._update_status(message="Cannot set object in current state")
            return self._status

        logger.info(f"Target object set: {object_name}")
        if self.object_detector is None:
            return self._update_status(message="Object detection is disabled")
        resolve_target = getattr(self.object_detector, "resolve_target", None)
        if callable(resolve_target):
            resolved_target = resolve_target(object_name)
            if resolved_target is None:
                return self._update_status(
                    message=f"Object class '{object_name}' is not supported"
                )
            object_name = resolved_target
        set_detector_target = getattr(self.object_detector, "set_target", None)
        if callable(set_detector_target):
            set_detector_target(object_name)
        self._last_detection_revision = None
        self._goal = NavigationGoal(
            object_name=object_name,
            stand_off_distance=safe_stand_off_distance(object_name),
        )
        self._state = (
            PolicyState.LOCALIZING
            if self.map_navigator is not None and not self.map_navigator.alignment_ready
            else PolicyState.DETECTING
        )
        self._object_not_found_count = 0
        self._slam_bootstrap_started = None
        self._object_search_started = None
        self._start_time = time.time()
        self._update_status(message=f"Searching for '{object_name}'...")
        return self._status

    def set_object(self, object_name: str) -> PolicyStatus:
        """Alias for set_target (backward compatibility)."""
        return self.set_target(object_name)

    def set_pose_goal(self, x: float, y: float, yaw: float = 0.0) -> PolicyStatus:
        """Set a goal pose in the active world frame.

        With a fixed map this is a map-frame pose. Without one it is expressed
        in the current SLAM session frame.
        """
        if self._state in (PolicyState.IDLE, PolicyState.ERROR):
            return self._update_status(message="Cannot set pose goal in current state")
        pose = np.array([float(x), float(y), float(yaw)])
        if self.map_navigator is not None and not self.map_navigator.validate_goal(
            pose
        ):
            return self._update_status(message="Requested pose is occupied in the map")
        self._goal = NavigationGoal(
            object_name="map goal" if self.map_navigator is not None else "pose goal",
            kind="pose",
            approach_pose=pose,
        )
        if self.map_navigator is not None:
            self.map_navigator.set_goal(pose)
        self._start_time = time.time()
        self._state = (
            PolicyState.PLANNING
            if self.map_navigator is None or self.map_navigator.alignment_ready
            else PolicyState.LOCALIZING
        )
        return self._update_status(message="Pose goal accepted")

    def step(self) -> PolicyStatus:
        """Execute one step of the policy loop.

        This should be called at a fixed frequency (e.g., 10 Hz).

        Returns:
            Updated PolicyStatus
        """
        if not self._running:
            return self._update_status(message="Policy not running")

        # Get sensor data (from real robot or sim)
        if self._is_sim:
            return self._step_sim()
        else:
            sensor_data = self._get_sensor_data()
            if sensor_data is None:
                self._last_sensor = None
                self._stop_robot()
                return self._update_status(
                    state=PolicyState.ERROR, message="Failed to get sensor data"
                )
            self._last_sensor = sensor_data

            try:
                localized = localize_sensor_frame(
                    sensor_data,
                    slam=self.slam,
                    pose_estimator=self.pose_estimator,
                    depth_processor=self.depth_processor,
                    safety=self.safety,
                )
                self._last_obstacle_info = localized.obstacle_info
            except Exception as exc:
                logger.exception("Navigation localization failure")
                self._stop_robot()
                return self._update_status(
                    state=PolicyState.ERROR, message=f"Localization failed: {exc}"
                )
            if not localized.safety_status.is_safe:
                self._stop_robot()
                min_distance = float(
                    localized.obstacle_info.get("min_distance", float("inf"))
                )
                obstacle_limit = float(
                    getattr(self.safety, "min_obstacle_distance", 0.25)
                )
                if min_distance < obstacle_limit:
                    return self._update_status(
                        current_pose=localized.fused_pose,
                        safety_status=localized.safety_status,
                        velocity_command=VelocityCommand(),
                        message=(
                            f"Motion blocked by obstacle at {min_distance:.2f}m; "
                            "move it clear to resume"
                        ),
                    )
                return self._update_status(
                    state=PolicyState.ERROR,
                    current_pose=localized.fused_pose,
                    safety_status=localized.safety_status,
                    message=f"Safety violation: {localized.safety_status.reason}",
                )
            if localized.slam_pose.tracking_status != "OK":
                self._stop_robot()
                waiting = self._goal is None
                command = VelocityCommand()
                self._state = (
                    PolicyState.WAITING_FOR_OBJECT
                    if waiting and self.map_navigator is None
                    else PolicyState.LOCALIZING
                )
                message = (
                    "Waiting for ORB-SLAM tracking"
                    if waiting
                    else "ORB-SLAM tracking lost; holding target"
                )
                if not waiting and not self._slam_ever_tracked:
                    command, message = self._slam_bootstrap_command(localized)
                    if command.angular_z:
                        send_velocity(self.robot, command)
                return self._update_status(
                    current_pose=localized.fused_pose,
                    safety_status=localized.safety_status,
                    velocity_command=command,
                    message=message,
                )

            self._slam_ever_tracked = True
            self._slam_bootstrap_started = None

            if self.map_navigator is not None:
                try:
                    aligned, detail = self.map_navigator.update_alignment(
                        localized, self.depth_processor
                    )
                except Exception as exc:
                    logger.exception("Map alignment failed")
                    self._stop_robot()
                    return self._update_status(
                        state=PolicyState.ERROR,
                        safety_status=localized.safety_status,
                        message=f"Map alignment failed: {exc}",
                    )
                if not aligned:
                    self._state = PolicyState.LOCALIZING
                    obstacle_blocked = localized.obstacle_info.get(
                        "has_obstacle", False
                    )
                    spin_speed = (
                        self.map_navigator.config.localization_spin_speed
                        if self._goal is not None and not obstacle_blocked
                        else 0.0
                    )
                    command = VelocityCommand(angular_z=spin_speed)
                    send_velocity(self.robot, command)
                    return self._update_status(
                        current_pose=localized.fused_pose,
                        safety_status=localized.safety_status,
                        velocity_command=command,
                        message=(
                            f"{detail}; localization spin blocked by nearby obstacle"
                            if obstacle_blocked and self._goal is not None
                            else detail
                        ),
                    )
                if self._state == PolicyState.LOCALIZING:
                    if self._goal is None:
                        self._state = PolicyState.WAITING_FOR_OBJECT
                    elif self._goal.kind == "pose":
                        self._state = PolicyState.PLANNING
                    else:
                        self._state = PolicyState.DETECTING
            elif self._state == PolicyState.LOCALIZING:
                if self._goal is None:
                    self._state = PolicyState.WAITING_FOR_OBJECT
                elif self._goal.kind == "pose":
                    self._state = PolicyState.PLANNING
                elif self._goal.approach_pose is None:
                    self._state = PolicyState.DETECTING
                else:
                    self._state = PolicyState.NAVIGATING

            # Process based on state
            if self._state == PolicyState.SHOWING_CAMERA:
                self._state = PolicyState.WAITING_FOR_OBJECT
                return self._update_status(message="Ready to receive object name")

            elif self._state == PolicyState.WAITING_FOR_OBJECT:
                return self._update_status(message="Waiting for object name...")

            elif self._state == PolicyState.DETECTING:
                return self._step_detecting(sensor_data, localized)

            elif self._state in (
                PolicyState.PLANNING,
                PolicyState.NAVIGATING,
                PolicyState.TRACKING_OBJECT,
                PolicyState.ARRIVED,
            ):
                if self._goal is not None and self._goal.kind == "pose":
                    return self._step_pose_goal(sensor_data, localized)
                return self._step_navigating(sensor_data, localized)

            elif self._state == PolicyState.LOST:
                return self._update_status(message="Lost - cannot find object")

            elif self._state == PolicyState.ERROR:
                return self._update_status(message="Error state")

            return self._update_status()

    def _step_sim(self) -> PolicyStatus:
        """Process policy step for simulation mode."""
        frame = self.sim_env.get_frame()
        pose = self.sim_env.get_pose()

        if frame is None:
            return self._update_status(
                state=PolicyState.ERROR, message="No camera frame"
            )

        if self._state == PolicyState.SHOWING_CAMERA:
            self._state = PolicyState.WAITING_FOR_OBJECT
            return self._update_status(
                robot_pose=pose, message="Ready to receive object name"
            )

        elif self._state == PolicyState.WAITING_FOR_OBJECT:
            return self._update_status(
                robot_pose=pose, message="Waiting for object name..."
            )

        elif self._state == PolicyState.DETECTING:
            return self._step_sim_detecting(frame, pose)

        elif self._state in (
            PolicyState.NAVIGATING,
            PolicyState.TRACKING_OBJECT,
            PolicyState.ARRIVED,
        ):
            return self._step_sim_navigating(frame, pose)

        elif self._state == PolicyState.LOST:
            return self._update_status(
                robot_pose=pose, message="Lost - cannot find object"
            )

        elif self._state == PolicyState.ERROR:
            return self._update_status(robot_pose=pose, message="Error state")

        return self._update_status(robot_pose=pose)

    def _matching_target_detections(
        self, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        """Keep non-target detections out of policy state and telemetry."""
        if self._goal is None:
            return []
        target_name = self._goal.object_name.lower()
        return [
            detection
            for detection in detections
            if target_name in detection.label.lower()
        ]

    def _step_sim_detecting(self, frame: np.ndarray, pose: np.ndarray) -> PolicyStatus:
        """Step in DETECTING state for simulation."""
        detections = self._matching_target_detections(self.sim_env.get_detections())
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is None or target.position_3d is None:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                return self._update_status(
                    robot_pose=pose,
                    detections=detections,
                    message=f"Could not find '{self._goal.object_name}'",
                )
            return self._update_status(
                robot_pose=pose,
                detections=detections,
                message=f"Searching for '{self._goal.object_name}'... ({self._object_not_found_count}/{self._max_object_not_found})",
            )

        object_world = planar_detection_to_world(target.position_3d, pose)
        self._observe_object(target, object_world, pose)
        self._state = PolicyState.NAVIGATING
        return self._update_status(
            robot_pose=pose,
            detections=detections,
            message=f"World-frame goal acquired for '{self._goal.object_name}'",
        )

    def _step_sim_navigating(self, frame: np.ndarray, pose: np.ndarray) -> PolicyStatus:
        """Step in NAVIGATING state for simulation."""
        detections = self._matching_target_detections(self.sim_env.get_detections())
        target = self.object_detector.find_object(detections, self._goal.object_name)

        observed_now = target is not None and target.position_3d is not None
        if observed_now:
            object_world = planar_detection_to_world(target.position_3d, pose)
            self._observe_object(target, object_world, pose)
        elif not self._goal_is_fresh():
            self._state = PolicyState.LOST
            self._stop_robot()
            return self._update_status(
                robot_pose=pose,
                detections=detections,
                message="Lost object; world-frame goal expired",
            )

        if self._goal.approach_pose is None:
            cmd = VelocityCommand()
        elif observed_now and self._goal_reached(pose):
            self._state = PolicyState.ARRIVED
            cmd = VelocityCommand()
        else:
            self._state = PolicyState.NAVIGATING
            cmd = self.controller.compute_goal_velocity(
                pose,
                self._goal.approach_pose,
                yaw_tolerance=self._goal_yaw_tolerance,
            )

        self.sim_env.set_velocity(cmd.linear_x, cmd.linear_y, cmd.angular_z)

        state_text = (
            "holding goal pose" if self._state == PolicyState.ARRIVED else "navigating"
        )
        return self._update_status(
            robot_pose=pose,
            velocity_command=cmd,
            detections=detections,
            message=f"{state_text.capitalize()} for '{self._goal.object_name}'",
        )

    def _observe_object(
        self,
        detection: ObjectDetection,
        object_position_world: np.ndarray,
        robot_pose: np.ndarray,
    ) -> None:
        """Update the filtered object track and its derived approach pose."""
        self._last_detection = detection
        self._goal.detection = detection
        self._object_not_found_count = 0
        self._goal.object_position_world = blend_world_position(
            self._goal.object_position_world,
            object_position_world,
            self._object_position_filter,
        )
        self._goal.approach_pose = approach_pose(
            robot_pose,
            self._goal.object_position_world,
            self._goal.stand_off_distance,
        )
        self._goal.last_observed_monotonic = time.monotonic()

    def _goal_is_fresh(self) -> bool:
        observed = self._goal.last_observed_monotonic
        return (
            observed is not None
            and time.monotonic() - observed <= self._object_track_timeout
        )

    def _goal_reached(self, robot_pose: np.ndarray) -> bool:
        return (
            self._goal.approach_pose is not None
            and self.controller.has_reached_pose(
                robot_pose,
                self._goal.approach_pose,
                yaw_tolerance=self._goal_yaw_tolerance,
            )
        )

    def _stop_robot(self) -> None:
        """Best-effort stop that cannot turn a safety hold into a process crash."""
        try:
            if self._is_sim:
                self.sim_env.set_velocity(0.0, 0.0, 0.0)
            elif self.robot:
                stop = getattr(self.robot, "stop", None)
                if callable(stop):
                    stop()
                else:
                    send_velocity(self.robot)
        except RuntimeError:
            # Booster returns code 400 when its walking controller is already
            # unavailable. There is no accepted velocity command in that state;
            # preserve the policy and its safety telemetry instead of crashing.
            logger.exception("Robot locomotion controller rejected the stop command")

    def _slam_bootstrap_command(
        self, localized: LocalizedFrame
    ) -> tuple[VelocityCommand, str]:
        """Create bounded in-place excitation for first IMU-RGBD initialization."""
        obstacle_info = localized.obstacle_info
        min_distance = float(obstacle_info.get("min_distance", 0.0))
        if obstacle_info.get("sensor_blind", False) or obstacle_info.get(
            "has_obstacle", True
        ):
            return VelocityCommand(), (
                "IMU-RGBD initialization needs motion, but the localization spin "
                f"is blocked by depth at {min_distance:.2f}m"
            )

        now = time.monotonic()
        if self._slam_bootstrap_started is None:
            self._slam_bootstrap_started = now
        elapsed = now - self._slam_bootstrap_started
        if elapsed >= self._slam_bootstrap_timeout:
            self._state = PolicyState.LOST
            return VelocityCommand(), "IMU-RGBD initialization motion timed out"

        leg = int(elapsed / self._slam_bootstrap_leg_seconds)
        direction = 1.0 if leg % 2 == 0 else -1.0
        command = VelocityCommand(
            angular_z=direction * self._slam_bootstrap_angular_velocity
        )
        return command, (
            "Initializing IMU-RGBD with guarded in-place motion "
            f"({elapsed:.1f}/{self._slam_bootstrap_timeout:.0f}s)"
        )

    def _object_search_command(
        self, localized: LocalizedFrame
    ) -> tuple[VelocityCommand, str]:
        """Scan slowly for the requested object without translating."""
        obstacle_info = localized.obstacle_info
        min_distance = float(obstacle_info.get("min_distance", 0.0))
        if obstacle_info.get("sensor_blind", False) or obstacle_info.get(
            "has_obstacle", True
        ):
            return (
                VelocityCommand(),
                f"search spin blocked by depth at {min_distance:.2f}m",
            )
        if self._object_search_started is None:
            self._object_search_started = time.monotonic()
        elapsed = time.monotonic() - self._object_search_started
        direction = 1.0 if int(elapsed / 15.0) % 2 == 0 else -1.0
        return (
            VelocityCommand(angular_z=direction * self._object_search_angular_velocity),
            "scanning in place",
        )

    def _step_detecting(
        self, sensor_data: SensorFrame, localized: LocalizedFrame
    ) -> PolicyStatus:
        """Step in DETECTING state."""
        rgb = sensor_data.rgb
        depth = sensor_data.depth
        camera_info = sensor_data.camera_info

        # Detect objects
        detections = self._matching_target_detections(
            self.object_detector.detect(rgb, depth, camera_info)
        )
        local_pose = localized.fused_pose.position_2d
        robot_pose = (
            self.map_navigator.to_map_pose(local_pose)
            if self.map_navigator is not None
            else local_pose
        )
        status_pose = self._pose_in_active_frame(localized.fused_pose, robot_pose)
        search_command, search_detail = self._object_search_command(localized)
        revision = getattr(self.object_detector, "result_revision", None)
        if revision is not None and revision == self._last_detection_revision:
            send_velocity(self.robot, search_command)
            return self._update_status(
                current_pose=status_pose,
                safety_status=localized.safety_status,
                velocity_command=search_command,
                detections=detections,
                message=f"Searching for '{self._goal.object_name}'; {search_detail}",
            )
        self._last_detection_revision = revision
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None and target.position_3d is not None:
            self._last_detection = target
            self._goal.detection = target
            self._goal.last_observed_monotonic = time.monotonic()
            self._object_not_found_count = 0
            self._object_search_started = None
            self._state = PolicyState.NAVIGATING
            self._stop_robot()
            return self._update_status(
                current_pose=status_pose,
                safety_status=localized.safety_status,
                velocity_command=VelocityCommand(),
                detections=detections,
                message=f"Found '{self._goal.object_name}'; acquiring world-frame goal",
            )
        else:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                self._stop_robot()
                return self._update_status(
                    current_pose=status_pose,
                    safety_status=localized.safety_status,
                    velocity_command=VelocityCommand(),
                    detections=detections,
                    message=f"Could not find '{self._goal.object_name}'",
                )
            send_velocity(self.robot, search_command)
            return self._update_status(
                current_pose=status_pose,
                safety_status=localized.safety_status,
                velocity_command=search_command,
                detections=detections,
                message=(
                    f"Searching for '{self._goal.object_name}'... "
                    f"({self._object_not_found_count}/{self._max_object_not_found}); "
                    f"{search_detail}"
                ),
            )

    def _step_navigating(
        self,
        sensor_data: SensorFrame | dict,
        localized: LocalizedFrame | None = None,
    ) -> PolicyStatus:
        """Step in NAVIGATING state."""
        if isinstance(sensor_data, dict):
            sensor_data = SensorFrame(
                rgb=sensor_data["rgb"],
                depth=sensor_data["depth"],
                timestamp=sensor_data.get("timestamp", 0.0),
                camera_info=sensor_data.get("camera_info"),
                imu_rpy=np.asarray(sensor_data.get("imu_rpy", np.zeros(3))),
                imu_samples=sensor_data.get("imu_samples"),
                odometry=np.asarray(sensor_data.get("odometry", np.zeros(3))),
            )
        if localized is None:
            localized = localize_sensor_frame(
                sensor_data,
                slam=self.slam,
                pose_estimator=self.pose_estimator,
                depth_processor=self.depth_processor,
                safety=self.safety,
            )
        rgb = sensor_data.rgb
        depth = sensor_data.depth
        camera_info = sensor_data.camera_info
        slam_pose = localized.slam_pose
        fused_pose = localized.fused_pose
        safety_status = localized.safety_status
        if not safety_status.is_safe:
            self._state = PolicyState.ERROR
            self._stop_robot()
            return self._update_status(
                safety_status=safety_status,
                message=f"Safety violation: {safety_status.reason}",
            )

        obstacle_info = localized.obstacle_info

        # Check navigation timeout
        if (
            self._start_time
            and (time.time() - self._start_time) > self._navigation_timeout
        ):
            self._state = PolicyState.LOST
            self._stop_robot()
            return self._update_status(message="Navigation timeout")

        # Detect object again (for tracking)
        detections = self._matching_target_detections(
            self.object_detector.detect(rgb, depth, camera_info)
        )
        revision = getattr(self.object_detector, "result_revision", None)
        new_detection_result = (
            revision is None or revision != self._last_detection_revision
        )
        if new_detection_result:
            self._last_detection_revision = revision
        target = self.object_detector.find_object(detections, self._goal.object_name)

        observed_now = (
            new_detection_result
            and target is not None
            and target.position_3d is not None
            and slam_pose.tracking_status == "OK"
        )
        local_robot_pose = fused_pose.position_2d
        robot_pose = (
            self.map_navigator.to_map_pose(local_robot_pose)
            if self.map_navigator is not None
            else local_robot_pose
        )
        status_pose = self._pose_in_active_frame(fused_pose, robot_pose)
        if observed_now:
            object_world = (
                body_point_to_world(target.position_3d, robot_pose)
                if target.coordinate_frame == "body"
                else camera_point_to_world(target.position_3d, slam_pose.to_matrix())
            )
            if target.coordinate_frame != "body" and self.map_navigator is not None:
                object_world = self.map_navigator.transform_slam_point(object_world)
            self._observe_object(target, object_world, robot_pose)
        elif not self._goal_is_fresh():
            self._state = PolicyState.LOST
            self._stop_robot()
            return self._update_status(
                current_pose=status_pose,
                safety_status=safety_status,
                detections=detections,
                message="Lost object; world-frame goal expired",
            )

        route_message = None
        if self._goal.approach_pose is None:
            cmd = VelocityCommand()
        elif self.map_navigator is not None:
            route = self._map_route(robot_pose, obstacle_info)
            route_message = route.message
            if route.failed:
                self._state = PolicyState.LOST
                cmd = VelocityCommand()
            elif route.arrived and observed_now:
                self._state = PolicyState.ARRIVED
                cmd = VelocityCommand()
            else:
                self._state = PolicyState.NAVIGATING
                cmd = route.command
        elif observed_now and self._goal_reached(robot_pose):
            self._state = PolicyState.ARRIVED
            cmd = VelocityCommand()
        else:
            self._state = PolicyState.NAVIGATING
            cmd = self.controller.compute_goal_velocity(
                current_pose=robot_pose,
                goal_pose=self._goal.approach_pose,
                obstacle_info=obstacle_info,
                yaw_tolerance=self._goal_yaw_tolerance,
            )

        # Send velocity command
        if self.robot:
            send_velocity(self.robot, cmd)

        state_text = (
            "holding goal pose" if self._state == PolicyState.ARRIVED else "navigating"
        )
        return self._update_status(
            current_pose=status_pose,
            safety_status=safety_status,
            velocity_command=cmd,
            detections=detections,
            message=route_message
            or f"{state_text.capitalize()} for '{self._goal.object_name}'",
        )

    def _step_pose_goal(
        self,
        sensor_data: SensorFrame,
        localized: LocalizedFrame | None,
    ) -> PolicyStatus:
        """Follow an explicit pose through the same localized safety loop."""
        if localized is None:
            localized = localize_sensor_frame(
                sensor_data,
                slam=self.slam,
                pose_estimator=self.pose_estimator,
                depth_processor=self.depth_processor,
                safety=self.safety,
            )
        if not localized.safety_status.is_safe:
            self._stop_robot()
            return self._update_status(
                state=PolicyState.ERROR,
                current_pose=localized.fused_pose,
                safety_status=localized.safety_status,
                message=f"Safety violation: {localized.safety_status.reason}",
            )
        local_pose = localized.fused_pose.position_2d
        robot_pose = (
            self.map_navigator.to_map_pose(local_pose)
            if self.map_navigator is not None
            else local_pose
        )
        status_pose = self._pose_in_active_frame(localized.fused_pose, robot_pose)
        if self.map_navigator is not None:
            route = self._map_route(robot_pose, localized.obstacle_info)
            command = route.command
            message = route.message
            state = (
                PolicyState.LOST
                if route.failed
                else PolicyState.ARRIVED if route.arrived else PolicyState.NAVIGATING
            )
        else:
            arrived = self.controller.has_reached_pose(
                robot_pose,
                self._goal.approach_pose,
                yaw_tolerance=self._goal_yaw_tolerance,
            )
            command = (
                VelocityCommand()
                if arrived
                else self.controller.compute_goal_velocity(
                    robot_pose,
                    self._goal.approach_pose,
                    localized.obstacle_info,
                    yaw_tolerance=self._goal_yaw_tolerance,
                )
            )
            state = PolicyState.ARRIVED if arrived else PolicyState.NAVIGATING
            message = "Arrived at pose goal" if arrived else "Navigating to pose goal"
        self._state = state
        send_velocity(self.robot, command)
        return self._update_status(
            current_pose=status_pose,
            safety_status=localized.safety_status,
            velocity_command=command,
            message=message,
        )

    @staticmethod
    def _pose_in_active_frame(fused_pose: FusedPose, pose_2d: np.ndarray) -> FusedPose:
        position = np.asarray(fused_pose.position, dtype=float).copy()
        position[:2] = pose_2d[:2]
        return FusedPose(
            position=position,
            yaw=float(pose_2d[2]),
            timestamp=fused_pose.timestamp,
            confidence=fused_pose.confidence,
            source=(
                "map_fused"
                if not np.allclose(pose_2d, fused_pose.position_2d)
                else fused_pose.source
            ),
        )

    def _map_route(self, robot_pose: np.ndarray, obstacle_info: dict) -> MapRouteResult:
        """Fail closed if fixed-map planning cannot produce a safe command."""
        try:
            return self.map_navigator.route(
                robot_pose,
                self._goal.approach_pose,
                self.controller,
                obstacle_info,
            )
        except Exception as exc:
            logger.exception("Map routing failed")
            return MapRouteResult(failed=True, message=f"Map routing failed: {exc}")

    def stop(self) -> PolicyStatus:
        """Stop the navigation policy."""
        logger.info("Stopping navigation policy")
        self._running = False
        self._state = PolicyState.IDLE

        self._cleanup_components()

        self._update_status(message="Stopped")
        return self._status

    def _cleanup_components(self) -> None:
        """Best-effort cleanup, including partially initialized startup."""
        cleanup = [self._stop_robot]
        shutdown_slam = getattr(self.slam, "shutdown", None)
        if callable(shutdown_slam):
            cleanup.append(shutdown_slam)
        close_detector = getattr(self.object_detector, "close", None)
        if callable(close_detector):
            cleanup.append(close_detector)
        if self.sim_env:
            cleanup.append(self.sim_env.stop)
        for action in cleanup:
            try:
                action()
            except Exception:
                logger.exception("Navigation component cleanup failed")

    def _get_sensor_data(self) -> Optional[SensorFrame]:
        """Get latest sensor data from robot."""
        if self.robot is None:
            return None

        try:
            return read_sensor_frame(self.robot)
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return None

    def _update_status(
        self,
        state: Optional[PolicyState] = None,
        current_pose: Optional["FusedPose"] = None,
        safety_status: Optional["SafetyStatus"] = None,
        velocity_command: Optional[VelocityCommand] = None,
        message: str = "",
        detections: Optional[list[ObjectDetection]] = None,
        robot_pose: Optional[np.ndarray] = None,
    ) -> PolicyStatus:
        """Update and return policy status."""
        if state is not None:
            self._state = state

        pose = current_pose
        if self.map_navigator is not None and not self.map_navigator.alignment_ready:
            pose = None
        else:
            get_pose = getattr(self.pose_estimator, "get_pose", None)
            if pose is None and callable(get_pose):
                pose = get_pose()
                if pose is not None and self.map_navigator is not None:
                    map_pose = self.map_navigator.to_map_pose(pose.position_2d)
                    pose = self._pose_in_active_frame(pose, map_pose)

        planned_path = self._planned_path(pose, robot_pose)

        self._status = PolicyStatus(
            state=self._state,
            current_goal=self._goal,
            current_pose=pose,
            safety_status=safety_status,
            velocity_command=velocity_command,
            message=message,
            detections=detections or [],
            robot_pose=robot_pose,
            planned_path=planned_path,
            obstacle_info=self._last_obstacle_info,
        )
        return self._status

    def _planned_path(
        self, pose: FusedPose | None, robot_pose: np.ndarray | None
    ) -> np.ndarray:
        """Expose the current controller route in the active world frame."""
        visible_states = {
            PolicyState.PLANNING,
            PolicyState.NAVIGATING,
            PolicyState.TRACKING_OBJECT,
            PolicyState.ARRIVED,
        }
        if (
            self._state not in visible_states
            or self._goal is None
            or self._goal.approach_pose is None
        ):
            return np.empty((0, 3), dtype=float)
        if self.map_navigator is not None:
            map_path = self.map_navigator.current_path
            if len(map_path):
                return map_path
        current = (
            np.asarray(robot_pose, dtype=float)
            if robot_pose is not None
            else pose.position_2d if pose is not None else None
        )
        if current is None:
            return np.empty((0, 3), dtype=float)
        goal = np.asarray(self._goal.approach_pose, dtype=float)
        return np.asarray(
            [[current[0], current[1], 0.0], [goal[0], goal[1], 0.0]],
            dtype=float,
        )

    def reset(self) -> PolicyStatus:
        """Reset the policy to initial state."""
        self._state = PolicyState.SHOWING_CAMERA if self._running else PolicyState.IDLE
        self._goal = None
        self._object_not_found_count = 0
        self._last_detection = None
        self._last_detection_revision = None
        self._slam_bootstrap_started = None
        self._object_search_started = None
        if self.map_navigator is not None:
            self.map_navigator.reset()
        self._stop_robot()
        return self._update_status(message="Reset complete")

    @property
    def status(self) -> PolicyStatus:
        return self._status

    @property
    def state(self) -> PolicyState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_sensor(self) -> SensorFrame | None:
        return self._last_sensor

    @property
    def map_alignment_ready(self) -> bool:
        return self.map_navigator is not None and self.map_navigator.alignment_ready

    @property
    def grid(self):
        return self.map_navigator.grid if self.map_navigator is not None else None

    def transform_slam_points(self, points: np.ndarray) -> np.ndarray:
        if self.map_navigator is None:
            return np.asarray(points, dtype=float).copy()
        return self.map_navigator.transform_slam_points(points)

    def render_map(self) -> np.ndarray:
        if self.map_navigator is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.map_navigator.render_map()
