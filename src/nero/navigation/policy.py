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
    DETECTING = "detecting"
    NAVIGATING = "navigating"
    TRACKING_OBJECT = "tracking_object"
    ARRIVED = "arrived"
    LOST = "lost"
    ERROR = "error"


@dataclass
class NavigationGoal:
    """A navigation goal."""

    object_name: str
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


class NavigationPolicy:
    """Main policy loop for object-following navigation.

    Supports both real robot (via RobotInterface) and simulation (via SimEnvironment).

    State machine:
    IDLE -> SHOWING_CAMERA -> WAITING_FOR_OBJECT -> DETECTING -> NAVIGATING -> ARRIVED

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

        self.object_detector = object_detector or ObjectDetector()
        self.controller = VelocityController(**(navigation_config or {}))

        # State
        self._state = PolicyState.IDLE
        self._goal: Optional[NavigationGoal] = None
        self._status = PolicyStatus()
        self._running = False
        self._start_time: Optional[float] = None

        # Tracking
        self._object_not_found_count = 0
        self._max_object_not_found = 10  # frames before giving up
        self._navigation_timeout = 120.0  # seconds (longer for sim)
        self._last_detection: Optional[ObjectDetection] = None
        self._last_detection_revision: int | None = None
        if object_track_timeout <= 0:
            raise ValueError("object_track_timeout must be positive")
        if not 0 < object_position_filter <= 1:
            raise ValueError("object_position_filter must be in (0, 1]")
        if goal_yaw_tolerance <= 0:
            raise ValueError("goal_yaw_tolerance must be positive")
        self._object_track_timeout = object_track_timeout
        self._object_position_filter = object_position_filter
        self._goal_yaw_tolerance = goal_yaw_tolerance

    def start(self) -> PolicyStatus:
        """Start the navigation policy.

        Initializes all components and transitions to SHOWING_CAMERA state.
        """
        logger.info(f"Starting navigation policy (sim={self._is_sim})")
        self._running = True
        self._start_time = time.time()

        # Initialize components
        if self._is_sim:
            self.sim_env.initialize()
        elif self.robot:
            initialize_sensor_navigation(
                self.robot, self.slam, self.pose_estimator, self.safety
            )
        elif self.slam:
            self.slam.initialize()

        if not self._is_sim and not self.object_detector.initialize():
            raise RuntimeError(
                "No live object detector is available; install the configured ONNX model"
            )
        if self.safety:
            self.safety.reset()
        if self.pose_estimator:
            self.pose_estimator.reset()

        self._state = PolicyState.SHOWING_CAMERA
        self._update_status(message="Camera stream ready. Waiting for object name...")
        return self._status

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
        ):
            self._update_status(message="Cannot set object in current state")
            return self._status

        logger.info(f"Target object set: {object_name}")
        set_detector_target = getattr(self.object_detector, "set_target", None)
        if callable(set_detector_target):
            set_detector_target(object_name)
        self._last_detection_revision = None
        self._goal = NavigationGoal(
            object_name=object_name,
            stand_off_distance=safe_stand_off_distance(object_name),
        )
        self._state = PolicyState.DETECTING
        self._object_not_found_count = 0
        self._start_time = time.time()
        self._update_status(message=f"Searching for '{object_name}'...")
        return self._status

    def set_object(self, object_name: str) -> PolicyStatus:
        """Alias for set_target (backward compatibility)."""
        return self.set_target(object_name)

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
                return self._update_status(
                    state=PolicyState.ERROR, message="Failed to get sensor data"
                )

            # Process based on state
            if self._state == PolicyState.SHOWING_CAMERA:
                self._state = PolicyState.WAITING_FOR_OBJECT
                return self._update_status(message="Ready to receive object name")

            elif self._state == PolicyState.WAITING_FOR_OBJECT:
                return self._update_status(message="Waiting for object name...")

            elif self._state == PolicyState.DETECTING:
                return self._step_detecting(sensor_data)

            elif self._state in (
                PolicyState.NAVIGATING,
                PolicyState.TRACKING_OBJECT,
                PolicyState.ARRIVED,
            ):
                return self._step_navigating(sensor_data)

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
        """Stop the robot."""
        if self._is_sim:
            self.sim_env.set_velocity(0.0, 0.0, 0.0)
        elif self.robot:
            self.robot.stop()

    def _step_detecting(self, sensor_data: SensorFrame) -> PolicyStatus:
        """Step in DETECTING state."""
        rgb = sensor_data.rgb
        depth = sensor_data.depth
        camera_info = sensor_data.camera_info

        # Detect objects
        detections = self._matching_target_detections(
            self.object_detector.detect(rgb, depth, camera_info)
        )
        revision = getattr(self.object_detector, "result_revision", None)
        if revision is not None and revision == self._last_detection_revision:
            return self._update_status(
                detections=detections,
                message=f"Searching for '{self._goal.object_name}'...",
            )
        self._last_detection_revision = revision
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None and target.position_3d is not None:
            self._last_detection = target
            self._goal.detection = target
            self._goal.last_observed_monotonic = time.monotonic()
            self._object_not_found_count = 0
            self._state = PolicyState.NAVIGATING
            return self._update_status(
                detections=detections,
                message=f"Found '{self._goal.object_name}'; acquiring world-frame goal",
            )
        else:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                return self._update_status(
                    detections=detections,
                    message=f"Could not find '{self._goal.object_name}'",
                )
            return self._update_status(
                detections=detections,
                message=f"Searching for '{self._goal.object_name}'... ({self._object_not_found_count}/{self._max_object_not_found})",
            )

    def _step_navigating(self, sensor_data: SensorFrame | dict) -> PolicyStatus:
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
            and
            target is not None
            and target.position_3d is not None
            and slam_pose.tracking_status == "OK"
        )
        robot_pose = fused_pose.position_2d
        if observed_now:
            object_world = (
                body_point_to_world(target.position_3d, robot_pose)
                if target.coordinate_frame == "body"
                else camera_point_to_world(target.position_3d, slam_pose.to_matrix())
            )
            self._observe_object(target, object_world, robot_pose)
        elif not self._goal_is_fresh():
            self._state = PolicyState.LOST
            if self.robot:
                self.robot.stop()
            return self._update_status(
                current_pose=fused_pose,
                safety_status=safety_status,
                detections=detections,
                message="Lost object; world-frame goal expired",
            )

        if self._goal.approach_pose is None:
            cmd = VelocityCommand()
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
            current_pose=fused_pose,
            safety_status=safety_status,
            velocity_command=cmd,
            detections=detections,
            message=f"{state_text.capitalize()} for '{self._goal.object_name}'",
        )

    def stop(self) -> PolicyStatus:
        """Stop the navigation policy."""
        logger.info("Stopping navigation policy")
        self._running = False
        self._state = PolicyState.IDLE

        # Stop robot
        self._stop_robot()

        if self.slam:
            self.slam.shutdown()

        close_detector = getattr(self.object_detector, "close", None)
        if callable(close_detector):
            close_detector()
        if self.sim_env:
            self.sim_env.stop()

        self._update_status(message="Stopped")
        return self._status

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
        if pose is None and self.pose_estimator:
            pose = self.pose_estimator.get_pose()

        self._status = PolicyStatus(
            state=self._state,
            current_goal=self._goal,
            current_pose=pose,
            safety_status=safety_status,
            velocity_command=velocity_command,
            message=message,
            detections=detections or [],
            robot_pose=robot_pose,
        )
        return self._status

    def reset(self) -> PolicyStatus:
        """Reset the policy to initial state."""
        self._state = PolicyState.SHOWING_CAMERA if self._running else PolicyState.IDLE
        self._goal = None
        self._object_not_found_count = 0
        self._last_detection = None
        self._last_detection_revision = None
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
