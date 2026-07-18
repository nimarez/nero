"""Main navigation policy for object-following behavior.

This module implements the agent/policy loop:
1. Show external camera stream of a space
2. Accept object name from user
3. Detect the object in the camera stream
4. Navigate to the object while avoiding obstacles
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nero.perception.object_detector import ObjectDetector, ObjectDetection
from nero.perception.depth_processor import DepthProcessor
from nero.slam.orb_slam3_node import ORBSLAM3Node, SLAMPose
from nero.slam.pose_estimator import PoseEstimator, FusedPose
from nero.navigation.controller import VelocityController, VelocityCommand
from nero.navigation.safety import SafetyMonitor, SafetyStatus

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
    position: Optional[np.ndarray] = None  # [x, y, z] in world frame
    detection: Optional[ObjectDetection] = None
    target_distance: float = 1.0  # Stop this far from object


@dataclass
class PolicyStatus:
    """Current status of the navigation policy."""
    state: PolicyState = PolicyState.IDLE
    current_goal: Optional[NavigationGoal] = None
    current_pose: Optional[FusedPose] = None
    safety_status: Optional[SafetyStatus] = None
    velocity_command: Optional[VelocityCommand] = None
    message: str = ""
    detections: list[ObjectDetection] = field(default_factory=list)


class NavigationPolicy:
    """Main policy loop for object-following navigation.

    State machine:
    IDLE -> SHOWING_CAMERA -> WAITING_FOR_OBJECT -> DETECTING -> NAVIGATING -> ARRIVED

    Transitions:
    - IDLE: start() -> SHOWING_CAMERA
    - SHOWING_CAMERA: (auto) -> WAITING_FOR_OBJECT
    - WAITING_FOR_OBJECT: set_object(name) -> DETECTING
    - DETECTING: object_found -> NAVIGATING, not_found -> WAITING_FOR_OBJECT
    - NAVIGATING: arrived -> ARRIVED, lost -> LOST
    - TRACKING_OBJECT: arrived -> ARRIVED, lost -> LOST
    - Any state: stop() -> IDLE, safety_violation -> ERROR
    """

    def __init__(
        self,
        robot=None,  # RobotInterface
        slam_config=None,
        navigation_config=None,
        safety_config=None,
    ):
        # Components
        self.robot = robot
        self.slam = ORBSLAM3Node(config=slam_config)
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetector()
        self.depth_processor = DepthProcessor()
        self.controller = VelocityController(**(navigation_config or {}))
        self.safety = SafetyMonitor(**(safety_config or {}))

        # State
        self._state = PolicyState.IDLE
        self._goal: Optional[NavigationGoal] = None
        self._status = PolicyStatus()
        self._running = False
        self._start_time: Optional[float] = None

        # Tracking
        self._object_not_found_count = 0
        self._max_object_not_found = 10  # frames before giving up
        self._navigation_timeout = 60.0  # seconds
        self._last_detection: Optional[ObjectDetection] = None

    def start(self) -> PolicyStatus:
        """Start the navigation policy.

        Initializes all components and transitions to SHOWING_CAMERA state.
        """
        logger.info("Starting navigation policy")
        self._running = True
        self._start_time = time.time()

        # Initialize components
        if self.robot:
            self.robot.initialize()
            camera_info = self.robot.get_camera_info()
            self.slam.initialize(camera_info)
        else:
            self.slam.initialize()

        self.object_detector.initialize()
        self.safety.reset()
        self.pose_estimator.reset()

        self._state = PolicyState.SHOWING_CAMERA
        self._update_status(message="Camera stream ready. Waiting for object name...")
        return self._status

    def set_object(self, object_name: str) -> PolicyStatus:
        """Set the target object to navigate to.

        Args:
            object_name: Name of the object to find

        Returns:
            Updated PolicyStatus
        """
        if self._state not in (PolicyState.SHOWING_CAMERA, PolicyState.WAITING_FOR_OBJECT):
            self._update_status(message="Cannot set object in current state")
            return self._status

        logger.info(f"Target object set: {object_name}")
        self._goal = NavigationGoal(object_name=object_name)
        self._state = PolicyState.DETECTING
        self._object_not_found_count = 0
        self._update_status(message=f"Searching for '{object_name}'...")
        return self._status

    def step(self) -> PolicyStatus:
        """Execute one step of the policy loop.

        This should be called at a fixed frequency (e.g., 10 Hz).

        Returns:
            Updated PolicyStatus
        """
        if not self._running:
            return self._update_status(message="Policy not running")

        # Get sensor data
        sensor_data = self._get_sensor_data()
        if sensor_data is None:
            return self._update_status(
                state=PolicyState.ERROR,
                message="Failed to get sensor data"
            )

        # Process based on state
        if self._state == PolicyState.SHOWING_CAMERA:
            self._state = PolicyState.WAITING_FOR_OBJECT
            return self._update_status(message="Ready to receive object name")

        elif self._state == PolicyState.WAITING_FOR_OBJECT:
            return self._update_status(message="Waiting for object name...")

        elif self._state == PolicyState.DETECTING:
            return self._step_detecting(sensor_data)

        elif self._state in (PolicyState.NAVIGATING, PolicyState.TRACKING_OBJECT):
            return self._step_navigating(sensor_data)

        elif self._state == PolicyState.ARRIVED:
            return self._update_status(message=f"Arrived at '{self._goal.object_name}'")

        elif self._state == PolicyState.LOST:
            return self._update_status(message="Lost - cannot find object")

        elif self._state == PolicyState.ERROR:
            return self._update_status(message="Error state")

        return self._update_status()

    def _step_detecting(self, sensor_data: dict) -> PolicyStatus:
        """Step in DETECTING state."""
        rgb = sensor_data["rgb"]
        depth = sensor_data["depth"]
        camera_info = sensor_data.get("camera_info")

        # Detect objects
        detections = self.object_detector.detect(rgb, depth, camera_info)
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None:
            self._last_detection = target
            self._goal.detection = target
            self._object_not_found_count = 0

            # Check if object is close enough
            if target.distance < self._goal.target_distance:
                self._state = PolicyState.ARRIVED
                return self._update_status(
                    detections=detections,
                    message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m"
                )

            # Start navigating
            self._state = PolicyState.NAVIGATING
            return self._update_status(
                detections=detections,
                message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m, navigating..."
            )
        else:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                return self._update_status(
                    detections=detections,
                    message=f"Could not find '{self._goal.object_name}'"
                )
            return self._update_status(
                detections=detections,
                message=f"Searching for '{self._goal.object_name}'... ({self._object_not_found_count}/{self._max_object_not_found})"
            )

    def _step_navigating(self, sensor_data: dict) -> PolicyStatus:
        """Step in NAVIGATING state."""
        rgb = sensor_data["rgb"]
        depth = sensor_data["depth"]
        camera_info = sensor_data.get("camera_info")
        imu_rpy = sensor_data.get("imu_rpy")
        odom = sensor_data.get("odometry")

        # Update SLAM
        slam_pose = self.slam.track_frame(rgb, depth)

        # Update pose estimator
        fused_pose = self.pose_estimator.update(
            slam_pose=slam_pose,
            odom_pose=odom,
            imu_rpy=imu_rpy,
        )

        # Check safety
        safety_status = self.safety.check_safety(
            imu_rpy=imu_rpy,
            slam_tracking=slam_pose.tracking_status == "OK",
        )
        if not safety_status.is_safe:
            self._state = PolicyState.ERROR
            return self._update_status(
                safety_status=safety_status,
                message=f"Safety violation: {safety_status.reason}"
            )

        # Process depth for obstacles
        depth_m = self.depth_processor.preprocess(depth)
        obstacle_info = self.depth_processor.detect_obstacles(depth_m)

        # Check navigation timeout
        if self._start_time and (time.time() - self._start_time) > self._navigation_timeout:
            self._state = PolicyState.LOST
            return self._update_status(message="Navigation timeout")

        # Detect object again (for tracking)
        detections = self.object_detector.detect(rgb, depth, camera_info)
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None:
            self._last_detection = target
            self._goal.detection = target
            self._object_not_found_count = 0

            # Check if arrived
            if target.distance < self._goal.target_distance:
                self._state = PolicyState.ARRIVED
                return self._update_status(
                    detections=detections,
                    message=f"Arrived at '{self._goal.object_name}'"
                )

            # Track object
            cmd = self.controller.compute_object_tracking_velocity(
                object_position=target.position_3d,
                obstacle_info=obstacle_info,
                target_distance=self._goal.target_distance,
            )
        else:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                return self._update_status(
                    detections=detections,
                    message="Lost object during navigation"
                )
            # Continue toward last known position
            if self._last_detection and self._last_detection.position_3d is not None:
                cmd = self.controller.compute_object_tracking_velocity(
                    object_position=self._last_detection.position_3d,
                    obstacle_info=obstacle_info,
                    target_distance=self._goal.target_distance,
                )
            else:
                cmd = VelocityCommand()

        # Send velocity command
        if self.robot:
            self.robot.set_velocity(cmd.linear_x, cmd.angular_z)

        return self._update_status(
            current_pose=fused_pose,
            safety_status=safety_status,
            velocity_command=cmd,
            detections=detections,
            message=f"Navigating to '{self._goal.object_name}' ({target.distance if target else 'unknown':.2f}m)"
        )

    def stop(self) -> PolicyStatus:
        """Stop the navigation policy."""
        logger.info("Stopping navigation policy")
        self._running = False
        self._state = PolicyState.IDLE

        # Stop robot
        if self.robot:
            self.robot.stop()

        self.slam.shutdown()
        self._update_status(message="Stopped")
        return self._status

    def _get_sensor_data(self) -> Optional[dict]:
        """Get latest sensor data from robot."""
        if self.robot is None:
            return None

        try:
            rgb = self.robot.get_rgb_image()
            depth = self.robot.get_depth_image()
            camera_info = self.robot.get_camera_info()
            imu_rpy = self.robot.get_imu_rpy()
            odometry = self.robot.get_odometry()

            return {
                "rgb": rgb,
                "depth": depth,
                "camera_info": camera_info,
                "imu_rpy": imu_rpy,
                "odometry": odometry,
            }
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return None

    def _update_status(
        self,
        state: Optional[PolicyState] = None,
        current_pose: Optional[FusedPose] = None,
        safety_status: Optional[SafetyStatus] = None,
        velocity_command: Optional[VelocityCommand] = None,
        message: str = "",
        detections: Optional[list[ObjectDetection]] = None,
    ) -> PolicyStatus:
        """Update and return policy status."""
        if state is not None:
            self._state = state

        self._status = PolicyStatus(
            state=self._state,
            current_goal=self._goal,
            current_pose=current_pose or self.pose_estimator.get_pose(),
            safety_status=safety_status,
            velocity_command=velocity_command,
            message=message,
            detections=detections or [],
        )
        return self._status

    @property
    def status(self) -> PolicyStatus:
        return self._status

    @property
    def state(self) -> PolicyState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running