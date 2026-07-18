"""Main navigation policy for object-following behavior.

This module implements the agent/policy loop:
1. Read the K1's built-in RGB-D camera stream
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
from nero.navigation.controller import VelocityController, VelocityCommand

# These modules are hardware-independent. The vendor SDK is isolated inside the
# injected robot adapter, so policies work unchanged with ROS simulation.
try:
    from nero.perception.depth_processor import DepthProcessor
    from nero.slam.orb_slam3_node import ORBSLAM3Node, SLAMPose
    from nero.slam.pose_estimator import PoseEstimator, FusedPose
    from nero.navigation.safety import SafetyMonitor, SafetyStatus

    HAS_BOOSTEROS = True
except ImportError:
    DepthProcessor = None
    ORBSLAM3Node = None
    SLAMPose = None
    PoseEstimator = None
    FusedPose = None
    SafetyMonitor = None
    SafetyStatus = None
    HAS_BOOSTEROS = False

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
    ):
        # Environment: real robot or simulation
        self.robot = robot
        self.sim_env = sim_env
        self._is_sim = sim_env is not None

        # Components are required for every sensor-backed robot adapter. The
        # lightweight in-process SimEnvironment has its own synthetic path.
        if not self._is_sim:
            if not HAS_BOOSTEROS:
                raise RuntimeError("navigation dependencies are unavailable")
            self.slam = ORBSLAM3Node(config=slam_config, **(slam_options or {}))
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
            self.robot.initialize()
            if HAS_BOOSTEROS and self.slam:
                camera_info = self.robot.get_camera_info()
                self.slam.initialize(camera_info)
        elif HAS_BOOSTEROS and self.slam:
            self.slam.initialize()

        self.object_detector.initialize()
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
        self._goal = NavigationGoal(object_name=object_name)
        self._state = PolicyState.DETECTING
        self._object_not_found_count = 0
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

            elif self._state in (PolicyState.NAVIGATING, PolicyState.TRACKING_OBJECT):
                return self._step_navigating(sensor_data)

            elif self._state == PolicyState.ARRIVED:
                return self._update_status(
                    message=f"Arrived at '{self._goal.object_name}'"
                )

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

        elif self._state in (PolicyState.NAVIGATING, PolicyState.TRACKING_OBJECT):
            return self._step_sim_navigating(frame, pose)

        elif self._state == PolicyState.ARRIVED:
            return self._update_status(
                robot_pose=pose, message=f"Arrived at '{self._goal.object_name}'"
            )

        elif self._state == PolicyState.LOST:
            return self._update_status(
                robot_pose=pose, message="Lost - cannot find object"
            )

        elif self._state == PolicyState.ERROR:
            return self._update_status(robot_pose=pose, message="Error state")

        return self._update_status(robot_pose=pose)

    def _step_sim_detecting(self, frame: np.ndarray, pose: np.ndarray) -> PolicyStatus:
        """Step in DETECTING state for simulation."""
        detections = self.sim_env.get_detections()
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None:
            self._last_detection = target
            self._goal.detection = target
            self._object_not_found_count = 0

            if self._has_arrived(target.distance):
                self._state = PolicyState.ARRIVED
                return self._update_status(
                    robot_pose=pose,
                    detections=detections,
                    message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m",
                )

            self._state = PolicyState.NAVIGATING
            return self._update_status(
                robot_pose=pose,
                detections=detections,
                message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m, navigating...",
            )
        else:
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

    def _step_sim_navigating(self, frame: np.ndarray, pose: np.ndarray) -> PolicyStatus:
        """Step in NAVIGATING state for simulation."""
        detections = self.sim_env.get_detections()
        target = self.object_detector.find_object(detections, self._goal.object_name)

        if target is not None:
            self._last_detection = target
            self._goal.detection = target
            self._object_not_found_count = 0

            if self._has_arrived(target.distance):
                self._state = PolicyState.ARRIVED
                self._stop_robot()
                return self._update_status(
                    robot_pose=pose,
                    detections=detections,
                    message=f"Arrived at '{self._goal.object_name}'",
                )

            cmd = self._compute_tracking_velocity(target)
        else:
            self._object_not_found_count += 1
            if self._object_not_found_count >= self._max_object_not_found:
                self._state = PolicyState.LOST
                self._stop_robot()
                return self._update_status(
                    robot_pose=pose,
                    detections=detections,
                    message="Lost object during navigation",
                )
            if self._last_detection and self._last_detection.position_3d is not None:
                cmd = self._compute_tracking_velocity(self._last_detection)
            else:
                cmd = VelocityCommand()

        self.sim_env.set_velocity(cmd.linear_x, cmd.linear_y, cmd.angular_z)

        distance_text = f"{target.distance:.2f}m" if target is not None else "unknown"
        return self._update_status(
            robot_pose=pose,
            velocity_command=cmd,
            detections=detections,
            message=f"Navigating to '{self._goal.object_name}' ({distance_text})",
        )

    def _compute_tracking_velocity(self, detection: ObjectDetection) -> VelocityCommand:
        """Compute velocity command to track detected object."""
        k_linear = 0.5
        k_angular = 1.0

        error_distance = detection.distance - self._goal.target_distance
        error_angle = detection.angle

        linear_x = np.clip(k_linear * error_distance, -0.3, 0.3)
        angular_z = np.clip(-k_angular * error_angle, -1.0, 1.0)

        return VelocityCommand(linear_x=linear_x, linear_y=0.0, angular_z=angular_z)

    def _has_arrived(self, distance: float) -> bool:
        """Check the target distance with a small numerical/control tolerance."""
        return distance <= self._goal.target_distance + 0.01

    def _stop_robot(self) -> None:
        """Stop the robot."""
        if self._is_sim:
            self.sim_env.set_velocity(0.0, 0.0, 0.0)
        elif self.robot:
            self.robot.stop()

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
            if self._has_arrived(target.distance):
                self._state = PolicyState.ARRIVED
                return self._update_status(
                    detections=detections,
                    message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m",
                )

            # Start navigating
            self._state = PolicyState.NAVIGATING
            return self._update_status(
                detections=detections,
                message=f"Found '{self._goal.object_name}' at {target.distance:.2f}m, navigating...",
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

    def _step_navigating(self, sensor_data: dict) -> PolicyStatus:
        """Step in NAVIGATING state."""
        rgb = sensor_data["rgb"]
        depth = sensor_data["depth"]
        camera_info = sensor_data.get("camera_info")
        imu_rpy = sensor_data.get("imu_rpy")
        odom = sensor_data.get("odometry")

        # Update SLAM
        slam_pose = self.slam.track_frame(
            rgb,
            depth,
            imu_data=sensor_data.get("imu_samples"),
            timestamp=sensor_data.get("timestamp"),
        )

        # Update pose estimator
        body_slam_pose = self.slam.body_pose(slam_pose)
        fused_pose = self.pose_estimator.update(
            slam_pose=body_slam_pose,
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
                message=f"Safety violation: {safety_status.reason}",
            )

        # Process depth for obstacles
        depth_m = self.depth_processor.preprocess(depth)
        obstacle_info = self.depth_processor.detect_obstacles(depth_m)

        # Check navigation timeout
        if (
            self._start_time
            and (time.time() - self._start_time) > self._navigation_timeout
        ):
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
            if self._has_arrived(target.distance):
                self._state = PolicyState.ARRIVED
                return self._update_status(
                    detections=detections,
                    message=f"Arrived at '{self._goal.object_name}'",
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
                    detections=detections, message="Lost object during navigation"
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
            self.robot.set_velocity(cmd.linear_x, cmd.linear_y, cmd.angular_z)

        distance_text = f"{target.distance:.2f}" if target else "unknown"
        return self._update_status(
            current_pose=fused_pose,
            safety_status=safety_status,
            velocity_command=cmd,
            detections=detections,
            message=f"Navigating to '{self._goal.object_name}' ({distance_text}m)",
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
        if self.sim_env:
            self.sim_env.stop()

        self._update_status(message="Stopped")
        return self._status

    def _get_sensor_data(self) -> Optional[dict]:
        """Get latest sensor data from robot."""
        if self.robot is None:
            return None

        try:
            state = self.robot.get_state(include_images=True)

            return {
                "rgb": self.robot.image_to_array(state.rgb),
                "depth": self.robot.image_to_array(state.depth),
                "timestamp": self.robot.image_timestamp(state.rgb),
                "camera_info": state.camera_info,
                "imu_rpy": state.orientation_rpy,
                "imu_samples": getattr(state, "imu_samples", None),
                "odometry": state.position_2d,
            }
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
