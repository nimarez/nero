"""Mapping policy for autonomous space exploration and Gaussian splat reconstruction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from nero.robot import RobotInterface
from nero.robot import RobotState
from nero.slam.orb_slam3_node import ORBSLAM3Node
from nero.slam.pose_estimator import PoseEstimator
from nero.perception.depth_processor import DepthProcessor
from nero.mapping.gaussian_splat import GaussianSplatMapper, FrameData, MappingResult
from nero.mapping.trajectory_recorder import TrajectoryRecorder
from nero.navigation.controller import VelocityCommand

logger = logging.getLogger(__name__)


class MappingState(Enum):
    """Mapping policy states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    EXPLORING = "exploring"
    COLLECTING = "collecting"
    RETURNING = "returning"
    TRAINING = "training"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class MappingStatus:
    """Current mapping status."""
    state: MappingState
    message: str
    frames_collected: int
    max_frames: int
    trajectory_length: float
    training_progress: float
    elapsed_time: float
    velocity_command: Optional[VelocityCommand] = None


@dataclass
class MappingConfig:
    """Configuration for mapping mission."""
    max_frames: int = 500
    frame_skip: int = 5
    exploration_speed: float = 0.3
    min_obstacle_distance: float = 0.5
    area_size: float = 10.0  # Estimated area size in meters
    coverage_pattern: str = "spiral"  # spiral, boustrophedon, random
    output_dir: str = "output/splats"


class MappingPolicy:
    """Autonomous mapping policy for Gaussian splat reconstruction.

    The policy explores the space while collecting RGB-D frames with poses,
    then trains a 3D Gaussian Splat model.
    """

    def __init__(
        self,
        robot: Optional[RobotInterface],
        slam_config: Optional[dict] = None,
        mapping_config: Optional[MappingConfig] = None,
        safety_config: Optional[dict] = None,
    ):
        self._robot = robot
        self._config = mapping_config or MappingConfig()
        self._state = MappingState.IDLE
        self._message = "Ready to map"
        self._start_time: Optional[float] = None

        # Initialize SLAM
        self._slam = ORBSLAM3Node(
            voc_path=slam_config.get("voc_path", "config/ORBvoc.txt") if slam_config else "",
            settings_path=slam_config.get("settings_path", "config/orbslam3_settings.yaml") if slam_config else "",
        )
        self._pose_estimator = PoseEstimator()
        self._depth_processor = DepthProcessor()

        # Initialize mapping components
        self._splat_mapper = GaussianSplatMapper(
            output_dir=self._config.output_dir,
            max_frames=self._config.max_frames,
            frame_skip=self._config.frame_skip,
        )
        self._trajectory = TrajectoryRecorder()

        # Exploration state
        self._exploration_angle = 0.0
        self._exploration_radius = 0.0
        self._obstacle_detected = False
        self._last_pose: Optional[np.ndarray] = None

        # Safety
        self._safety_config = safety_config or {}
        self._min_obstacle_distance = self._safety_config.get("min_obstacle_distance", 0.5)

    def start(self) -> None:
        """Start mapping mission."""
        self._state = MappingState.INITIALIZING
        self._message = "Initializing SLAM..."
        self._start_time = time.time()

        # Initialize SLAM
        if self._robot:
            state = self._robot.get_state()
            if state.depth is not None:
                self._slam.initialize()

        self._trajectory.start()
        self._splat_mapper.start_collection()
        self._state = MappingState.EXPLORING
        self._message = "Starting exploration"
        logger.info("Mapping mission started")

    def step(self) -> MappingStatus:
        """Execute one step of the mapping policy.

        Returns:
            Current mapping status
        """
        if self._state == MappingState.IDLE:
            return self._status("Waiting to start")

        if self._state == MappingState.INITIALIZING:
            return self._status("Initializing...")

        if self._state == MappingState.TRAINING:
            return self._handle_training()

        if self._state == MappingState.COMPLETE:
            return self._status("Mapping complete")

        if self._state == MappingState.ERROR:
            return self._status(self._message)

        # Main exploration loop
        return self._handle_exploration()

    def stop(self) -> None:
        """Stop mapping mission."""
        self._splat_mapper.stop_collection()
        self._trajectory.stop()

        if self._robot:
            self._robot.set_velocity(0.0, 0.0, 0.0)

        self._state = MappingState.IDLE
        self._message = "Mapping stopped"
        logger.info("Mapping mission stopped")

    def start_training(self) -> None:
        """Manually trigger training."""
        if self._splat_mapper.get_frame_count() < 10:
            self._state = MappingState.ERROR
            self._message = "Not enough frames for training"
            return

        self._state = MappingState.TRAINING
        self._message = "Starting Gaussian Splat training..."
        logger.info("Starting training...")

    def get_result(self) -> Optional[MappingResult]:
        """Get mapping result after completion."""
        if self._state != MappingState.COMPLETE:
            return None
        return None  # Would store result after training

    def _handle_exploration(self) -> MappingStatus:
        """Handle exploration state."""
        # Get robot state
        robot_state = self._robot.get_state() if self._robot else None
        if robot_state is None:
            return self._status("Robot not connected")

        # Get current pose from SLAM
        current_pose = self._get_current_pose(robot_state)
        if current_pose is None:
            return self._status("SLAM not initialized")

        self._last_pose = current_pose
        self._trajectory.add_point(current_pose)

        # Capture frame for splatting
        if robot_state.rgb is not None:
            frame = FrameData(
                timestamp=time.time(),
                image=robot_state.rgb.data,
                depth=robot_state.depth.data if robot_state.depth is not None else None,
                pose=current_pose,
                frame_id=self._splat_mapper.get_frame_count(),
            )
            self._splat_mapper.add_frame(frame)

        # Check if we have enough frames
        if self._splat_mapper.get_frame_count() >= self._config.max_frames:
            self._state = MappingState.TRAINING
            self._message = "Max frames reached, starting training..."
            return self._status(self._message)

        # Check for obstacles
        if robot_state.depth is not None:
            self._obstacle_detected = self._depth_processor.check_obstacle(
                robot_state.depth.data,
                min_distance=self._min_obstacle_distance,
            )

        # Compute exploration velocity
        if self._obstacle_detected:
            # Avoid obstacle
            cmd = VelocityCommand(
                linear_x=0.0,
                angular_z=0.5,  # Turn away
            )
            self._message = "Avoiding obstacle"
        else:
            # Continue exploration pattern
            cmd = self._compute_exploration_velocity(current_pose)
            self._message = f"Exploring ({self._splat_mapper.get_frame_count()}/{self._config.max_frames} frames)"

        # Send velocity command
        if self._robot:
            self._robot.set_velocity(cmd.linear_x, cmd.linear_y, cmd.angular_z)

        return MappingStatus(
            state=self._state,
            message=self._message,
            frames_collected=self._splat_mapper.get_frame_count(),
            max_frames=self._config.max_frames,
            trajectory_length=self._trajectory.get_length(),
            training_progress=0.0,
            elapsed_time=time.time() - (self._start_time or time.time()),
            velocity_command=cmd,
        )

    def _handle_training(self) -> MappingStatus:
        """Handle training state."""
        if not self._splat_mapper.is_training():
            # Start training
            try:
                self._splat_mapper.train()
                self._state = MappingState.COMPLETE
                self._message = "Training complete!"
            except Exception as e:
                self._state = MappingState.ERROR
                self._message = f"Training failed: {e}"

        return MappingStatus(
            state=self._state,
            message=self._message,
            frames_collected=self._splat_mapper.get_frame_count(),
            max_frames=self._config.max_frames,
            trajectory_length=self._trajectory.get_length(),
            training_progress=self._splat_mapper.get_training_progress(),
            elapsed_time=time.time() - (self._start_time or time.time()),
        )

    def _compute_exploration_velocity(self, pose: np.ndarray) -> VelocityCommand:
        """Compute velocity for exploration pattern.

        Uses spiral pattern for coverage.
        """
        pattern = self._config.coverage_pattern

        if pattern == "spiral":
            return self._spiral_exploration(pose)
        elif pattern == "boustrophedon":
            return self._boustrophedon_exploration(pose)
        else:
            return self._random_exploration()

    def _spiral_exploration(self, pose: np.ndarray) -> VelocityCommand:
        """Spiral exploration pattern."""
        speed = self._config.exploration_speed

        # Increase radius over time
        self._exploration_radius += 0.001
        self._exploration_angle += 0.02

        # Spiral: move forward while turning
        linear = speed * 0.5
        angular = 0.3 + 0.1 * np.sin(self._exploration_angle)

        return VelocityCommand(linear_x=linear, angular_z=angular)

    def _boustrophedon_exploration(self, pose: np.ndarray) -> VelocityCommand:
        """Boustrophedon (lawnmower) pattern."""
        speed = self._config.exploration_speed
        area = self._config.area_size

        # Simple back-and-forth pattern
        x = pose[0, 3]
        y = pose[1, 3]

        # Move forward, turn at edges
        if abs(y) > area / 2:
            return VelocityCommand(linear_x=0.0, angular_z=0.5)

        return VelocityCommand(linear_x=speed, angular_z=0.0)

    def _random_exploration(self) -> VelocityCommand:
        """Random exploration with obstacle avoidance."""
        speed = self._config.exploration_speed

        if self._obstacle_detected:
            return VelocityCommand(linear_x=0.0, angular_z=0.5)

        # Random walk
        if np.random.random() < 0.05:
            self._exploration_angle = np.random.uniform(-0.3, 0.3)

        return VelocityCommand(
            linear_x=speed,
            angular_z=self._exploration_angle,
        )

    def _get_current_pose(self, robot_state: RobotState) -> Optional[np.ndarray]:
        """Get current robot pose from SLAM."""
        if robot_state.rgb is None:
            return self._last_pose

        # Try SLAM pose first
        slam_pose = self._slam.get_current_pose()
        if slam_pose is not None:
            return slam_pose

        # Fallback to odometry
        if robot_state.odom is not None:
            x, y, yaw = robot_state.odom.pose_2d
            pose = np.eye(4)
            pose[0, 3] = x
            pose[1, 3] = y
            pose[:2, :2] = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ])
            return pose

        return self._last_pose

    def _status(self, message: str) -> MappingStatus:
        """Create status with default values."""
        return MappingStatus(
            state=self._state,
            message=message,
            frames_collected=self._splat_mapper.get_frame_count(),
            max_frames=self._config.max_frames,
            trajectory_length=self._trajectory.get_length(),
            training_progress=0.0,
            elapsed_time=time.time() - (self._start_time or time.time()),
        )