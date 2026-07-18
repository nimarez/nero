"""Simulation environment that ties together robot and camera.

Provides a unified interface for running the agent/policy loop
in simulation mode.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from nero.simulation.mock_robot import MockRobot
from nero.simulation.sim_camera import SimCamera, CameraMode
from nero.perception.object_detector import ObjectDetection

logger = logging.getLogger(__name__)


class SimEnvironment:
    """Complete simulation environment for testing agents.

    Combines MockRobot and SimCamera into a single interface
    that can be used as a drop-in replacement for real hardware.
    """

    def __init__(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw: float = 0.0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        camera_mode: CameraMode = CameraMode.TOP_DOWN,
    ):
        self.robot = MockRobot(
            initial_x=robot_x,
            initial_y=robot_y,
            initial_yaw=robot_yaw,
        )
        self.camera = SimCamera(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            mode=camera_mode,
        )
        self._running = False

    def initialize(self) -> None:
        """Initialize the simulation environment."""
        self.robot.initialize()
        self.camera.start()
        self._running = True
        logger.info("Simulation environment initialized")

    def stop(self) -> None:
        """Stop the simulation environment."""
        self._running = False
        self.robot.stop()
        self.camera.stop()
        logger.info("Simulation environment stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame with robot position."""
        pose = self.robot.get_pose()
        return self.camera.get_frame(pose[0], pose[1], pose[2])

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get current depth frame with robot position."""
        pose = self.robot.get_pose()
        return self.camera.get_depth_frame(pose[0], pose[1], pose[2])

    def get_pose(self) -> np.ndarray:
        """Get current robot pose."""
        return self.robot.get_pose()

    def get_detections(self) -> list[ObjectDetection]:
        """Return exact synthetic detections for objects in the scene."""
        robot_x, robot_y, robot_yaw = self.robot.get_pose()
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        detections = []

        for name, (object_x, object_y) in self.camera.get_objects().items():
            dx = object_x - robot_x
            dy = object_y - robot_y
            forward = dx * cos_yaw + dy * sin_yaw
            lateral = -dx * sin_yaw + dy * cos_yaw
            distance = math.hypot(forward, lateral)
            if forward <= 0:
                continue

            detections.append(
                ObjectDetection(
                    label=name,
                    confidence=1.0,
                    bbox=(0, 0, 1, 1),
                    position_3d=np.array([lateral, 0.0, forward], dtype=float),
                    distance=distance,
                )
            )

        return detections

    def set_velocity(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set robot velocity."""
        self.robot.set_velocity(vx, vy, vyaw)

    def add_object(self, name: str, x: float, y: float) -> None:
        """Add an object to the environment."""
        self.camera.add_object(name, x, y)

    def add_obstacle(self, x: float, y: float) -> None:
        """Add an obstacle to the environment."""
        self.camera.add_obstacle(x, y)
        self.robot.add_obstacle(x, y)

    def clear_environment(self) -> None:
        """Clear all objects and obstacles."""
        self.camera.clear_objects()
        self.camera.clear_obstacles()
        self.robot.clear_obstacles()

    def reset_robot(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Reset robot to initial position."""
        self.robot.reset(x, y, yaw)

    def setup_demo_scene(self) -> None:
        """Set up a demo scene with some objects and obstacles.

        Creates a simple room with:
        - A chair at (2, 0)
        - A table at (0, 3)
        - A bottle at (3, 2)
        - Some obstacles
        """
        self.clear_environment()

        # Add objects
        self.add_object("chair", 2.0, 0.0)
        self.add_object("table", 0.0, 3.0)
        self.add_object("bottle", 3.0, 2.0)
        self.add_object("lamp", -2.0, 1.0)

        # Add obstacles
        self.add_obstacle(1.0, 1.0)
        self.add_obstacle(-1.0, 2.0)
        self.add_obstacle(2.0, -1.0)

        logger.info("Demo scene set up with chair, table, bottle, lamp, and obstacles")
