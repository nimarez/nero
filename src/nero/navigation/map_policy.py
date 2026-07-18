"""Static-map navigation on the shared IMU-RGBD localization runtime."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from nero.navigation.controller import VelocityCommand, VelocityController
from nero.navigation.map_loader import OccupancyGrid, load_occupancy_grid
from nero.navigation.path_planner import Path, astar, follow_path, smooth_path
from nero.navigation.runtime import (
    SensorFrame,
    initialize_sensor_navigation,
    localize_sensor_frame,
    read_sensor_frame,
    send_velocity,
)
from nero.navigation.safety import SafetyMonitor
from nero.perception.depth_processor import DepthProcessor
from nero.slam.orb_slam3_node import ORBSLAM3Node
from nero.slam.pose_estimator import FusedPose, PoseEstimator

logger = logging.getLogger(__name__)


class MapNavState(Enum):
    IDLE = "idle"
    LOCALIZING = "localizing"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    LOST = "lost"
    ERROR = "error"


@dataclass
class MapNavConfig:
    """Configuration for planning in a map-aligned world frame."""

    map_path: str = ""
    yaml_path: Optional[str] = None
    resolution: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inflation_radius: float = 0.3
    path_smoothing: bool = True
    lookahead_distance: float = 0.5
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.5
    goal_threshold: float = 0.3
    goal_yaw_tolerance: float = 0.15


@dataclass
class MapPolicyStatus:
    state: MapNavState = MapNavState.IDLE
    pose: Optional[np.ndarray] = None
    goal_pose: Optional[np.ndarray] = None
    fused_pose: Optional[FusedPose] = None
    velocity_command: VelocityCommand = field(default_factory=VelocityCommand)
    safety_status: object | None = None
    message: str = ""


class MapNavigationPolicy:
    """A* map policy using the same K1 SLAM, safety, and controller as object nav.

    ORB-SLAM's session frame is aligned to the occupancy map by ``initial_pose``:
    the first localized body pose is treated as that map-frame pose.
    """

    def __init__(
        self,
        config: Optional[MapNavConfig] = None,
        *,
        robot=None,
        slam_config=None,
        slam_options=None,
        safety_config=None,
    ):
        self._config = config or MapNavConfig()
        self.robot = robot
        self.slam = ORBSLAM3Node(config=slam_config, **dict(slam_options or {}))
        self.pose_estimator = PoseEstimator()
        self.depth_processor = DepthProcessor()
        self.safety = SafetyMonitor(**(safety_config or {}))
        self.controller = VelocityController(
            max_linear_velocity=self._config.max_linear_vel,
            max_angular_velocity=self._config.max_angular_vel,
            goal_threshold=self._config.goal_threshold,
        )

        self._state = MapNavState.IDLE
        self._grid: Optional[OccupancyGrid] = None
        self._current_path: Optional[Path] = None
        self._goal_world: Optional[tuple[float, float, float]] = None
        self._current_pose = np.asarray(self._config.initial_pose, dtype=float)
        self._local_origin: Optional[np.ndarray] = None
        self._last_sensor: Optional[SensorFrame] = None
        self._status = MapPolicyStatus()
        self._running = False

    def load_map(self) -> bool:
        try:
            self._grid = load_occupancy_grid(
                map_path=self._config.map_path,
                yaml_path=self._config.yaml_path,
                resolution=self._config.resolution,
                origin=self._config.origin,
            )
            logger.info(
                "Loaded map: %dx%d at %.3fm/px",
                self._grid.width,
                self._grid.height,
                self._grid.resolution,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load map: %s", exc)
            return False

    def start(self) -> MapPolicyStatus:
        if self.robot is None:
            raise RuntimeError("Map navigation requires a robot environment adapter")
        if self._grid is None and not self.load_map():
            raise RuntimeError(f"Could not load occupancy map {self._config.map_path!r}")
        initialize_sensor_navigation(self.robot, self.slam, self.pose_estimator, self.safety)
        self._running = True
        self._state = MapNavState.LOCALIZING
        return self._update_status(message="Waiting for an IMU-RGBD SLAM pose")

    def set_goal(self, x: float, y: float, yaw: float = 0.0) -> bool:
        if self._grid is None:
            logger.error("No map loaded")
            return False
        if self._grid.is_occupied(x, y, radius=self._config.inflation_radius):
            logger.warning("Goal (%.2f, %.2f) is occupied", x, y)
            return False
        self._goal_world = (float(x), float(y), float(yaw))
        self._current_path = None
        self._state = (
            MapNavState.PLANNING if self._local_origin is not None else MapNavState.LOCALIZING
        )
        return True

    def set_goal_from_pixel(self, px: int, py: int, yaw: float | None = None) -> bool:
        if self._grid is None:
            return False
        x, y = self._grid.pixel_to_world(px, py)
        return self.set_goal(x, y, self._current_pose[2] if yaw is None else yaw)

    def step(self) -> MapPolicyStatus:
        if not self._running:
            return self._update_status(message="Policy not running")
        try:
            sensor = read_sensor_frame(self.robot)
            self._last_sensor = sensor
            localized = localize_sensor_frame(
                sensor,
                slam=self.slam,
                pose_estimator=self.pose_estimator,
                depth_processor=self.depth_processor,
                safety=self.safety,
            )
        except Exception as exc:
            logger.exception("Map navigation sensor/localization failure")
            return self._fail(MapNavState.ERROR, f"Localization failed: {exc}")

        if not localized.safety_status.is_safe:
            return self._fail(
                MapNavState.ERROR,
                f"Safety violation: {localized.safety_status.reason}",
                fused_pose=localized.fused_pose,
                safety_status=localized.safety_status,
            )
        if localized.slam_pose.tracking_status != "OK":
            return self._fail(
                MapNavState.LOST,
                "ORB-SLAM tracking lost",
                fused_pose=localized.fused_pose,
                safety_status=localized.safety_status,
            )

        self._current_pose = self._to_map_pose(localized.fused_pose.position_2d)
        if self._goal_world is None:
            self._state = MapNavState.IDLE
            send_velocity(self.robot)
            return self._update_status(
                fused_pose=localized.fused_pose,
                safety_status=localized.safety_status,
                message="Localized; waiting for a map-frame goal",
            )

        if self._current_path is None:
            self._state = MapNavState.PLANNING
            path = astar(
                self._grid,
                tuple(self._current_pose[:2]),
                tuple(self._goal_world[:2]),
                inflation_radius=self._config.inflation_radius,
            )
            if path is None:
                return self._fail(MapNavState.LOST, "No collision-free path to goal")
            self._current_path = (
                smooth_path(path, self._grid) if self._config.path_smoothing else path
            )

        goal = np.asarray(self._goal_world, dtype=float)
        if self.controller.has_reached_pose(
            self._current_pose, goal, yaw_tolerance=self._config.goal_yaw_tolerance
        ):
            self._state = MapNavState.ARRIVED
            send_velocity(self.robot)
            return self._update_status(
                fused_pose=localized.fused_pose,
                safety_status=localized.safety_status,
                message="Arrived at map goal",
            )

        waypoint = follow_path(
            self._current_path,
            tuple(self._current_pose[:2]),
            self._config.lookahead_distance,
        )
        waypoint_goal = np.array(
            [
                waypoint[0],
                waypoint[1],
                math.atan2(
                    waypoint[1] - self._current_pose[1], waypoint[0] - self._current_pose[0]
                ),
            ]
        )
        if np.linalg.norm(goal[:2] - self._current_pose[:2]) <= self._config.lookahead_distance:
            waypoint_goal = goal
        command = self.controller.compute_goal_velocity(
            self._current_pose,
            waypoint_goal,
            localized.obstacle_info,
            yaw_tolerance=self._config.goal_yaw_tolerance,
        )
        self._state = MapNavState.NAVIGATING
        send_velocity(self.robot, command)
        return self._update_status(
            command=command,
            fused_pose=localized.fused_pose,
            safety_status=localized.safety_status,
            message="Following occupancy-grid path",
        )

    def _to_map_pose(self, local_pose: np.ndarray) -> np.ndarray:
        local_pose = np.asarray(local_pose, dtype=float)
        if self._local_origin is None:
            self._local_origin = local_pose.copy()
        dx, dy = local_pose[:2] - self._local_origin[:2]
        delta_yaw = self._normalize_angle(local_pose[2] - self._local_origin[2])
        x0, y0, yaw0 = self._config.initial_pose
        c, s = math.cos(yaw0), math.sin(yaw0)
        return np.array(
            [x0 + c * dx - s * dy, y0 + s * dx + c * dy, self._normalize_angle(yaw0 + delta_yaw)]
        )

    def _fail(self, state: MapNavState, message: str, **kwargs) -> MapPolicyStatus:
        self._state = state
        if self.robot is not None:
            send_velocity(self.robot)
        return self._update_status(message=message, **kwargs)

    def _update_status(
        self,
        *,
        command: Optional[VelocityCommand] = None,
        fused_pose: Optional[FusedPose] = None,
        safety_status=None,
        message: str = "",
    ) -> MapPolicyStatus:
        self._status = MapPolicyStatus(
            state=self._state,
            pose=self._current_pose.copy(),
            goal_pose=(
                np.asarray(self._goal_world, dtype=float) if self._goal_world is not None else None
            ),
            fused_pose=fused_pose,
            velocity_command=command or VelocityCommand(),
            safety_status=safety_status,
            message=message,
        )
        return self._status

    def reset(self) -> None:
        self._goal_world = None
        self._current_path = None
        self._local_origin = None
        self._current_pose = np.asarray(self._config.initial_pose, dtype=float)
        self.pose_estimator.reset()
        self.safety.reset()
        self._state = MapNavState.LOCALIZING if self._running else MapNavState.IDLE
        if self.robot is not None:
            send_velocity(self.robot)

    def stop(self) -> None:
        self._running = False
        self._state = MapNavState.IDLE
        if self.robot is not None:
            send_velocity(self.robot)
        self.slam.shutdown()

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @property
    def state(self) -> MapNavState:
        return self._state

    @property
    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    @property
    def grid(self) -> Optional[OccupancyGrid]:
        return self._grid

    @property
    def last_sensor(self) -> Optional[SensorFrame]:
        return self._last_sensor

    def render_map(self) -> np.ndarray:
        if self._grid is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        image = np.zeros((self._grid.height, self._grid.width, 3), dtype=np.uint8)
        image[self._grid.data == 0] = [254, 254, 254]
        image[self._grid.data == 100] = [0, 0, 0]
        image[self._grid.data == -1] = [205, 205, 205]
        if self._current_path is not None:
            for first, second in zip(self._current_path.pixels, self._current_path.pixels[1:]):
                cv2.line(image, first, second, (0, 255, 0), 2)
        px, py = self._grid.world_to_pixel(*self._current_pose[:2])
        if 0 <= px < self._grid.width and 0 <= py < self._grid.height:
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
        if self._goal_world is not None:
            gx, gy = self._grid.world_to_pixel(*self._goal_world[:2])
            if 0 <= gx < self._grid.width and 0 <= gy < self._grid.height:
                cv2.circle(image, (gx, gy), 8, (255, 0, 0), -1)
        return image
