"""Optional fixed-map layer for the unified navigation policy."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from nero.navigation.controller import VelocityCommand, VelocityController
from nero.navigation.global_localization import (
    GlobalLocalizationConfig,
    GridLocalizer,
    depth_to_planar_scan,
)
from nero.navigation.map_loader import OccupancyGrid, load_occupancy_grid
from nero.navigation.path_planner import Path, astar, follow_path, smooth_path

logger = logging.getLogger(__name__)

_LOCALIZE_YAW_BIN = math.radians(30.0)
_LOCALIZE_BIN_POINTS = 600


@dataclass
class MapNavConfig:
    """Configuration for optional planning in a map-aligned world frame."""

    map_path: str = ""
    yaml_path: Optional[str] = None
    resolution: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    auto_localize: bool = False
    localization: GlobalLocalizationConfig = field(
        default_factory=GlobalLocalizationConfig
    )
    localization_spin_speed: float = 0.3
    inflation_radius: float = 0.3
    path_smoothing: bool = True
    lookahead_distance: float = 0.5
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.5
    goal_threshold: float = 0.3
    goal_yaw_tolerance: float = 0.15
    replan_distance: float = 0.25


@dataclass(frozen=True)
class MapRouteResult:
    """One map-layer route decision consumed by ``NavigationPolicy``."""

    command: VelocityCommand = field(default_factory=VelocityCommand)
    arrived: bool = False
    failed: bool = False
    message: str = ""


class MapNavigator:
    """Fixed-map alignment and routing without its own sensor/control loop.

    The unified policy owns sensors, IMU-RGBD SLAM, safety, goals, and actuator
    output. This layer contributes only the map-to-SLAM transform and A* route.
    """

    def __init__(self, config: MapNavConfig):
        self.config = config
        self._grid: OccupancyGrid | None = None
        self._current_path: Path | None = None
        self._goal_world: np.ndarray | None = None
        self._current_pose = np.asarray(config.initial_pose, dtype=float)
        self._local_origin: np.ndarray | None = None
        self._anchor_map_pose: np.ndarray | None = None
        self._localizer: GridLocalizer | None = None
        self._scan_bins: dict[int, np.ndarray] = {}
        self._points_at_last_attempt = 0
        self._bins_at_last_attempt = 0
        self._steps_since_attempt = 0

    def load_map(self) -> None:
        self._grid = load_occupancy_grid(
            map_path=self.config.map_path,
            yaml_path=self.config.yaml_path,
            resolution=self.config.resolution,
            origin=self.config.origin,
        )
        logger.info(
            "Loaded map: %dx%d at %.3fm/px",
            self._grid.width,
            self._grid.height,
            self._grid.resolution,
        )

    def set_grid(self, grid: OccupancyGrid) -> None:
        """Inject a grid for tests or an already-loaded map provider."""
        self._grid = grid

    def validate_goal(self, pose: np.ndarray) -> bool:
        if self._grid is None:
            raise RuntimeError("fixed map is not loaded")
        return not self._grid.is_occupied(
            float(pose[0]),
            float(pose[1]),
            radius=self.config.inflation_radius,
        )

    def set_goal(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,):
            raise ValueError("map goal must have shape (3,)")
        changed = (
            self._goal_world is None
            or np.linalg.norm(pose[:2] - self._goal_world[:2])
            >= self.config.replan_distance
        )
        self._goal_world = pose.copy()
        if changed:
            self._current_path = None

    def clear_goal(self) -> None:
        self._goal_world = None
        self._current_path = None

    def update_alignment(self, localized, depth_processor) -> tuple[bool, str]:
        """Initialize the rigid SLAM-to-map transform from a trusted anchor."""
        if self.alignment_ready:
            return True, "Map alignment ready"
        local_pose = np.asarray(localized.fused_pose.position_2d, dtype=float)
        if not self.config.auto_localize:
            self._set_alignment(
                local_pose, np.asarray(self.config.initial_pose, dtype=float)
            )
            return True, "Anchored SLAM frame at the configured initial pose"

        self._accumulate_scan(localized, local_pose, depth_processor)
        self._steps_since_attempt += 1
        total_points = sum(len(points) for points in self._scan_bins.values())
        localization = self.config.localization
        if total_points < localization.min_scan_points:
            return False, f"Collecting depth scan points ({total_points} so far)"
        grown = (
            total_points >= self._points_at_last_attempt + localization.min_scan_points
        )
        new_view = len(self._scan_bins) > self._bins_at_last_attempt
        if not (grown or new_view or self._steps_since_attempt >= 100):
            return False, "Global localization pending; scanning for a new viewpoint"
        self._points_at_last_attempt = total_points
        self._bins_at_last_attempt = len(self._scan_bins)
        self._steps_since_attempt = 0

        if self._localizer is None:
            self._localizer = GridLocalizer(self._grid, localization)
        result = self._localizer.localize(self._composite_scan(local_pose))
        if not result.is_confident:
            return False, (
                "Global localization not confident yet "
                f"(score {result.score:.2f}, ambiguity {result.ambiguity:.2f}, "
                f"{result.num_points} points over {len(self._scan_bins)} viewpoints)"
            )
        self._set_alignment(local_pose, result.pose)
        logger.info(
            "Globally localized in map at (%.2f, %.2f, %.2f) score=%.2f ambiguity=%.2f",
            result.pose[0],
            result.pose[1],
            result.pose[2],
            result.score,
            result.ambiguity,
        )
        return True, "Globally localized in the fixed map"

    def route(
        self,
        current_pose: np.ndarray,
        goal_pose: np.ndarray,
        controller: VelocityController,
        obstacle_info: dict,
    ) -> MapRouteResult:
        """Plan or follow a global route while retaining live local avoidance."""
        self._current_pose = np.asarray(current_pose, dtype=float).copy()
        self.set_goal(goal_pose)
        if self._current_path is None:
            path = astar(
                self._grid,
                tuple(self._current_pose[:2]),
                tuple(self._goal_world[:2]),
                inflation_radius=self.config.inflation_radius,
            )
            if path is None:
                return MapRouteResult(
                    failed=True, message="No collision-free path to goal"
                )
            self._current_path = (
                smooth_path(path, self._grid) if self.config.path_smoothing else path
            )

        if controller.has_reached_pose(
            self._current_pose,
            self._goal_world,
            yaw_tolerance=self.config.goal_yaw_tolerance,
        ):
            return MapRouteResult(arrived=True, message="Arrived at map goal")

        waypoint = follow_path(
            self._current_path,
            tuple(self._current_pose[:2]),
            self.config.lookahead_distance,
        )
        waypoint_goal = np.array(
            [
                waypoint[0],
                waypoint[1],
                math.atan2(
                    waypoint[1] - self._current_pose[1],
                    waypoint[0] - self._current_pose[0],
                ),
            ]
        )
        if (
            np.linalg.norm(self._goal_world[:2] - self._current_pose[:2])
            <= self.config.lookahead_distance
        ):
            waypoint_goal = self._goal_world
        command = controller.compute_goal_velocity(
            self._current_pose,
            waypoint_goal,
            obstacle_info,
            yaw_tolerance=self.config.goal_yaw_tolerance,
        )
        return MapRouteResult(command=command, message="Following occupancy-grid path")

    def _accumulate_scan(self, localized, local_pose, depth_processor) -> None:
        scan = depth_to_planar_scan(
            depth_processor.preprocess(localized.sensor.depth),
            camera_info=localized.sensor.camera_info,
            imu_rpy=localized.sensor.imu_rpy,
            config=self.config.localization,
        )
        if not len(scan):
            return
        yaw = float(local_pose[2])
        c, s = math.cos(yaw), math.sin(yaw)
        session = scan @ np.array([[c, s], [-s, c]]) + local_pose[:2]
        bin_index = int((yaw % (2.0 * math.pi)) // _LOCALIZE_YAW_BIN)
        merged = np.concatenate(
            [self._scan_bins.get(bin_index, np.empty((0, 2))), session]
        )
        if len(merged) > _LOCALIZE_BIN_POINTS:
            keep = np.linspace(0, len(merged) - 1, _LOCALIZE_BIN_POINTS).astype(int)
            merged = merged[keep]
        self._scan_bins[bin_index] = merged

    def _composite_scan(self, local_pose: np.ndarray) -> np.ndarray:
        session = np.concatenate(list(self._scan_bins.values()))
        yaw = float(local_pose[2])
        c, s = math.cos(-yaw), math.sin(-yaw)
        body = (session - local_pose[:2]) @ np.array([[c, s], [-s, c]])
        limit = self.config.localization.max_scan_points
        if len(body) > limit:
            body = body[np.linspace(0, len(body) - 1, limit).astype(int)]
        return body

    def _set_alignment(self, local_pose: np.ndarray, map_pose: np.ndarray) -> None:
        self._local_origin = np.asarray(local_pose, dtype=float).copy()
        self._anchor_map_pose = np.asarray(map_pose, dtype=float).copy()

    def to_map_pose(self, local_pose: np.ndarray) -> np.ndarray:
        if not self.alignment_ready:
            raise RuntimeError("SLAM-to-map alignment is not initialized")
        local_pose = np.asarray(local_pose, dtype=float)
        transform = self.slam_to_map_transform
        position = transform[:2, :2] @ local_pose[:2] + transform[:2, 2]
        yaw_offset = math.atan2(transform[1, 0], transform[0, 0])
        return np.array(
            [
                position[0],
                position[1],
                self._normalize_angle(local_pose[2] + yaw_offset),
            ]
        )

    @property
    def slam_to_map_transform(self) -> np.ndarray:
        if not self.alignment_ready:
            raise RuntimeError("SLAM-to-map alignment is not initialized")
        x0, y0, yaw0 = self._anchor_map_pose
        yaw_offset = self._normalize_angle(yaw0 - self._local_origin[2])
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        rotation = np.array([[c, -s], [s, c]])
        transform = np.eye(3)
        transform[:2, :2] = rotation
        transform[:2, 2] = np.array([x0, y0]) - rotation @ self._local_origin[:2]
        return transform

    def transform_slam_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        transform = self.slam_to_map_transform
        transformed = points.copy()
        transformed[:, :2] = points[:, :2] @ transform[:2, :2].T + transform[:2, 2]
        return transformed

    def transform_slam_point(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).copy()
        transform = self.slam_to_map_transform
        point[:2] = transform[:2, :2] @ point[:2] + transform[:2, 2]
        return point

    def reset(self, *, clear_alignment: bool = False) -> None:
        self.clear_goal()
        if clear_alignment:
            self._local_origin = None
            self._anchor_map_pose = None
            self._localizer = None
            self._scan_bins = {}
            self._points_at_last_attempt = 0
            self._bins_at_last_attempt = 0
            self._steps_since_attempt = 0

    @property
    def alignment_ready(self) -> bool:
        return self._local_origin is not None and self._anchor_map_pose is not None

    @property
    def grid(self) -> OccupancyGrid | None:
        return self._grid

    @property
    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    def render_map(self) -> np.ndarray:
        if self._grid is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        image = np.zeros((self._grid.height, self._grid.width, 3), dtype=np.uint8)
        image[self._grid.data == 0] = [254, 254, 254]
        image[self._grid.data == 100] = [0, 0, 0]
        image[self._grid.data == -1] = [205, 205, 205]
        if self._current_path is not None:
            for first, second in zip(
                self._current_path.pixels, self._current_path.pixels[1:]
            ):
                cv2.line(image, first, second, (0, 255, 0), 2)
        px, py = self._grid.world_to_pixel(*self._current_pose[:2])
        if 0 <= px < self._grid.width and 0 <= py < self._grid.height:
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
        if self._goal_world is not None:
            gx, gy = self._grid.world_to_pixel(*self._goal_world[:2])
            if 0 <= gx < self._grid.width and 0 <= gy < self._grid.height:
                cv2.circle(image, (gx, gy), 8, (255, 0, 0), -1)
        return image

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
