"""Depth image processing utilities."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DepthProcessor:
    """Processes depth images for obstacle detection and navigation."""

    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 6.0,
        obstacle_threshold: float = 0.6,
        obstacle_region_height: int = 60,
    ):
        """Initialize depth processor.

        Args:
            min_depth: Minimum valid depth (m)
            max_depth: Maximum valid depth (m)
            obstacle_threshold: Distance below which something is an obstacle (m)
            obstacle_region_height: Height of the obstacle detection region from bottom of image
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_region_height = obstacle_region_height

    def preprocess(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to meters and filter invalid values.

        Args:
            depth: Raw depth image (uint16 in mm or float32 in m)

        Returns:
            Depth image in meters with invalid values set to NaN
        """
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32).copy()

        # The K1 Geek depth camera is specified only for 0.5-6 m. Its lower
        # image frequently contains self/floor artifacts around 0.2 m; those
        # out-of-range returns must not permanently inhibit walking.
        depth_m[
            (depth_m < self.min_depth)
            | (depth_m > self.max_depth)
            | ~np.isfinite(depth_m)
        ] = np.nan

        return depth_m

    def detect_obstacles(
        self,
        depth_m: np.ndarray,
        image_width: Optional[int] = None,
    ) -> dict:
        """Detect obstacles in the depth image.

        Analyzes the lower portion of the depth image (ground level)
        to detect obstacles in the robot's path.

        Args:
            depth_m: Preprocessed depth image in meters
            image_width: Optional width to use (defaults to full width)

        Returns:
            Dict with obstacle info:
                - has_obstacle: bool
                - min_distance: float (closest obstacle distance)
                - obstacle_mask: np.ndarray (binary mask of obstacles)
                - left_clear: bool (is left side clear?)
                - right_clear: bool (is right side clear?)
                - center_clear: bool (is center clear?)
        """
        h, w = depth_m.shape
        width = image_width or w

        # Look at lower portion of image (ground level)
        region = depth_m[h - self.obstacle_region_height :, :]

        # Create obstacle mask
        obstacle_mask = region < self.obstacle_threshold

        # Check for obstacles
        # Get minimum distance (ignoring NaN)
        valid_depth = region[~np.isnan(region)]
        sensor_blind = len(valid_depth) == 0
        min_distance = float(np.min(valid_depth)) if not sensor_blind else 0.0
        has_obstacle = bool(np.any(obstacle_mask) or sensor_blind)

        # Check left, center, right thirds
        third_w = width // 3
        left_region = obstacle_mask[:, :third_w]
        center_region = obstacle_mask[:, third_w : 2 * third_w]
        right_region = obstacle_mask[:, 2 * third_w :]

        return {
            "has_obstacle": has_obstacle,
            "min_distance": min_distance,
            "obstacle_mask": obstacle_mask,
            "sensor_blind": sensor_blind,
            "left_clear": not sensor_blind and not np.any(left_region),
            "center_clear": not sensor_blind and not np.any(center_region),
            "right_clear": not sensor_blind and not np.any(right_region),
        }

    def get_clear_path(
        self,
        depth_m: np.ndarray,
        corridor_width: float = 0.5,
    ) -> dict:
        """Determine if there's a clear path forward.

        Args:
            depth_m: Preprocessed depth image in meters
            corridor_width: Width of the corridor to check (m)

        Returns:
            Dict with path info:
                - is_clear: bool
                - max_distance: float (how far can we go)
                - preferred_direction: str ('center', 'left', 'right')
        """
        h, w = depth_m.shape

        # Convert corridor width to pixel width at various depths
        # Using camera intrinsics (approximate for K1)
        # Check at multiple depth slices
        slices = [h // 2, 2 * h // 3, 3 * h // 4]
        clear_distances = []

        for slice_y in slices:
            row = depth_m[slice_y, :]
            valid = ~np.isnan(row)
            if np.any(valid):
                clear_distances.append(float(np.min(row[valid])))
            else:
                clear_distances.append(self.max_depth)

        max_distance = min(clear_distances) if clear_distances else self.max_depth

        # Determine preferred direction
        third_w = w // 3
        center_depth = np.nanmedian(depth_m[h // 2 :, third_w : 2 * third_w])
        left_depth = np.nanmedian(depth_m[h // 2 :, :third_w])
        right_depth = np.nanmedian(depth_m[h // 2 :, 2 * third_w :])

        # Handle NaN values
        if np.isnan(center_depth):
            center_depth = self.max_depth
        if np.isnan(left_depth):
            left_depth = self.max_depth
        if np.isnan(right_depth):
            right_depth = self.max_depth

        depths = {"center": center_depth, "left": left_depth, "right": right_depth}
        preferred_direction = max(depths, key=depths.get)

        is_clear = max_distance > self.obstacle_threshold

        return {
            "is_clear": is_clear,
            "max_distance": max_distance,
            "preferred_direction": preferred_direction,
            "depths": depths,
        }

    def compute_ground_plane(
        self,
        depth_m: np.ndarray,
        camera_info=None,
    ) -> Optional[dict]:
        """Estimate ground plane from depth image.

        Uses RANSAC to fit a plane to the lower portion of the depth image.

        Args:
            depth_m: Preprocessed depth image in meters
            camera_info: Optional camera intrinsics

        Returns:
            Dict with plane parameters or None if fitting failed
        """
        h, w = depth_m.shape

        # Get 3D points from lower portion
        y_start = h // 2
        valid_mask = ~np.isnan(depth_m[y_start:, :])

        if np.sum(valid_mask) < 10:
            return None

        # Get camera intrinsics
        if camera_info is not None:
            fx = camera_info.k[0, 0]
            fy = camera_info.k[1, 1]
            cx = camera_info.k[0, 2]
            cy = camera_info.k[1, 2]
        else:
            fx = fy = 216.5
            cx = 160.0
            cy = 120.0

        # Back-project to 3D
        ys, xs = np.where(valid_mask)
        ys += y_start
        z = depth_m[ys, xs]

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        Z = z

        points = np.column_stack([X, Y, Z])

        # Simple plane fitting using SVD
        if len(points) < 3:
            return None

        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1, :]

        # Ensure normal points upward (positive Y in camera frame)
        if normal[1] < 0:
            normal = -normal

        return {
            "normal": normal,
            "centroid": centroid,
            "distance": float(np.dot(normal, centroid)),
        }
