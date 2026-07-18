"""Global initial localization against a fixed point-cloud-derived occupancy map.

A new ORB-SLAM session has an arbitrary origin, so map navigation needs the
robot's startup pose in the fixed map frame. This module estimates that pose
from depth observations: depth images are back-projected into gravity-aligned
planar obstacle scans, then matched against the occupancy grid with a
coarse-to-fine correlative search over the map's free space. Scoring rewards
scan points landing on persistent structure boundaries and penalizes scan
rays that would have to pass through mapped structure, and every estimate
carries a score and a rival-pose ambiguity so callers can keep gathering
viewpoints until the match is trustworthy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from nero.navigation.map_loader import OccupancyGrid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalLocalizationConfig:
    """Tuning for depth-scan extraction and correlative grid matching."""

    camera_height: float = 1.1
    min_height: float = 0.3
    max_height: float = 2.0
    max_range: float = 5.0
    pixel_stride: int = 4
    max_scan_points: int = 400
    min_scan_points: int = 50
    yaw_step: float = math.radians(10.0)
    yaw_separation: float = math.radians(30.0)
    position_stride: float = 0.25
    refine_iterations: int = 2
    sigma: float = 0.15
    min_structure_area: float = 0.05
    free_space_weight: float = 1.0
    ray_spacing: float = 0.3
    ray_margin: float = 0.3
    min_score: float = 0.5
    ambiguity_radius: float = 1.0
    max_ambiguity: float = 0.95


@dataclass(frozen=True)
class GlobalLocalizationResult:
    """Best map-frame pose for a scan plus how much to trust it."""

    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    score: float = 0.0
    ambiguity: float = 1.0
    num_points: int = 0
    min_score: float = 0.5
    max_ambiguity: float = 0.95

    @property
    def is_confident(self) -> bool:
        return self.score >= self.min_score and self.ambiguity <= self.max_ambiguity


def depth_to_planar_scan(
    depth_m: np.ndarray,
    camera_info: Any = None,
    imu_rpy: Optional[np.ndarray] = None,
    config: Optional[GlobalLocalizationConfig] = None,
) -> np.ndarray:
    """Project a metric depth image into a gravity-aligned 2D obstacle scan.

    Args:
        depth_m: Depth image in meters with invalid pixels as NaN
        camera_info: Object with a 3x3 ``k`` intrinsics matrix, or None
        imu_rpy: Body (roll, pitch, yaw); roll/pitch remove the camera tilt
        config: Height band, range, and subsampling settings

    Returns:
        (N, 2) obstacle points in the gravity-aligned body frame
        (x forward, y left), suitable for matching against the map grid.
    """
    config = config or GlobalLocalizationConfig()
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D depth image in meters")

    k = getattr(camera_info, "k", None)
    if k is not None:
        k = np.asarray(k, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    else:
        fx = fy = 216.5
        cx, cy = depth.shape[1] / 2.0, depth.shape[0] / 2.0

    stride = max(1, int(config.pixel_stride))
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us].ravel()
    us, vs = us.ravel(), vs.ravel()
    valid = np.isfinite(z) & (z > 0) & (z <= config.max_range)
    z, us, vs = z[valid], us[valid], vs[valid]
    if len(z) == 0:
        return np.empty((0, 2))

    # Optical frame (x right, y down, z forward) to body frame (x fwd, y left, z up)
    body = np.column_stack([z, -(us - cx) * z / fx, -(vs - cy) * z / fy])
    if imu_rpy is not None:
        roll, pitch = float(imu_rpy[0]), float(imu_rpy[1])
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        rotation = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
        ) @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        body = body @ rotation.T

    height = body[:, 2] + config.camera_height
    keep = (height >= config.min_height) & (height <= config.max_height)
    scan = body[keep, :2]
    if len(scan) > config.max_scan_points:
        indices = np.linspace(0, len(scan) - 1, config.max_scan_points).astype(int)
        scan = scan[indices]
    return scan


class GridLocalizer:
    """Correlative scan matcher over the free space of an occupancy grid.

    The raw grid is first reduced to persistent structure: occupied components
    smaller than ``min_structure_area`` are treated as reconstruction speckle
    and dropped, and matching targets the structure *boundaries*, since a depth
    camera observes surfaces rather than blob interiors. A pose is scored by
    the mean boundary likelihood at the scan points minus a penalty for scan
    rays that would have to pass through occupied structure. The coarse stage
    evaluates every grid cell for each candidate yaw with FFT cross-correlation
    and a direct coarse-to-fine search then refines the winning pose.
    """

    def __init__(
        self, grid: OccupancyGrid, config: Optional[GlobalLocalizationConfig] = None
    ):
        self.grid = grid
        self.config = config or GlobalLocalizationConfig()
        occupied = grid.data == 100
        if not occupied.any():
            raise ValueError("Occupancy grid has no occupied cells to localize against")
        labels, _ = ndimage.label(occupied)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        min_cells = max(
            1, int(round(self.config.min_structure_area / grid.resolution**2))
        )
        structure = (sizes >= min_cells)[labels]
        if not structure.any():
            raise ValueError(
                "Occupancy grid has no persistent structure to localize against"
            )
        boundary = structure & ~ndimage.binary_erosion(structure)
        distance = distance_transform_edt(~boundary) * grid.resolution
        self._hit_field = np.exp(-0.5 * (distance / self.config.sigma) ** 2)
        self._occupied_field = structure.astype(float)
        self._mask = (grid.data == 0) & ~structure & (distance <= self.config.max_range)
        if not self._mask.any():
            raise ValueError("Occupancy grid has no free cells near mapped structure")

    def _ray_samples(self, scan: np.ndarray) -> np.ndarray:
        """Free-space sample points along the sensor ray to each scan point."""
        spacing, margin = self.config.ray_spacing, self.config.ray_margin
        ranges = np.linalg.norm(scan, axis=1)
        samples = []
        for point, range_m in zip(scan, ranges):
            count = int((range_m - margin) / spacing)
            if count > 0:
                fractions = np.arange(1, count + 1) * spacing / range_m
                samples.append(point[None, :] * fractions[:, None])
        return np.concatenate(samples) if samples else np.empty((0, 2))

    def _lookup(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Vectorized field lookup; out-of-map points score zero."""
        grid = self.grid
        px = np.floor((points[..., 0] - grid.origin[0]) / grid.resolution).astype(int)
        py = (
            grid.height
            - 1
            - np.floor((points[..., 1] - grid.origin[1]) / grid.resolution).astype(int)
        )
        inside = (px >= 0) & (px < grid.width) & (py >= 0) & (py < grid.height)
        scores = np.zeros(points.shape[:-1])
        scores[inside] = field[py[inside], px[inside]]
        return scores

    def _score(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
        scan: np.ndarray,
        samples: np.ndarray,
    ) -> np.ndarray:
        """Return a (len(yaws), len(positions)) penalized score matrix."""
        scores = np.empty((len(yaws), len(positions)))
        for index, yaw in enumerate(yaws):
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, s], [-s, c]])
            points = positions[:, None, :] + (scan @ rotation)[None, :, :]
            scores[index] = self._lookup(self._hit_field, points).mean(axis=1)
            if len(samples):
                rays = positions[:, None, :] + (samples @ rotation)[None, :, :]
                scores[index] -= self.config.free_space_weight * self._lookup(
                    self._occupied_field, rays
                ).mean(axis=1)
        return scores

    def localize(self, scan_xy: np.ndarray) -> GlobalLocalizationResult:
        """Estimate the scan's body pose (x, y, yaw) in the map frame."""
        config = self.config
        scan = np.asarray(scan_xy, dtype=float)
        if scan.ndim != 2 or scan.shape[1] != 2:
            raise ValueError("scan_xy must have shape (N, 2)")
        if len(scan) < config.min_scan_points:
            return GlobalLocalizationResult(
                num_points=len(scan),
                min_score=config.min_score,
                max_ambiguity=config.max_ambiguity,
            )

        samples = self._ray_samples(scan)
        yaws = np.arange(0.0, 2.0 * math.pi, config.yaw_step)
        best_by_cell = np.full(self.grid.data.shape, -np.inf)
        for yaw in yaws:
            cell_scores = self._correlate(scan, samples, yaw)
            np.maximum(best_by_cell, cell_scores, out=best_by_cell)
        best_by_cell[~self._mask] = -np.inf
        best_index = np.unravel_index(np.argmax(best_by_cell), best_by_cell.shape)
        best_score = float(best_by_cell[best_index])
        best_position = np.array(self.grid.pixel_to_world(best_index[1], best_index[0]))

        yaw_scores = self._score(best_position[None, :], yaws, scan, samples)[:, 0]
        best_yaw = float(yaws[int(np.argmax(yaw_scores))])
        position, yaw, score = self._refine(
            best_position, best_yaw, best_score, scan, samples
        )
        ambiguity = self._ambiguity(
            best_by_cell, best_index, yaws, yaw_scores, score, scan, samples
        )
        return GlobalLocalizationResult(
            pose=np.array([position[0], position[1], _normalize_angle(yaw)]),
            score=score,
            ambiguity=ambiguity,
            num_points=len(scan),
            min_score=config.min_score,
            max_ambiguity=config.max_ambiguity,
        )

    def _correlate(
        self, scan: np.ndarray, samples: np.ndarray, yaw: float
    ) -> np.ndarray:
        """Penalized mean scan likelihood for every grid cell at one yaw, via FFT."""
        score = self._correlate_field(self._hit_field, scan, yaw)
        if len(samples):
            score = score - self.config.free_space_weight * self._correlate_field(
                self._occupied_field, samples, yaw
            )
        return score

    def _correlate_field(
        self, field: np.ndarray, points: np.ndarray, yaw: float
    ) -> np.ndarray:
        resolution = self.grid.resolution
        c, s = math.cos(yaw), math.sin(yaw)
        rotated = points @ np.array([[c, s], [-s, c]])
        half = int(math.ceil(self.config.max_range / resolution))
        dx = np.clip(np.round(rotated[:, 0] / resolution).astype(int), -half, half)
        dy = np.clip(np.round(-rotated[:, 1] / resolution).astype(int), -half, half)
        kernel = np.zeros((2 * half + 1, 2 * half + 1))
        np.add.at(kernel, (dy + half, dx + half), 1.0)
        correlation = fftconvolve(field, kernel[::-1, ::-1], mode="same")
        return correlation / len(points)

    def _ambiguity(
        self,
        best_by_cell: np.ndarray,
        best_index: tuple,
        yaws: np.ndarray,
        yaw_scores: np.ndarray,
        refined_best: float,
        scan: np.ndarray,
        samples: np.ndarray,
    ) -> float:
        """Ratio of the strongest refined rival pose to the winner (1.0 = tied).

        Rivals — the best coarse pose far from the winner in position, and the
        best far-in-yaw fit at the winning cell — are refined the same way as
        the winner so coarse quantization does not distort the comparison.
        """
        if refined_best <= 0.0:
            return 1.0
        config = self.config
        radius = config.ambiguity_radius / self.grid.resolution
        rows = np.arange(best_by_cell.shape[0])[:, None] - best_index[0]
        cols = np.arange(best_by_cell.shape[1])[None, :] - best_index[1]
        far = rows**2 + cols**2 > radius**2
        rivals = []
        if far.any():
            masked = np.where(far, best_by_cell, -np.inf)
            rival_index = np.unravel_index(np.argmax(masked), masked.shape)
            if np.isfinite(masked[rival_index]):
                rival_position = np.array(
                    self.grid.pixel_to_world(rival_index[1], rival_index[0])
                )
                rival_yaws = self._score(rival_position[None, :], yaws, scan, samples)[
                    :, 0
                ]
                rivals.append((rival_position, float(yaws[int(np.argmax(rival_yaws))])))
        best_yaw = float(yaws[int(np.argmax(yaw_scores))])
        far_yaw = np.abs(_normalize_angle(yaws - best_yaw)) > config.yaw_separation
        if far_yaw.any():
            rival_yaw_scores = np.where(far_yaw, yaw_scores, -np.inf)
            best_position = np.array(
                self.grid.pixel_to_world(best_index[1], best_index[0])
            )
            rivals.append(
                (best_position, float(yaws[int(np.argmax(rival_yaw_scores))]))
            )
        rival_score = 0.0
        for position, yaw in rivals:
            _, _, refined = self._refine(position, yaw, -np.inf, scan, samples)
            rival_score = max(rival_score, refined)
        return rival_score / refined_best

    def _refine(
        self,
        position: np.ndarray,
        yaw: float,
        score: float,
        scan: np.ndarray,
        samples: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        config = self.config
        position_step = config.position_stride
        yaw_step = config.yaw_step
        for _ in range(config.refine_iterations):
            fine_position = position_step / 4.0
            fine_yaw = yaw_step / 4.0
            steps = np.arange(
                -position_step, position_step + fine_position / 2, fine_position
            )
            xs, ys = np.meshgrid(position[0] + steps, position[1] + steps)
            positions = np.column_stack([xs.ravel(), ys.ravel()])
            yaws = yaw + np.arange(-yaw_step, yaw_step + fine_yaw / 2, fine_yaw)
            scores = self._score(positions, yaws, scan, samples)
            yaw_index, position_index = np.unravel_index(
                np.argmax(scores), scores.shape
            )
            if scores[yaw_index, position_index] > score:
                score = float(scores[yaw_index, position_index])
                position = positions[position_index]
                yaw = float(yaws[yaw_index])
            position_step, yaw_step = fine_position, fine_yaw
        return position, yaw, score


def localize_scan(
    grid: OccupancyGrid,
    scan_xy: np.ndarray,
    config: Optional[GlobalLocalizationConfig] = None,
) -> GlobalLocalizationResult:
    """One-shot convenience wrapper around :class:`GridLocalizer`."""
    return GridLocalizer(grid, config).localize(scan_xy)


def _normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
