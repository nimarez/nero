import math
from types import SimpleNamespace

import numpy as np
import pytest

from nero.navigation.global_localization import (
    GlobalLocalizationConfig,
    GlobalLocalizationResult,
    GridLocalizer,
    depth_to_planar_scan,
    localize_scan,
)
from nero.navigation.map_loader import OccupancyGrid


def room_grid(size_m=8.0, resolution=0.1):
    """Walled room with an interior L-wall so the layout has a unique fit."""
    cells = int(size_m / resolution)
    data = np.zeros((cells, cells), dtype=np.int8)
    data[0, :] = 100
    data[-1, :] = 100
    data[:, 0] = 100
    data[:, -1] = 100
    grid = OccupancyGrid(
        data=data, resolution=resolution, origin=(0.0, 0.0), width=cells, height=cells
    )
    for x in np.arange(2.0, 5.0, resolution):
        px, py = grid.world_to_pixel(x, 5.5)
        data[py, px] = 100
    for y in np.arange(4.0, 5.5, resolution):
        px, py = grid.world_to_pixel(2.0, y)
        data[py, px] = 100
    return grid


def simulate_scan(grid, pose, max_range=5.0, fov=math.radians(110.0), n_rays=200):
    """Ray-cast a depth-camera-like scan, expressed in the body frame."""
    points = []
    for bearing in np.linspace(-fov / 2, fov / 2, n_rays):
        angle = pose[2] + bearing
        dx, dy = math.cos(angle), math.sin(angle)
        r = grid.resolution
        while r <= max_range:
            wx, wy = pose[0] + r * dx, pose[1] + r * dy
            px, py = grid.world_to_pixel(wx, wy)
            if px < 0 or px >= grid.width or py < 0 or py >= grid.height:
                break
            if grid.data[py, px] == 100:
                c, s = math.cos(-pose[2]), math.sin(-pose[2])
                points.append(
                    (
                        c * (wx - pose[0]) - s * (wy - pose[1]),
                        s * (wx - pose[0]) + c * (wy - pose[1]),
                    )
                )
                break
            r += grid.resolution
    return np.array(points) if points else np.empty((0, 2))


def test_localize_scan_recovers_known_pose():
    grid = room_grid()
    true_pose = (5.0, 6.5, math.pi)
    scan = simulate_scan(grid, true_pose)
    result = localize_scan(grid, scan)
    assert result.is_confident
    np.testing.assert_allclose(result.pose[:2], true_pose[:2], atol=0.15)
    yaw_error = math.atan2(
        math.sin(result.pose[2] - true_pose[2]), math.cos(result.pose[2] - true_pose[2])
    )
    assert abs(yaw_error) < 0.1


def test_localize_scan_recovers_pose_facing_backwards():
    grid = room_grid()
    true_pose = (6.0, 6.0, math.radians(-135.0))
    scan = simulate_scan(grid, true_pose, fov=math.radians(160.0))
    result = localize_scan(grid, scan)
    assert result.is_confident
    np.testing.assert_allclose(result.pose[:2], true_pose[:2], atol=0.2)


def test_symmetric_view_is_flagged_ambiguous():
    resolution = 0.1
    cells = 60
    data = np.zeros((cells, cells), dtype=np.int8)
    data[0, :] = 100
    data[-1, :] = 100
    data[:, 0] = 100
    data[:, -1] = 100
    grid = OccupancyGrid(
        data=data, resolution=resolution, origin=(0.0, 0.0), width=cells, height=cells
    )
    scan = simulate_scan(grid, (3.0, 3.0, 0.0), max_range=4.0, fov=math.radians(90.0))
    result = localize_scan(grid, scan)
    assert result.ambiguity > 0.9
    assert not result.is_confident


def test_too_few_points_is_not_confident():
    grid = room_grid()
    result = localize_scan(grid, np.zeros((3, 2)))
    assert result.num_points == 3
    assert result.score == 0.0
    assert not result.is_confident


def test_localizer_rejects_degenerate_grids():
    empty = OccupancyGrid(
        data=np.zeros((4, 4), dtype=np.int8),
        resolution=0.1,
        origin=(0, 0),
        width=4,
        height=4,
    )
    with pytest.raises(ValueError):
        GridLocalizer(empty)
    full = OccupancyGrid(
        data=np.full((4, 4), 100, dtype=np.int8),
        resolution=0.1,
        origin=(0, 0),
        width=4,
        height=4,
    )
    with pytest.raises(ValueError):
        GridLocalizer(full)


def test_depth_to_planar_scan_projects_wall_in_height_band():
    depth = np.full((40, 60), 2.0)
    camera_info = SimpleNamespace(
        k=np.array([[100.0, 0.0, 30.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])
    )
    config = GlobalLocalizationConfig(camera_height=1.0, min_height=0.5, max_height=2.0)
    scan = depth_to_planar_scan(
        depth, camera_info=camera_info, imu_rpy=np.zeros(3), config=config
    )
    assert len(scan) > 0
    np.testing.assert_allclose(scan[:, 0], 2.0, atol=1e-6)
    # Height band keeps pixels whose back-projected height is 0.5..2.0m
    assert np.all(np.abs(scan[:, 1]) <= 2.0 * 30.0 / 100.0 + 1e-6)


def test_depth_to_planar_scan_pitch_correction():
    depth = np.full((41, 61), 3.0)
    camera_info = SimpleNamespace(
        k=np.array([[100.0, 0.0, 30.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])
    )
    config = GlobalLocalizationConfig(camera_height=1.0, min_height=0.9, max_height=1.1)
    flat = depth_to_planar_scan(
        depth, camera_info=camera_info, imu_rpy=np.zeros(3), config=config
    )
    pitched = depth_to_planar_scan(
        depth, camera_info=camera_info, imu_rpy=np.array([0.0, 0.2, 0.0]), config=config
    )
    # A pitched camera still yields forward-range points near the same distance
    assert len(pitched) > 0
    assert abs(np.median(pitched[:, 0]) - np.median(flat[:, 0])) < 0.2


def test_depth_to_planar_scan_handles_empty_and_invalid_input():
    assert depth_to_planar_scan(np.full((10, 10), np.nan)).shape == (0, 2)
    with pytest.raises(ValueError):
        depth_to_planar_scan(np.zeros(5))


def test_result_confidence_thresholds():
    confident = GlobalLocalizationResult(score=0.8, ambiguity=0.5, num_points=100)
    assert confident.is_confident
    assert not GlobalLocalizationResult(score=0.2, ambiguity=0.5).is_confident
    assert not GlobalLocalizationResult(score=0.8, ambiguity=0.97).is_confident
