import time

import numpy as np
import pytest
from PIL import Image

from nero.navigation.map_loader import (
    OccupancyGrid,
    load_occupancy_grid,
    pointcloud_to_grid,
    save_grid_as_png,
    save_grid_yaml,
)
from nero.navigation.path_planner import (
    Path,
    astar,
    follow_path,
    get_neighbors,
    line_of_sight,
    smooth_path,
)
from nero.navigation.safety import SafetyMonitor
from nero.perception.depth_processor import DepthProcessor
from nero.slam.orb_slam3_node import SLAMPose
from nero.slam.pose_estimator import PoseEstimator


def free_grid(size=5, resolution=1.0):
    return OccupancyGrid(
        data=np.zeros((size, size), dtype=np.int8),
        resolution=resolution,
        origin=(0.0, 0.0),
        width=size,
        height=size,
    )


def test_occupancy_grid_coordinates_costs_and_radius():
    grid = free_grid()
    grid.data[3, 1] = 100
    assert grid.world_to_pixel(1, 1) == (1, 3)
    assert grid.pixel_to_world(1, 3) == (1, 1)
    assert grid.is_occupied(1, 1)
    assert grid.is_occupied(0, 1, radius=1.0)
    assert grid.is_occupied(-1, 0)
    assert grid.get_cost(-1, 0) == float("inf")
    grid.data[2, 2] = -1
    assert grid.get_cost(2, 2) == 50


def test_ros_png_thresholds_and_round_trip_metadata(tmp_path):
    image_path = tmp_path / "map.png"
    yaml_path = tmp_path / "map.yaml"
    Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(image_path)
    yaml_path.write_text(
        "resolution: 0.1\norigin: [-1.0, -2.0, 0.0]\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.25\nnegate: 0\n"
    )
    grid = load_occupancy_grid(image_path, yaml_path)
    np.testing.assert_array_equal(grid.data, [[100, -1, 0]])
    assert grid.origin == (-1.0, -2.0)
    assert grid.resolution == 0.1

    saved_png = tmp_path / "saved.png"
    saved_yaml = tmp_path / "saved.yaml"
    save_grid_as_png(grid, saved_png)
    save_grid_yaml(grid, saved_yaml)
    assert saved_png.is_file()
    assert "resolution: 0.1" in saved_yaml.read_text()


def test_numpy_grid_loading_and_pointcloud_projection(tmp_path):
    data = np.array([[0, 100], [-1, 0]], dtype=np.int8)
    path = tmp_path / "map.npy"
    np.save(path, data)
    loaded = load_occupancy_grid(path, resolution=0.5, origin=(1, 2))
    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.width == loaded.height == 2

    points = np.array([[0, 0, 0.1], [1, 1, 0.8], [2, 2, 1.0]], dtype=float)
    projected = pointcloud_to_grid(
        points, resolution=1.0, grid_size=3.0, origin=(0, 0), height_threshold=0.5
    )
    assert projected.data[1, 1] == 100
    assert projected.data[0, 2] == 100
    assert projected.data[2, 0] == 0


def test_neighbor_lookup_uses_image_coordinates_without_double_flip():
    grid = free_grid()
    grid.data[3, 2] = 100
    neighbors = get_neighbors(1, 3, grid, allow_diagonal=False)
    assert (2, 3, 1.0) not in neighbors


def test_astar_smoothing_visibility_and_following():
    grid = free_grid(size=6)
    path = astar(grid, (0, 0), (5, 5))
    assert path is not None
    assert path.waypoints[0] == (0.0, 0.0)
    assert path.waypoints[-1] == (5.0, 5.0)
    assert line_of_sight(grid, (0, 0), (5, 5))
    smoothed = smooth_path(path, grid)
    assert smoothed.waypoints == [(0.0, 0.0), (5.0, 5.0)]
    assert follow_path(smoothed, (0, 0), lookahead_distance=2) == (5.0, 5.0)
    assert follow_path(Path([], [], 0), (2, 3)) == (2, 3)

    grid.data[:, 2] = 100
    assert astar(grid, (0, 0), (5, 0), allow_diagonal=False) is None
    assert not line_of_sight(grid, (0, 0), (5, 0))


def test_depth_preprocessing_obstacles_clear_path_and_ground_plane():
    processor = DepthProcessor(obstacle_region_height=2)
    raw = np.array(
        [[100, 1000, 6000], [1000, 1000, 1000], [1000, 300, 1000]],
        dtype=np.uint16,
    )
    depth = processor.preprocess(raw)
    assert np.isnan(depth[0, 0]) and np.isnan(depth[0, 2])
    obstacles = processor.detect_obstacles(depth)
    assert obstacles["has_obstacle"]
    assert obstacles["min_distance"] == pytest.approx(0.3)
    clear = processor.get_clear_path(depth)
    assert not clear["is_clear"]

    plane_depth = np.full((20, 20), 2.0, dtype=np.float32)
    plane = processor.compute_ground_plane(plane_depth)
    assert plane is not None
    assert np.isfinite(plane["normal"]).all()


def test_safety_monitor_emergency_and_tracking_timeout():
    monitor = SafetyMonitor(max_tilt_angle=0.2, max_tracking_lost_time=0.01)
    tilted = monitor.check_safety(imu_rpy=np.array([0.3, 0, 0]))
    assert not tilted.is_safe and tilted.emergency_stop

    monitor.reset()
    monitor.check_safety(slam_tracking=False)
    monitor._tracking_lost_since = time.time() - 1
    lost = monitor.check_safety(slam_tracking=False)
    assert not lost.is_safe and "tracking lost" in lost.reason

    critical = monitor.check_safety(battery_level=4)
    assert not critical.is_safe and critical.reason == "Critical battery level"


def test_pose_estimator_fuses_sources_and_preserves_zero_timestamp():
    estimator = PoseEstimator(slam_weight=0.75, odom_weight=0.25)
    empty = estimator.update(timestamp=0.0)
    assert empty.timestamp == 0.0 and empty.confidence == 0

    slam = SLAMPose(
        position=np.array([2.0, 4.0, 1.0]),
        orientation=np.array([0, 0, 0, 1]),
        tracking_status="OK",
    )
    fused = estimator.update(
        slam_pose=slam,
        odom_pose=np.array([0.0, 0.0, 0.2]),
        imu_rpy=np.array([0.0, 0.0, 0.4]),
        timestamp=1.0,
    )
    np.testing.assert_allclose(fused.position, [1.5, 3.0, 1.0])
    assert fused.yaw == pytest.approx(0.3)
    assert fused.source == "fused"
    estimator.reset()
    assert estimator.get_pose() is None
