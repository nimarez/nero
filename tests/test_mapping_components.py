import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero.mapping.trajectory_recorder import TrajectoryRecorder
from nero.slam.map_manager import Keyframe, MapManager
from nero.utils.pointcloud_converter import load_pointcloud


def pose(x, y, z, yaw=0.0):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_trajectory_recording_length_bounds_save_and_load(tmp_path):
    recorder = TrajectoryRecorder(str(tmp_path))
    recorder.add_point(pose(10, 10, 10))
    assert recorder.get_point_count() == 0
    recorder.start()
    recorder.add_point(pose(0, 0, 0))
    recorder.add_point(pose(3, 4, 0, yaw=0.5))
    recorder.stop()
    assert recorder.get_length() == pytest.approx(5.0)
    assert recorder.get_bounds() == {"min": [0, 0, 0], "max": [3, 4, 0]}
    path = recorder.save("trajectory.json")
    data = json.loads((tmp_path / "trajectory.json").read_text())
    assert data["num_points"] == 2
    assert data["length_meters"] == pytest.approx(5.0)

    loaded = TrajectoryRecorder(str(tmp_path / "other"))
    points = loaded.load(path)
    assert len(points) == 2
    assert points[-1].yaw == pytest.approx(0.5)


def test_map_manager_round_trip_listing_trajectory_and_delete(tmp_path):
    manager = MapManager(str(tmp_path))
    keyframe = Keyframe(1, [1, 2, 3], [0, 0, 0, 1], 4.0)
    path = manager.save_map(
        "office",
        [[0, 0, 0], [2, 3, 0]],
        [keyframe],
        description="test map",
    )
    assert path.endswith("office.json")
    loaded = manager.load_map("office")
    assert loaded is not None
    assert loaded.metadata.area_covered_m2 == 6
    assert loaded.keyframes == [keyframe]
    np.testing.assert_array_equal(
        manager.get_trajectory("office"), [[0, 0, 0], [2, 3, 0]]
    )
    assert [item.name for item in manager.list_maps()] == ["office"]
    assert manager.delete_map("office")
    assert not manager.delete_map("office")
    assert manager.load_map("missing") is None


def test_simple_ascii_ply_and_numpy_pointcloud_loading(tmp_path):
    npy_path = tmp_path / "points.npy"
    expected = np.array([[1.0, 2.0, 3.0]])
    np.save(npy_path, expected)
    np.testing.assert_array_equal(load_pointcloud(npy_path), expected)

    ply_path = tmp_path / "points.ply"
    ply_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n1 2 3\n4 5 6\n"
    )
    points = load_pointcloud(ply_path)
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])

    unsupported = tmp_path / "points.xyz"
    unsupported.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        load_pointcloud(unsupported)
