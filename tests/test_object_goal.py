import numpy as np

from nero.navigation.object_goal import (
    approach_pose,
    blend_world_position,
    body_point_to_world,
    camera_point_to_world,
    planar_detection_to_world,
)


def test_camera_point_uses_full_world_transform():
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 20.0, 1.0]
    pose[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(
        camera_point_to_world(np.array([1.0, 0.0, 2.0]), pose),
        [10.0, 21.0, 3.0],
    )


def test_planar_projection_respects_camera_right_and_robot_yaw():
    # A point on camera-right becomes robot-right (negative local y).
    point = planar_detection_to_world(
        np.array([1.0, 0.0, 2.0]), np.array([5.0, 6.0, 0.0])
    )
    np.testing.assert_allclose(point, [7.0, 5.0, 0.0])

    rotated = planar_detection_to_world(
        np.array([0.0, 0.0, 2.0]), np.array([5.0, 6.0, np.pi / 2])
    )
    np.testing.assert_allclose(rotated, [5.0, 8.0, 0.0], atol=1e-12)


def test_body_point_projection_uses_forward_left_convention():
    world = body_point_to_world(
        np.array([2.0, 1.0, 0.5]), np.array([5.0, 6.0, np.pi / 2])
    )
    np.testing.assert_allclose(world, [4.0, 8.0, 0.5], atol=1e-12)


def test_approach_pose_has_position_and_facing_constraints():
    goal = approach_pose(np.zeros(3), np.array([3.0, 4.0, 0.0]), 1.0)
    np.testing.assert_allclose(goal[:2], [2.4, 3.2])
    assert goal[2] == np.arctan2(4.0, 3.0)

    # Inside the safety radius, turn to face the object without moving closer.
    close = approach_pose(np.zeros(3), np.array([0.0, 0.5, 0.0]), 1.0)
    np.testing.assert_allclose(close[:2], [0.0, 0.0])
    assert close[2] == np.pi / 2


def test_object_track_filter_is_explicit_and_bounded():
    observed = np.array([2.0, 4.0, 0.0])
    np.testing.assert_allclose(blend_world_position(None, observed, 0.25), observed)
    np.testing.assert_allclose(
        blend_world_position(np.zeros(3), observed, 0.25), [0.5, 1.0, 0.0]
    )
