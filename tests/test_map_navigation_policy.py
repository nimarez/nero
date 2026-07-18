from types import SimpleNamespace

import numpy as np

from nero.navigation.map_loader import OccupancyGrid
from nero.navigation.map_policy import MapNavConfig, MapNavigationPolicy, MapNavState
from nero.slam.orb_slam3_node import SLAMPose
from nero.slam.pose_estimator import FusedPose


class RecordingRobot:
    def __init__(self):
        self.commands = []

    def get_state(self, include_images=True):
        return SimpleNamespace(
            rgb=object(),
            depth=object(),
            camera_info=object(),
            orientation_rpy=np.zeros(3),
            imu_samples=[(0.0,) * 7],
            position_2d=np.zeros(3),
        )

    def image_to_array(self, image):
        return np.ones((4, 4), dtype=np.uint16)

    def image_timestamp(self, image):
        return 1.0

    def set_velocity(self, vx, vy, vyaw):
        self.commands.append((vx, vy, vyaw))


def _policy():
    robot = RecordingRobot()
    policy = MapNavigationPolicy(
        MapNavConfig(initial_pose=(2.0, 3.0, np.pi / 2), goal_threshold=0.1),
        robot=robot,
    )
    policy._grid = OccupancyGrid(
        data=np.zeros((10, 10), dtype=np.int8),
        resolution=1.0,
        origin=(0.0, 0.0),
        width=10,
        height=10,
    )
    slam_pose = SLAMPose(
        position=np.zeros(3),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        tracking_status="OK",
    )
    policy.slam = SimpleNamespace(
        track_frame=lambda *args, **kwargs: slam_pose,
        body_pose=lambda pose: pose,
        shutdown=lambda: None,
    )
    poses = iter(
        [
            FusedPose(position=np.array([10.0, 20.0, 0.0]), yaw=0.2),
            FusedPose(position=np.array([11.0, 20.0, 0.0]), yaw=0.2),
        ]
    )
    policy.pose_estimator = SimpleNamespace(update=lambda **kwargs: next(poses))
    policy.depth_processor = SimpleNamespace(
        preprocess=lambda depth: depth,
        detect_obstacles=lambda depth: {"has_obstacle": False},
    )
    policy.safety = SimpleNamespace(check_safety=lambda **kwargs: SimpleNamespace(is_safe=True))
    policy._running = True
    policy._state = MapNavState.LOCALIZING
    return policy, robot


def test_map_policy_uses_shared_localization_and_map_frame_alignment():
    policy, robot = _policy()
    assert policy.set_goal(2.0, 5.0, np.pi / 2)

    first = policy.step()
    np.testing.assert_allclose(first.pose, [2.0, 3.0, np.pi / 2])
    assert first.state == MapNavState.NAVIGATING
    assert robot.commands[-1][0] > 0

    second = policy.step()
    np.testing.assert_allclose(second.pose, [2.0, 4.0, np.pi / 2], atol=1e-7)
    assert second.state == MapNavState.NAVIGATING


def test_map_policy_stops_on_safety_violation():
    policy, robot = _policy()
    policy.safety = SimpleNamespace(
        check_safety=lambda **kwargs: SimpleNamespace(is_safe=False, reason="tilt")
    )
    policy.set_goal(2.0, 5.0)
    status = policy.step()
    assert status.state == MapNavState.ERROR
    assert robot.commands[-1] == (0.0, 0.0, 0.0)
