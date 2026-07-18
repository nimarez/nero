from types import SimpleNamespace

import numpy as np

from nero.navigation.map_loader import OccupancyGrid
from nero.navigation.map_policy import MapNavConfig
from nero.navigation.policy import NavigationPolicy, PolicyState
from nero.perception.object_detector import ObjectDetection
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
    policy = NavigationPolicy(
        robot=robot,
        map_config=MapNavConfig(initial_pose=(2.0, 3.0, np.pi / 2), goal_threshold=0.1),
    )
    policy.map_navigator.set_grid(
        OccupancyGrid(
            data=np.zeros((10, 10), dtype=np.int8),
            resolution=1.0,
            origin=(0.0, 0.0),
            width=10,
            height=10,
        )
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
    policy.safety = SimpleNamespace(
        check_safety=lambda **kwargs: SimpleNamespace(is_safe=True)
    )
    policy._running = True
    policy._state = PolicyState.LOCALIZING
    return policy, robot


def test_map_policy_uses_shared_localization_and_map_frame_alignment():
    policy, robot = _policy()
    policy.set_pose_goal(2.0, 5.0, np.pi / 2)

    first = policy.step()
    np.testing.assert_allclose(first.current_pose.position_2d, [2.0, 3.0, np.pi / 2])
    assert first.state == PolicyState.NAVIGATING
    assert robot.commands[-1][0] > 0

    second = policy.step()
    yaw_offset = np.pi / 2 - 0.2
    np.testing.assert_allclose(
        second.current_pose.position_2d,
        [2.0 + np.cos(yaw_offset), 3.0 + np.sin(yaw_offset), np.pi / 2],
        atol=1e-7,
    )
    assert second.state == PolicyState.NAVIGATING


def test_slam_map_points_use_the_same_non_identity_frame_alignment():
    policy, _ = _policy()
    policy.set_pose_goal(2.0, 5.0, np.pi / 2)
    policy.step()

    points = np.array([[10, 20, 1], [11, 20, 2]])
    transformed = policy.transform_slam_points(points)
    yaw_offset = np.pi / 2 - 0.2
    np.testing.assert_allclose(transformed[0], [2.0, 3.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(
        transformed[1],
        [2.0 + np.cos(yaw_offset), 3.0 + np.sin(yaw_offset), 2.0],
        atol=1e-7,
    )


def test_idle_status_pose_uses_map_frame_after_alignment():
    policy, _ = _policy()
    local_pose = np.array([10.0, 20.0, 0.2])
    policy.map_navigator._set_alignment(local_pose, np.array([2.0, 3.0, np.pi / 2]))
    policy.pose_estimator = SimpleNamespace(
        get_pose=lambda: FusedPose(position=np.array([11.0, 20.0, 0.0]), yaw=0.2)
    )

    status = policy._update_status(message="waiting")

    yaw_offset = np.pi / 2 - 0.2
    np.testing.assert_allclose(
        status.current_pose.position_2d,
        [2.0 + np.cos(yaw_offset), 3.0 + np.sin(yaw_offset), np.pi / 2],
        atol=1e-7,
    )


def test_map_policy_auto_localizes_before_anchoring(monkeypatch):
    policy, robot = _policy()
    policy.map_navigator.config.auto_localize = True
    results = iter(
        [
            SimpleNamespace(
                is_confident=False, score=0.1, ambiguity=1.0, num_points=12
            ),
            SimpleNamespace(
                is_confident=True,
                pose=np.array([2.0, 3.0, np.pi / 2]),
                score=0.9,
                ambiguity=0.3,
                num_points=200,
            ),
        ]
    )
    monkeypatch.setattr(
        "nero.navigation.map_policy.depth_to_planar_scan",
        lambda *args, **kwargs: np.zeros((100, 2)),
    )
    monkeypatch.setattr(
        "nero.navigation.map_policy.GridLocalizer",
        lambda grid, config: SimpleNamespace(localize=lambda scan: next(results)),
    )
    policy.set_pose_goal(2.0, 5.0, np.pi / 2)

    first = policy.step()
    assert first.state == PolicyState.LOCALIZING
    assert "not confident" in first.message
    assert robot.commands[-1] == (
        0.0,
        0.0,
        policy.map_navigator.config.localization_spin_speed,
    )
    assert not policy.map_alignment_ready

    second = policy.step()
    assert policy.map_alignment_ready
    assert second.state == PolicyState.NAVIGATING
    np.testing.assert_allclose(second.current_pose.position_2d, [2.0, 3.0, np.pi / 2])


def test_auto_localization_does_not_move_before_a_goal(monkeypatch):
    policy, robot = _policy()
    policy.map_navigator.config.auto_localize = True
    monkeypatch.setattr(
        "nero.navigation.map_policy.depth_to_planar_scan",
        lambda *args, **kwargs: np.zeros((100, 2)),
    )
    monkeypatch.setattr(
        "nero.navigation.map_policy.GridLocalizer",
        lambda grid, config: SimpleNamespace(
            localize=lambda scan: SimpleNamespace(
                is_confident=False,
                score=0.1,
                ambiguity=1.0,
                num_points=len(scan),
            )
        ),
    )

    status = policy.step()

    assert status.state == PolicyState.LOCALIZING
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_scan_accumulation_round_trips_through_session_frame():
    policy, _ = _policy()
    scan = np.array([[1.0, 0.0], [2.0, 0.5], [0.5, -1.0]])
    pose = np.array([1.0, 2.0, 0.5])
    localized = SimpleNamespace(
        sensor=SimpleNamespace(depth=scan, camera_info=None, imu_rpy=np.zeros(3))
    )
    policy.depth_processor = SimpleNamespace(preprocess=lambda depth: depth)
    import nero.navigation.map_policy as map_policy_module

    original = map_policy_module.depth_to_planar_scan
    map_policy_module.depth_to_planar_scan = lambda depth, **kwargs: depth
    try:
        policy.map_navigator._accumulate_scan(localized, pose, policy.depth_processor)
    finally:
        map_policy_module.depth_to_planar_scan = original
    np.testing.assert_allclose(
        policy.map_navigator._composite_scan(pose), scan, atol=1e-12
    )


def test_map_policy_stops_on_safety_violation():
    policy, robot = _policy()
    policy.safety = SimpleNamespace(
        check_safety=lambda **kwargs: SimpleNamespace(is_safe=False, reason="tilt")
    )
    policy.set_pose_goal(2.0, 5.0)
    status = policy.step()
    assert status.state == PolicyState.ERROR
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_same_policy_routes_live_object_goal_through_optional_map():
    policy, robot = _policy()
    detection = ObjectDetection(
        "green can",
        0.9,
        (0, 0, 3, 3),
        position_3d=np.array([2.0, 0.0, 0.0]),
        coordinate_frame="body",
    )
    policy.object_detector = SimpleNamespace(
        set_target=lambda name: None,
        detect=lambda *args: [detection],
        find_object=lambda detections, name: detections[0] if detections else None,
        result_revision=None,
    )

    policy.set_target("green can")
    assert policy.step().state == PolicyState.NAVIGATING
    status = policy.step()

    assert type(policy) is NavigationPolicy
    assert status.state == PolicyState.NAVIGATING
    assert status.current_goal.kind == "object"
    assert status.current_goal.approach_pose is not None
    assert robot.commands[-1] != (0.0, 0.0, 0.0)


def test_map_routing_exception_fails_closed():
    policy, robot = _policy()
    policy.set_pose_goal(2.0, 5.0)
    policy.map_navigator.route = lambda *args: (_ for _ in ()).throw(
        RuntimeError("planner failed")
    )

    status = policy.step()

    assert status.state == PolicyState.LOST
    assert "planner failed" in status.message
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_map_alignment_exception_fails_closed():
    policy, robot = _policy()
    policy.set_pose_goal(2.0, 5.0)
    policy.map_navigator.update_alignment = lambda *args: (_ for _ in ()).throw(
        RuntimeError("matcher failed")
    )

    status = policy.step()

    assert status.state == PolicyState.ERROR
    assert "matcher failed" in status.message
    assert robot.commands[-1] == (0.0, 0.0, 0.0)
