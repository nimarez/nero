from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from nero.navigation.policy import NavigationGoal, NavigationPolicy, PolicyState
from nero.navigation.runtime import LocalizedFrame, SensorFrame
from nero.perception.object_detector import ObjectDetection
from nero.slam.orb_slam3_node import SLAMPose
from nero.slam.pose_estimator import FusedPose


class RecordingRobot:
    def __init__(self):
        self.commands = []

    def set_velocity(self, vx, vy, vyaw):
        self.commands.append((vx, vy, vyaw))

    def stop(self):
        self.commands.append((0.0, 0.0, 0.0))


def test_policy_exposes_only_detections_matching_spoken_target():
    chair = ObjectDetection("red chair", 0.9, (0, 0, 10, 10))
    bottle = ObjectDetection("bottle", 0.8, (0, 0, 10, 10))
    policy = NavigationPolicy(sim_env=SimpleNamespace())
    policy._state = PolicyState.WAITING_FOR_OBJECT
    policy.set_target("chair")

    assert policy._matching_target_detections([bottle, chair]) == [chair]


def test_sensor_failure_stops_robot_before_entering_terminal_error():
    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot)
    policy._running = True
    policy._state = PolicyState.DETECTING
    policy._get_sensor_data = lambda: None

    status = policy.step()

    assert status.state == PolicyState.ERROR
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_rejected_stop_does_not_crash_the_policy(caplog):
    class UnavailableLocomotionRobot:
        def stop(self):
            raise RuntimeError("API call failed, code = 400")

    policy = NavigationPolicy(robot=UnavailableLocomotionRobot())

    policy._stop_robot()

    assert "rejected the stop command" in caplog.text


def test_real_policy_projects_camera_detection_into_world_goal():
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    camera_to_world[:3, 3] = [10.0, 20.0, 1.0]
    camera_pose = SLAMPose(
        position=camera_to_world[:3, 3],
        orientation=Rotation.from_matrix(camera_to_world[:3, :3]).as_quat(),
        tracking_status="OK",
    )
    body_pose = SLAMPose(
        position=np.array([10.0, 20.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        tracking_status="OK",
    )
    detection = ObjectDetection(
        "chair",
        0.9,
        (0, 0, 10, 10),
        position_3d=np.array([1.0, 0.0, 3.0]),
        distance=np.sqrt(10.0),
    )

    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot, object_position_filter=1.0)
    policy._state = PolicyState.WAITING_FOR_OBJECT
    policy.set_target("chair")
    policy.slam = SimpleNamespace(
        track_frame=lambda *args, **kwargs: camera_pose,
        body_pose=lambda _: body_pose,
    )
    policy.pose_estimator = SimpleNamespace(
        update=lambda **kwargs: FusedPose(
            position=np.array([10.0, 20.0, 0.0]), yaw=0.0, timestamp=1.0
        ),
        get_pose=lambda: None,
    )
    policy.safety = SimpleNamespace(
        check_safety=lambda **kwargs: SimpleNamespace(is_safe=True)
    )
    policy.depth_processor = SimpleNamespace(
        preprocess=lambda depth: depth,
        detect_obstacles=lambda depth: {"has_obstacle": False},
    )
    policy.object_detector = SimpleNamespace(
        detect=lambda *args: [detection],
        find_object=lambda detections, name: detections[0],
    )

    status = policy._step_navigating(
        {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "camera_info": None,
            "imu_rpy": np.zeros(3),
            "odometry": np.array([10.0, 20.0, 0.0]),
            "imu_samples": [(0.0,) * 7],
            "timestamp": 1.0,
        }
    )

    np.testing.assert_allclose(
        status.current_goal.object_position_world, [13.0, 19.0, 1.0]
    )
    expected_yaw = np.arctan2(-1.0, 3.0)
    assert status.current_goal.approach_pose[2] == expected_yaw
    assert status.state == PolicyState.NAVIGATING
    assert robot.commands[-1][0] > 0.0

    # Booster Studio publishes the same semantic point in the K1 trunk frame.
    detection.coordinate_frame = "body"
    detection.position_3d = np.array([2.0, 1.0, 0.0])
    status = policy._step_navigating(
        {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "camera_info": None,
            "imu_rpy": np.zeros(3),
            "odometry": np.array([10.0, 20.0, 0.0]),
            "imu_samples": [(0.0,) * 7],
            "timestamp": 2.0,
        }
    )
    np.testing.assert_allclose(
        status.current_goal.object_position_world, [12.0, 21.0, 0.0]
    )


def test_arrival_requires_full_pose_and_track_freshness(monkeypatch):
    policy = NavigationPolicy(sim_env=SimpleNamespace(), object_track_timeout=1.0)
    policy._state = PolicyState.WAITING_FOR_OBJECT
    policy.set_target("chair")
    policy._goal.approach_pose = np.array([0.0, 0.0, np.pi / 2])
    policy._goal.last_observed_monotonic = 10.0

    assert not policy._goal_reached(np.zeros(3))
    assert policy._goal_reached(np.array([0.0, 0.0, np.pi / 2]))
    monkeypatch.setattr("nero.navigation.policy.time.monotonic", lambda: 10.5)
    assert policy._goal_is_fresh()
    monkeypatch.setattr("nero.navigation.policy.time.monotonic", lambda: 11.1)
    assert not policy._goal_is_fresh()


def _localized_frame(*, tracking="OK", safe=True, min_distance=2.0):
    sensor = SensorFrame(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.full((4, 4), 2000, dtype=np.uint16),
        timestamp=1.0,
        camera_info=None,
        imu_rpy=np.zeros(3),
        imu_samples=[],
        odometry=np.zeros(3),
    )
    return LocalizedFrame(
        sensor=sensor,
        slam_pose=SLAMPose(
            position=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            tracking_status=tracking,
        ),
        fused_pose=FusedPose(position=np.zeros(3), yaw=0.0, timestamp=1.0),
        safety_status=SimpleNamespace(
            is_safe=safe, reason="obstacle" if not safe else ""
        ),
        obstacle_info={
            "has_obstacle": min_distance < 0.5,
            "sensor_blind": False,
            "min_distance": min_distance,
        },
    )


def test_initial_slam_loss_retains_target_and_executes_bounded_bootstrap(monkeypatch):
    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot)
    policy._running = True
    policy._state = PolicyState.DETECTING
    policy._goal = NavigationGoal("green can")
    localized = _localized_frame(tracking="LOST")
    policy._get_sensor_data = lambda: localized.sensor
    monkeypatch.setattr(
        "nero.navigation.policy.localize_sensor_frame", lambda *args, **kwargs: localized
    )

    status = policy.step()

    assert status.state == PolicyState.LOCALIZING
    assert status.current_goal.object_name == "green can"
    assert status.velocity_command.angular_z == policy._slam_bootstrap_angular_velocity
    assert robot.commands[-1] == (0.0, 0.0, policy._slam_bootstrap_angular_velocity)


def test_tracking_loss_after_initialization_holds_without_open_loop_motion(monkeypatch):
    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot)
    policy._running = True
    policy._state = PolicyState.NAVIGATING
    policy._goal = NavigationGoal("green can")
    policy._slam_ever_tracked = True
    localized = _localized_frame(tracking="LOST")
    policy._get_sensor_data = lambda: localized.sensor
    monkeypatch.setattr(
        "nero.navigation.policy.localize_sensor_frame", lambda *args, **kwargs: localized
    )

    status = policy.step()

    assert status.state == PolicyState.LOCALIZING
    assert status.current_goal.object_name == "green can"
    assert status.velocity_command.angular_z == 0.0
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_close_obstacle_blocks_motion_without_discarding_command(monkeypatch):
    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot)
    policy._running = True
    policy._state = PolicyState.NAVIGATING
    policy._goal = NavigationGoal("green can")
    localized = _localized_frame(safe=False, min_distance=0.2)
    policy._get_sensor_data = lambda: localized.sensor
    monkeypatch.setattr(
        "nero.navigation.policy.localize_sensor_frame", lambda *args, **kwargs: localized
    )

    status = policy.step()

    assert status.state == PolicyState.NAVIGATING
    assert status.current_goal.object_name == "green can"
    assert "blocked by obstacle" in status.message
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_navigation_safety_opt_out_keeps_raw_status_and_bypasses_obstacles(monkeypatch):
    robot = RecordingRobot()
    policy = NavigationPolicy(robot=robot, safety_enforced=False)
    policy._running = True
    policy._state = PolicyState.NAVIGATING
    policy._goal = NavigationGoal("green can")
    localized = _localized_frame(safe=False, min_distance=0.2)
    policy._get_sensor_data = lambda: localized.sensor
    monkeypatch.setattr(
        "nero.navigation.policy.localize_sensor_frame", lambda *args, **kwargs: localized
    )
    policy._step_navigating = lambda *_args: policy._update_status(
        safety_status=localized.safety_status,
        message="continued",
    )

    status = policy.step()
    control_obstacles = policy._control_obstacle_info(localized.obstacle_info)

    assert status.state == PolicyState.NAVIGATING
    assert status.safety_status.is_safe is False
    assert status.safety_enforced is False
    assert control_obstacles["has_obstacle"] is False
    assert control_obstacles["sensor_blind"] is False
    assert np.isinf(control_obstacles["min_distance"])
    assert robot.commands == []


def test_direct_navigation_exposes_current_to_goal_plan():
    policy = NavigationPolicy(sim_env=SimpleNamespace())
    policy._state = PolicyState.NAVIGATING
    policy._goal = NavigationGoal(
        "green can", approach_pose=np.array([3.0, 4.0, 0.5])
    )

    status = policy._update_status(robot_pose=np.array([1.0, 2.0, 0.0]))

    np.testing.assert_allclose(
        status.planned_path, [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
    )
