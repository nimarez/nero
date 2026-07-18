import time

import numpy as np

from nero.navigation.policy import NavigationPolicy, PolicyState
from nero.simulation.environment import SimEnvironment
from nero.simulation.mock_robot import MockRobot
from nero.simulation.sim_camera import CameraMode, SimCamera


def test_mock_robot_clamps_velocity_and_integrates_pose():
    robot = MockRobot()
    robot.initialize()
    robot.set_velocity(5.0, -5.0, 5.0)
    assert robot.state.vx == 0.3
    assert robot.state.vy == -0.2
    assert robot.state.vyaw == 1.0

    robot._last_update = time.time() - 1.0
    pose = robot.get_pose()
    assert pose[0] > 0.25
    assert pose[1] < -0.15
    assert 0.9 < pose[2] < 1.1


def test_sim_camera_modes_and_depth_shapes():
    for mode in CameraMode:
        camera = SimCamera(width=80, height=60, mode=mode)
        assert camera.start()
        frame = camera.get_frame()
        assert frame is not None
        assert frame.shape == (60, 80, 3)
        camera.stop()

    camera = SimCamera(width=40, height=30)
    camera.add_object("chair", 1.0, 0.0)
    camera.start()
    depth = camera.get_depth_frame()
    assert depth is not None
    assert depth.shape == (30, 40)
    assert depth.dtype == np.float32


def test_environment_returns_visible_detections_in_robot_frame():
    sim = SimEnvironment()
    sim.add_object("chair", 2.0, 0.0)
    sim.add_object("behind", -1.0, 0.0)
    detections = sim.get_detections()
    assert [d.label for d in detections] == ["chair"]
    assert detections[0].distance == 2.0
    np.testing.assert_allclose(detections[0].position_3d, [0.0, 0.0, 2.0])
    assert detections[0].angle == 0.0


def test_sim_policy_reaches_target():
    sim = SimEnvironment(camera_width=80, camera_height=60)
    sim.add_object("chair", 1.2, 0.0)
    policy = NavigationPolicy(sim_env=sim)
    policy.start()
    assert policy.step().state == PolicyState.WAITING_FOR_OBJECT
    policy.set_target("chair")
    policy._goal.target_distance = 1.0
    assert policy.step().state == PolicyState.NAVIGATING

    for _ in range(100):
        sim.robot._last_update = time.time() - 0.1
        status = policy.step()
        if status.state == PolicyState.ARRIVED:
            break

    assert status.state == PolicyState.ARRIVED
    assert sim.robot.state.vx == 0.0
    policy.stop()


def test_sim_policy_loses_missing_target_without_crashing():
    sim = SimEnvironment(camera_width=80, camera_height=60)
    sim.add_object("chair", 2.0, 0.0)
    policy = NavigationPolicy(sim_env=sim)
    policy.start()
    policy.set_target("chair")
    assert policy.step().state == PolicyState.NAVIGATING
    sim.clear_environment()

    for _ in range(policy._max_object_not_found):
        status = policy.step()

    assert status.state == PolicyState.LOST
    assert "Lost object" in status.message
    policy.stop()


def test_reset_resumes_live_scanning_while_policy_is_running():
    sim = SimEnvironment(camera_width=80, camera_height=60)
    policy = NavigationPolicy(sim_env=sim)
    policy.start()
    assert policy.reset().state == PolicyState.SHOWING_CAMERA
    policy.stop()
