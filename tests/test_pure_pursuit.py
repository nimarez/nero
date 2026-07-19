from types import SimpleNamespace

import numpy as np

import nero.agents.pure_pursuit_agent as pursuit_agent
from nero.agents.pure_pursuit_agent import DirectPursuitPolicy, PursuitState
from nero.navigation.pure_pursuit import PurePursuitConfig, PurePursuitController
from nero.perception.object_detector import ObjectDetection


def test_pure_pursuit_drives_straight_to_centered_target():
    command = PurePursuitController().compute_command(
        np.array([0.0, 0.0, 2.0]), stand_off=0.8
    )

    assert command.linear_x > 0
    assert command.angular_z == 0


def test_pure_pursuit_curves_toward_camera_right():
    command = PurePursuitController().compute_command(
        np.array([0.5, 0.0, 2.0]), stand_off=0.8
    )

    assert command.linear_x > 0
    assert command.angular_z < 0


def test_pure_pursuit_stops_at_stand_off():
    controller = PurePursuitController()
    target = np.array([0.0, 0.0, 0.85])

    assert controller.has_arrived(target, stand_off=0.8)
    assert controller.compute_command(target, stand_off=0.8).linear_x == 0


def test_pure_pursuit_turns_to_face_target_at_stand_off():
    controller = PurePursuitController()
    target = np.array([0.2, 0.0, 0.8])

    command = controller.compute_command(target, stand_off=0.8)

    assert command.linear_x == 0
    assert command.angular_z < 0


def test_pure_pursuit_rejects_minimum_above_configured_maximum():
    with np.testing.assert_raises_regex(ValueError, "min_linear_velocity"):
        PurePursuitController(
            PurePursuitConfig(
                max_linear_velocity=0.01,
                min_linear_velocity=0.05,
            )
        )


def test_direct_policy_pursues_live_detection_without_slam():
    velocities = []
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 2000, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=np.zeros(3),
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: None,
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *values: velocities.append(values),
    )
    detection = ObjectDetection(
        label="chair",
        confidence=0.9,
        bbox=(1, 1, 4, 4),
        position_3d=np.array([0.0, 0.0, 2.0]),
        distance=2.0,
    )
    detector = SimpleNamespace(
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        detect=lambda *_args: [detection],
        find_object=lambda detections, _name: detections[0],
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(robot, object_detector=detector)

    policy.start()
    policy.set_target("chair")
    status = policy.step()

    assert status.state == PursuitState.NAVIGATING
    assert status.velocity_command.linear_x > 0
    assert velocities[-1][0] > 0


def test_direct_policy_expires_replayed_async_detection(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pursuit_agent.time, "monotonic", lambda: now[0])
    velocities = []
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 2000, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=np.zeros(3),
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: None,
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *values: velocities.append(values),
    )
    detection = ObjectDetection(
        label="chair",
        confidence=0.9,
        bbox=(1, 1, 4, 4),
        position_3d=np.array([0.0, 0.0, 2.0]),
        distance=2.0,
    )
    detector = SimpleNamespace(
        result_revision=1,
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        detect=lambda *_args: [detection],
        find_object=lambda detections, _name: detections[0],
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(
        robot, object_detector=detector, target_timeout=1.0
    )
    policy.start()
    policy.set_target("chair")
    assert policy.step().state == PursuitState.NAVIGATING

    now[0] = 1.1
    status = policy.step()

    assert status.state == PursuitState.LOST
    assert velocities[-1] == (0.0, 0.0, 0.0)


def test_direct_policy_uses_separate_initial_acquisition_timeout(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pursuit_agent.time, "monotonic", lambda: now[0])
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 2000, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=np.zeros(3),
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: None,
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *_values: None,
    )
    detector = SimpleNamespace(
        result_revision=0,
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        detect=lambda *_args: [],
        find_object=lambda _detections, _name: None,
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(
        robot,
        object_detector=detector,
        target_timeout=1.0,
        acquisition_timeout=10.0,
    )
    policy.start()
    policy.set_target("chair")

    now[0] = 2.0
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 10.1
    assert policy.step().state == PursuitState.LOST


def test_run_agent_cleans_up_when_listener_start_fails(monkeypatch):
    events = []

    class FailedListener:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            events.append("listener start")
            raise RuntimeError("socket busy")

        def close(self):
            events.append("listener close")

    robot = SimpleNamespace(
        initialize=lambda: events.append("robot initialize"),
        stop=lambda: events.append("robot stop"),
        set_velocity=lambda *_values: events.append("robot velocity"),
    )
    detector = SimpleNamespace(
        initialize=lambda: events.append("detector initialize") or True,
        supports_target=lambda _name: True,
        close=lambda: events.append("detector close"),
    )
    args = SimpleNamespace(
        max_velocity=0.25,
        max_angular_velocity=0.7,
        target_timeout=3.0,
        acquisition_timeout=20.0,
        no_display=True,
    )
    monkeypatch.setattr(pursuit_agent, "NavigationTargetListener", FailedListener)
    monkeypatch.setattr(pursuit_agent.signal, "signal", lambda *_args: None)

    with np.testing.assert_raises_regex(RuntimeError, "socket busy"):
        pursuit_agent.run_agent(
            robot,
            args,
            object_detector=detector,
            command_source=SimpleNamespace(),
        )

    assert "robot velocity" in events
    assert "detector close" in events
    assert "robot stop" in events
    assert "listener close" in events
