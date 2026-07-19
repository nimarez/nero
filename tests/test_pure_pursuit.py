from types import SimpleNamespace

import numpy as np

import nero.agents.pure_pursuit_agent as pursuit_agent
from nero.agents.pure_pursuit_agent import (
    DEFAULT_HEAD_SCAN_POSES,
    DirectPursuitPolicy,
    HeadScanConfig,
    PursuitState,
)
from nero.navigation.pure_pursuit import PurePursuitConfig, PurePursuitController
from nero.perception.object_detector import ObjectDetection


def test_pure_pursuit_drives_straight_to_centered_target():
    command = PurePursuitController().compute_command(np.array([0.0, 0.0, 2.0]), stand_off=0.8)

    assert command.linear_x > 0
    assert command.angular_z == 0


def test_pure_pursuit_curves_toward_camera_right():
    command = PurePursuitController().compute_command(np.array([0.5, 0.0, 2.0]), stand_off=0.8)

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


def test_direct_policy_pursues_live_detection_without_slam(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pursuit_agent.time, "monotonic", lambda: now[0])
    velocities = []
    head_poses = []
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
        set_head_pose=lambda *values: head_poses.append(values),
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
    policy = DirectPursuitPolicy(
        robot,
        object_detector=detector,
        head_scan=HeadScanConfig(poses=((0.0, 0.0),), move_duration=0.01, settle_time=0.0),
    )

    policy.start()
    policy.set_target("chair")
    assert policy.step().state == PursuitState.EXPLORING
    now[0] = 0.02
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 0.03
    assert policy.step().state == PursuitState.ALIGNING
    now[0] = 0.05
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 0.06
    status = policy.step()

    assert status.state == PursuitState.NAVIGATING
    assert status.velocity_command.linear_x > 0
    assert velocities[-1][0] > 0
    assert status.stand_off_distance == 1.0
    assert status.stand_off_tolerance == 0.12
    assert status.target_position_camera == [0.0, 0.0, 2.0]
    assert head_poses == [(0.0, 0.0, 0.01), (0.0, 0.0, 0.01)]
    assert all(values == (0.0, 0.0, 0.0) for values in velocities[:-1])


def test_direct_policy_safety_opt_out_preserves_diagnostics_but_allows_motion():
    velocities = []
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 200, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=np.zeros(3),
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *values: velocities.append(values),
        set_head_pose=lambda *_values: None,
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
        detect=lambda *_args: [detection],
        find_object=lambda detections, _name: detections[0],
    )
    obstacles = {
        "has_obstacle": True,
        "sensor_blind": True,
        "min_distance": 0.0,
        "left_clear": False,
        "center_clear": False,
        "right_clear": False,
    }
    unsafe = SimpleNamespace(is_safe=False, reason="Depth sensor blind")
    policy = DirectPursuitPolicy(
        robot,
        object_detector=detector,
        depth_processor=SimpleNamespace(
            preprocess=lambda depth: depth,
            detect_obstacles=lambda _depth: obstacles,
        ),
        safety=SimpleNamespace(check_safety=lambda **_kwargs: unsafe),
        safety_enforced=False,
    )
    policy._running = True
    policy.target = "chair"

    status = policy.step()

    assert status.state == PursuitState.NAVIGATING
    assert status.safety_status is unsafe
    assert status.safety_enforced is False
    assert status.velocity_command.linear_x > 0.0
    assert velocities[-1][0] > 0.0


def test_direct_policy_enforces_safety_by_default():
    robot = SimpleNamespace(set_velocity=lambda *_values: None)
    policy = DirectPursuitPolicy(robot, object_detector=SimpleNamespace())

    assert policy.safety_enforced is True
    assert policy._status("ready").safety_enforced is True


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
        set_head_pose=lambda *_values: None,
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
    policy = DirectPursuitPolicy(robot, object_detector=detector, target_timeout=1.0)
    policy.start()
    policy.set_target("chair")
    assert policy.step().state == PursuitState.EXPLORING

    now[0] = 0.5
    assert policy.step().state == PursuitState.EXPLORING
    detector.result_revision = 2
    now[0] = 0.6
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 1.7
    status = policy.step()

    assert status.state == PursuitState.EXPLORING
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
        set_head_pose=lambda *_values: None,
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
    assert policy.step().state == PursuitState.EXPLORING
    now[0] = 10.1
    assert policy.step().state == PursuitState.LOST


def test_direct_policy_scans_every_default_head_pose_without_moving_base(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pursuit_agent.time, "monotonic", lambda: now[0])
    velocities = []
    head_poses = []
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
        set_head_pose=lambda *values: head_poses.append(values),
    )
    detector = SimpleNamespace(
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        detect=lambda *_args: [],
        find_object=lambda _detections, _name: None,
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(robot, object_detector=detector)
    policy.start()
    policy.set_target("chair")

    status = policy.step()
    for index in range(1, len(DEFAULT_HEAD_SCAN_POSES)):
        now[0] = index * 0.51
        status = policy.step()

    assert status.state == PursuitState.EXPLORING
    assert status.exploration_step == len(DEFAULT_HEAD_SCAN_POSES)
    assert status.exploration_steps == len(DEFAULT_HEAD_SCAN_POSES)
    assert head_poses == [(pitch, yaw, 0.35) for pitch, yaw in DEFAULT_HEAD_SCAN_POSES]
    assert velocities
    assert all(values == (0.0, 0.0, 0.0) for values in velocities)


def test_direct_policy_only_rotates_after_side_detection_then_reconfirms(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pursuit_agent.time, "monotonic", lambda: now[0])
    velocities = []
    odometry = np.zeros(3)
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 2000, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=odometry,
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: None,
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *values: velocities.append(values),
        set_head_pose=lambda *_values: None,
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
    policy = DirectPursuitPolicy(
        robot,
        object_detector=detector,
        head_scan=HeadScanConfig(poses=((0.0, 0.75),), move_duration=0.01, settle_time=0.0),
    )
    policy.start()
    policy.set_target("chair")

    assert policy.step().state == PursuitState.EXPLORING
    now[0] = 0.02
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 0.03
    assert policy.step().state == PursuitState.ALIGNING
    assert all(values == (0.0, 0.0, 0.0) for values in velocities)

    now[0] = 0.05
    aligning = policy.step()
    assert aligning.state == PursuitState.ALIGNING
    assert aligning.velocity_command.linear_x == 0.0
    assert aligning.velocity_command.angular_z > 0.0

    odometry[2] = 0.75
    now[0] = 0.06
    assert policy.step().state == PursuitState.DETECTING
    now[0] = 0.07
    navigating = policy.step()
    assert navigating.state == PursuitState.NAVIGATING
    assert navigating.velocity_command.linear_x > 0.0


def test_direct_policy_stops_when_head_scan_command_is_rejected():
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
        set_head_pose=lambda *_values: (_ for _ in ()).throw(
            RuntimeError("head service unavailable")
        ),
    )
    detector = SimpleNamespace(
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        detect=lambda *_args: [],
        find_object=lambda _detections, _name: None,
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(robot, object_detector=detector)
    policy.start()
    policy.set_target("chair")

    status = policy.step()

    assert status.state == PursuitState.ERROR
    assert "Head command failed" in status.message
    assert velocities[-1] == (0.0, 0.0, 0.0)


def test_direct_policy_never_starts_head_scan_when_base_stop_is_rejected():
    head_poses = []
    robot = SimpleNamespace(
        initialize=lambda: None,
        stop=lambda: None,
        set_velocity=lambda *_values: (_ for _ in ()).throw(
            RuntimeError("walking controller rejected stop")
        ),
        set_head_pose=lambda *values: head_poses.append(values),
    )
    detector = SimpleNamespace(
        initialize=lambda: True,
        supports_target=lambda _name: True,
        resolve_target=lambda name: name,
        set_target=lambda _name: None,
        close=lambda: None,
    )
    policy = DirectPursuitPolicy(robot, object_detector=detector)
    policy.start()

    status = policy.set_target("chair")

    assert status.state == PursuitState.ERROR
    assert "Locomotion command failed" in status.message
    assert head_poses == []
    assert policy.step().state == PursuitState.ERROR


def test_direct_policy_survives_rejected_zero_velocity_command():
    state = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.full((8, 8), 2000, dtype=np.uint16),
        camera_info=SimpleNamespace(k=np.eye(3)),
        orientation_rpy=np.zeros(3),
        position_2d=np.zeros(3),
        battery_level=100.0,
    )
    robot = SimpleNamespace(
        get_state=lambda include_images=True: state,
        image_to_array=np.asarray,
        image_timestamp=lambda _image: 1.0,
        set_velocity=lambda *_values: (_ for _ in ()).throw(
            RuntimeError("API call failed, code = 400")
        ),
    )
    policy = DirectPursuitPolicy(robot, object_detector=SimpleNamespace())
    policy._running = True
    policy.state = PursuitState.WAITING_FOR_OBJECT

    status = policy.step()

    assert status.state == PursuitState.WAITING_FOR_OBJECT


def test_run_agent_streams_telemetry_and_announces_missing_target_once(monkeypatch):
    spoken = []
    published = []
    sensor = SimpleNamespace(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=1.0,
        raw_state=SimpleNamespace(),
    )

    class LostPolicy:
        last_sensor = sensor

        def __init__(self, *args, **kwargs):
            self.calls = 0

        def start(self):
            pass

        def supports_target(self, _name):
            return True

        def set_target(self, name):
            self.target = name

        def step(self):
            self.calls += 1
            state = PursuitState.LOST if self.calls == 1 else PursuitState.ERROR
            return SimpleNamespace(
                state=state,
                message="not found",
                velocity_command=SimpleNamespace(linear_x=0.0, angular_z=0.0),
            )

        def reset(self):
            pass

        def stop(self):
            pass

    class Listener:
        def __init__(self, *args, **kwargs):
            self.commands = ["green cup"]

        def start(self):
            pass

        def poll(self):
            return self.commands.pop(0) if self.commands else None

        def close(self):
            pass

    telemetry = SimpleNamespace(
        publish_robot_state=lambda *args: published.append("sensors"),
        publish_policy=lambda *args: published.append("policy"),
        close=lambda: published.append("closed"),
    )
    monkeypatch.setattr(pursuit_agent, "DirectPursuitPolicy", LostPolicy)
    monkeypatch.setattr(pursuit_agent, "NavigationTargetListener", Listener)
    monkeypatch.setattr(pursuit_agent.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        pursuit_agent.RosObservabilityPublisher,
        "try_create",
        lambda **kwargs: telemetry,
    )
    robot = SimpleNamespace(
        stop=lambda: None,
        speak=spoken.append,
    )
    args = SimpleNamespace(
        max_velocity=0.25,
        max_angular_velocity=0.7,
        target_timeout=3.0,
        acquisition_timeout=20.0,
        search_angular_velocity=0.12,
        no_ros_observability=False,
        no_display=True,
    )

    pursuit_agent.run_agent(
        robot,
        args,
        object_detector=SimpleNamespace(),
        command_source=SimpleNamespace(),
    )

    assert spoken == ["I could not detect the green cup."]
    assert published.count("sensors") == 2
    assert published.count("policy") == 2
    assert published[-1] == "closed"


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
        set_head_pose=lambda *_values: events.append("head pose"),
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


def test_direct_policy_closes_robot_when_sensor_startup_fails():
    events = []
    robot = SimpleNamespace(
        initialize=lambda: (_ for _ in ()).throw(RuntimeError("camera silent")),
        stop=lambda: events.append("robot stop"),
        close=lambda: events.append("robot close"),
    )
    detector = SimpleNamespace(close=lambda: events.append("detector close"))
    policy = pursuit_agent.DirectPursuitPolicy(robot, object_detector=detector)

    with np.testing.assert_raises_regex(RuntimeError, "camera silent"):
        policy.start()

    assert events == ["detector close", "robot close"]
