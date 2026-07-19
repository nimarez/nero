import math
import os
import socket
import stat
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import nero.agents as agents
from nero.navigation.controller import VelocityController
from nero.perception.object_detector import COCO80, ObjectDetection, ObjectDetector
from nero.robot import RobotInterface
from nero.interaction import (
    K1VoiceCommandSource,
    NavigationTargetListener,
    TerminalCommandSource,
    UnixSocketCommandSource,
    parse_go_to_command,
    request_navigation_target,
    safe_stand_off_distance,
)


def test_lazy_agent_exports_are_callable():
    for name in agents.__all__:
        assert callable(getattr(agents, name))


def test_object_detection_geometry():
    detection = ObjectDetection(
        label="chair",
        confidence=0.9,
        bbox=(10, 20, 30, 50),
        position_3d=np.array([1.0, 0.0, 1.0]),
        distance=math.sqrt(2.0),
    )
    assert detection.center == (20.0, 35.0)
    assert detection.size == (20, 30)
    assert detection.angle == math.pi / 4


def test_depth_projection_and_name_matching():
    detector = ObjectDetector()
    depth = np.full((20, 20), 2000, dtype=np.uint16)
    position = detector._compute_3d_position((5, 5, 15, 15), depth)
    assert position is not None
    assert position[2] == 2.0

    far = ObjectDetection("chair", 1.0, (0, 0, 1, 1), distance=3.0)
    near = ObjectDetection("red chair", 1.0, (0, 0, 1, 1), distance=1.0)
    assert detector.find_object([far, near], "CHAIR") is near


def test_velocity_controller_stops_and_clamps():
    controller = VelocityController(max_linear_velocity=0.3, max_angular_velocity=1.0)
    stopped = controller.compute_goal_velocity(np.zeros(3), np.array([0.1, 0.0, 0.0]))
    assert stopped.linear_x == 0.0

    command = controller.compute_goal_velocity(np.zeros(3), np.array([10.0, 10.0, 0.0]))
    assert command.linear_x == 0.3
    assert command.angular_z == 1.0

    turn_in_place = controller.compute_goal_velocity(np.zeros(3), np.array([0.1, 0.0, 1.0]))
    assert turn_in_place.linear_x == 0.0
    assert turn_in_place.angular_z > 0.0
    assert not controller.has_reached_pose(np.zeros(3), np.array([0.1, 0.0, 1.0]))

    reverse = controller.compute_avoidance_velocity({"has_obstacle": True, "min_distance": 0.2})
    assert reverse.linear_x == -0.1


def test_robot_image_helpers_normalize_k1_images():
    image = np.arange(12).reshape(2, 2, 3)
    np.testing.assert_array_equal(RobotInterface.image_to_array(SimpleNamespace(data=image)), image)
    np.testing.assert_array_equal(RobotInterface.image_to_array(image), image)
    stamped = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000))
    )
    assert RobotInterface.image_timestamp(stamped) == 12.5


def test_robot_head_pose_uses_timed_sdk_command_and_validates_k1_limits():
    calls = []
    robot = RobotInterface.__new__(RobotInterface)
    robot._initialized = True
    robot._loco = SimpleNamespace(
        RotateHeadWithTime=lambda *values: calls.append(values) or 0,
    )

    robot.set_head_pose(0.65, -0.75, 0.35)

    assert calls == [(0.65, -0.75, 350)]
    with np.testing.assert_raises_regex(ValueError, "head pitch"):
        robot.set_head_pose(0.9, 0.0)
    with np.testing.assert_raises_regex(ValueError, "head yaw"):
        robot.set_head_pose(0.0, 1.1)


def test_robot_propagates_rejected_velocity_and_head_commands():
    robot = RobotInterface.__new__(RobotInterface)
    robot._initialized = True
    robot._loco = SimpleNamespace(
        Move=lambda *_values: 400,
        RotateHeadWithTime=lambda *_values: 503,
    )

    with np.testing.assert_raises_regex(RuntimeError, r"velocity command \(400\)"):
        robot.set_velocity(0.1, 0.0, 0.0)
    with np.testing.assert_raises_regex(RuntimeError, r"velocity command \(400\)"):
        robot.stop()
    with np.testing.assert_raises_regex(RuntimeError, r"head pose command \(503\)"):
        robot.set_head_pose(0.0, 0.0)


def test_robot_image_helpers_decode_production_k1_encodings():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    depth_message = SimpleNamespace(encoding="mono16", height=3, width=4, data=depth.tobytes())
    np.testing.assert_array_equal(RobotInterface.image_to_array(depth_message), depth)

    # Neutral NV12 encodes a gray image and exercises the K1's exact wire layout.
    nv12 = np.concatenate([np.full(4 * 4, 128, np.uint8), np.full(4 * 2, 128, np.uint8)])
    rgb_message = SimpleNamespace(encoding="nv12", height=4, width=4, data=nv12.tobytes())
    decoded = RobotInterface.image_to_array(rgb_message)
    assert decoded.shape == (4, 4, 3)
    assert decoded.dtype == np.uint8
    assert np.max(decoded) - np.min(decoded) == 0


def test_robot_requires_maximum_registered_k1_camera_resolution():
    robot = RobotInterface.__new__(RobotInterface)
    robot._rgb = SimpleNamespace(width=544, height=448)
    robot._depth = SimpleNamespace(width=544, height=448)
    robot._camera_info = SimpleNamespace(width=544, height=448)

    robot._validate_max_camera_resolution()

    robot._depth = SimpleNamespace(width=320, height=240)
    with np.testing.assert_raises_regex(
        RuntimeError,
        r"maximum registered RGB-D resolution 544x448.*depth=320x240",
    ):
        robot._validate_max_camera_resolution()


def test_robot_consumes_official_battery_soc_and_waits_for_a_new_rgbd_frame():
    robot = RobotInterface.__new__(RobotInterface)
    robot._lock = threading.Lock()
    robot._ready = threading.Condition(robot._lock)
    robot._timeout = 0.5
    robot._battery_level = None
    robot._camera_info = object()
    robot._imu = SimpleNamespace(rpy=np.zeros(3))
    robot._odom = SimpleNamespace(pose_2d=np.zeros(3))
    robot._joints = None
    robot._mode = 2
    robot._imu_samples = []
    robot._last_frame_timestamp = 1.0

    def image(stamp):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp), nanosec=0))
        )

    robot._rgb = image(1)
    robot._depth = image(1)
    robot._on_battery_state(SimpleNamespace(soc=42.5))
    assert robot._battery_level == 42.5

    states = []
    reader = threading.Thread(target=lambda: states.append(robot.get_state()))
    reader.start()
    time.sleep(0.02)
    assert reader.is_alive()
    with robot._ready:
        robot._rgb = image(2)
        robot._depth = image(2)
        robot._ready.notify_all()
    reader.join(timeout=1)

    assert not reader.is_alive()
    assert states[0].battery_level == 42.5
    assert RobotInterface.image_timestamp(states[0].rgb) == 2.0


def test_robot_pairs_rgbd_within_tolerance_and_reports_larger_offsets():
    robot = RobotInterface.__new__(RobotInterface)
    robot._lock = threading.Lock()
    robot._ready = threading.Condition(robot._lock)
    robot._pending_rgb = {}
    robot._pending_depth = {}
    robot._rgb = None
    robot._depth = None
    robot._rgbd_sync_tolerance_ns = 20_000_000
    robot._closest_rgbd_offset_ns = None
    robot._rgb_message_count = 0
    robot._depth_message_count = 0
    robot._camera_info_message_count = 0
    robot._camera_info = None
    robot._imu = None
    robot._odom = None
    robot._battery_level = None

    def image(nanosec):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=nanosec))
        )

    robot._on_rgb(image(0))
    robot._on_depth(image(30_000_000))
    assert robot._rgb is None
    assert "closest RGB-D offset=30.0ms" in robot.sensor_diagnostics()

    robot._on_depth(image(10_000_000))
    assert robot._rgb is not None
    assert robot._depth is not None


def test_robot_bounds_unmatched_full_resolution_frame_queues():
    robot = RobotInterface.__new__(RobotInterface)
    robot._lock = threading.Lock()
    robot._ready = threading.Condition(robot._lock)
    robot._pending_rgb = {}
    robot._pending_depth = {}
    robot._rgb = None
    robot._depth = None
    robot._rgbd_sync_tolerance_ns = 1_000_000
    robot._closest_rgbd_offset_ns = None
    robot._rgb_message_count = 0

    for index in range(60):
        nanoseconds = index * 50_000_000
        robot._on_rgb(
            SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(
                        sec=nanoseconds // 1_000_000_000,
                        nanosec=nanoseconds % 1_000_000_000,
                    )
                )
            )
        )

    assert len(robot._pending_rgb) <= 21


def test_opencv_yolo_backend_decodes_coco_detection_with_depth():
    class FakeNet:
        def setInput(self, blob):
            assert blob.shape == (1, 3, 640, 640)

        def forward(self):
            output = np.zeros((1, 84, 1), dtype=np.float32)
            output[0, :4, 0] = [320, 320, 200, 100]
            output[0, 4 + COCO80.index("chair"), 0] = 0.9
            return output

    detector = ObjectDetector(confidence_threshold=0.5)
    detector._net = FakeNet()
    detections = detector.detect(
        np.zeros((640, 640, 3), dtype=np.uint8),
        np.full((640, 640), 1000, dtype=np.uint16),
    )
    assert len(COCO80) == 80
    assert len(detections) == 1
    assert detections[0].label == "chair"
    assert detections[0].bbox == (220, 270, 420, 370)
    assert detections[0].position_3d[2] == 1.0


def test_yolo_world_accepts_arbitrary_target_and_runs_asynchronously(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeWorld:
        def __init__(self, _):
            self.classes = []

        def set_classes(self, classes):
            self.classes = list(classes)

        def predict(self, *_args, **_kwargs):
            boxes = SimpleNamespace(
                xyxy=Tensor([[1, 1, 9, 9]]),
                conf=Tensor([0.87]),
                cls=Tensor([0]),
            )
            return [SimpleNamespace(boxes=boxes, names={0: self.classes[0]})]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOWorld=FakeWorld))
    thread_counts = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=thread_counts.append,
            set_num_interop_threads=lambda count: thread_counts.append(("interop", count)),
        ),
    )
    model_path = tmp_path / "world.pt"
    model_path.write_bytes(b"weights")
    detector = ObjectDetector(model_path=model_path)
    assert detector.initialize()
    assert thread_counts == [4, ("interop", 1)]
    assert detector.inference_size == 256
    detector.set_target("unusual brass umbrella stand")
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = np.full((10, 10), 1000, dtype=np.uint16)

    assert detector.detect(rgb, depth) == []
    for _ in range(100):
        detections = detector.detect(rgb, depth)
        if detector.result_revision:
            break
        time.sleep(0.001)

    assert detector.result_revision == 1
    assert detections[0].label == "unusual brass umbrella stand"
    assert detections[0].confidence == 0.87
    assert detections[0].position_3d[2] == 1.0
    detector.close()


def test_yolo_world_runtime_tuning_is_validated(tmp_path, monkeypatch):
    model_path = tmp_path / "world.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setenv("NERO_YOLO_IMGSZ", "448")
    monkeypatch.setenv("NERO_YOLO_MAX_DETECTIONS", "4")

    detector = ObjectDetector(model_path=model_path)
    assert detector.inference_size == 448
    assert detector.max_detections == 4

    with np.testing.assert_raises_regex(ValueError, "divisible by 32"):
        ObjectDetector(model_path=model_path, inference_size=400)
    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        ObjectDetector(model_path=model_path, max_detections=0)


def test_yoloe_cpu_backend_accepts_arbitrary_target_asynchronously(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    models = []

    class FakeYOLOE:
        def __init__(self, path):
            self.path = path
            self.classes = []
            models.append(self)

        def set_classes(self, classes):
            self.classes = list(classes)

        def predict(self, *_args, **kwargs):
            assert kwargs["device"] == "cpu"
            assert kwargs["imgsz"] == 320
            boxes = SimpleNamespace(
                xyxy=Tensor([[2, 2, 8, 8]]),
                conf=Tensor([0.91]),
                cls=Tensor([0]),
            )
            return [SimpleNamespace(boxes=boxes, names={0: self.classes[0]})]

    settings = {}
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(SETTINGS=settings, YOLOE=FakeYOLOE),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda _count: None),
    )
    model_path = tmp_path / "cpu-open-vocab.pt"
    model_path.write_bytes(b"weights")
    text_model_path = tmp_path / "mobileclip2_b.ts"
    text_model_path.write_bytes(b"text weights")
    detector = ObjectDetector(
        backend="yoloe",
        model_path=model_path,
        text_model_path=text_model_path,
    )

    assert detector.backend == "yoloe"
    assert detector.inference_size == 320
    assert detector.initialize()
    assert settings["weights_dir"] == str(tmp_path.resolve())
    assert models[0].path == str(model_path)
    detector.set_target("striped ceramic flower pot")

    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    assert detector.detect(rgb) == []
    for _ in range(100):
        detections = detector.detect(rgb)
        if detector.result_revision:
            break
        time.sleep(0.001)

    assert detector.result_revision == 1
    assert detections[0].label == "striped ceramic flower pot"
    assert detections[0].confidence == 0.91
    assert detections[0].position_3d is None
    detector.close()


def test_object_backend_configuration_selects_defaults(monkeypatch):
    monkeypatch.delenv("NERO_OBJECT_BACKEND", raising=False)
    default = ObjectDetector()
    assert default.backend == "yolo-world"
    assert default.model_path == Path("config/yolov8s-worldv2.pt")
    assert default.inference_size == 256

    monkeypatch.setenv("NERO_OBJECT_BACKEND", "yoloe")
    detector = ObjectDetector()
    assert detector.backend == "yoloe"
    assert detector.model_path == Path("config/yoloe-26n-seg.pt")
    assert detector.inference_size == 320

    monkeypatch.delenv("NERO_OBJECT_BACKEND")
    inferred = ObjectDetector(model_path="weights/custom-yoloe-26n.pt")
    assert inferred.backend == "yoloe"

    with np.testing.assert_raises_regex(ValueError, "unsupported object detector"):
        ObjectDetector(backend="not-a-backend")


def test_hardware_agent_clis_use_k1_sensors_implicitly(monkeypatch):
    from nero.agents import (
        booster_studio_agent,
        map_nav_agent,
        mapping_agent,
        orb_slam_agent,
        pure_pursuit_agent,
    )

    monkeypatch.setattr(sys, "argv", ["nero-orb-slam"])
    orb_args = orb_slam_agent.parse_args()
    assert not hasattr(orb_args, "camera")
    assert not hasattr(orb_args, "depth_camera")
    assert not hasattr(orb_args, "robot_serial")
    assert not hasattr(orb_args, "object")
    assert not hasattr(orb_args, "target_distance")
    assert orb_args.disable_safety is False

    monkeypatch.setattr(sys, "argv", ["nero-pure-pursuit"])
    pursuit_args = pure_pursuit_agent.parse_args()
    assert not hasattr(pursuit_args, "camera")
    assert not hasattr(pursuit_args, "depth_camera")
    assert not hasattr(pursuit_args, "object")
    assert not hasattr(pursuit_args, "target_distance")
    assert pursuit_args.disable_safety is False

    monkeypatch.setattr(sys, "argv", ["nero-mapping"])
    mapping_args = mapping_agent.parse_args()
    assert not hasattr(mapping_args, "camera")
    assert not hasattr(mapping_args, "depth_camera")
    assert not hasattr(mapping_args, "robot_serial")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nero-map-nav",
            "--map",
            "map.npy",
            "--initial-pose",
            "0",
            "0",
            "0",
        ],
    )
    map_nav_args = map_nav_agent.parse_args()
    assert not hasattr(map_nav_args, "camera")
    assert not hasattr(map_nav_args, "depth_camera")
    assert not hasattr(map_nav_args, "robot_serial")
    assert map_nav_args.disable_safety is False

    monkeypatch.setattr(sys, "argv", ["nero-booster-studio"])
    studio_args = booster_studio_agent.parse_args()
    assert not hasattr(studio_args, "object")
    assert not hasattr(studio_args, "target_distance")


def test_hardware_navigation_clis_accept_explicit_safety_opt_out(monkeypatch):
    from nero.agents import map_nav_agent, orb_slam_agent, pure_pursuit_agent

    monkeypatch.setattr(sys, "argv", ["nero-orb-slam", "--disable-safety"])
    assert orb_slam_agent.parse_args().disable_safety is True

    monkeypatch.setattr(sys, "argv", ["nero-pure-pursuit", "--disable-safety"])
    assert pure_pursuit_agent.parse_args().disable_safety is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["nero-map-nav", "--map", "map.npy", "--disable-safety"],
    )
    assert map_nav_agent.parse_args().disable_safety is True


def test_go_to_command_parser_accepts_natural_object_names():
    assert parse_go_to_command("go to chair") == "chair"
    assert parse_go_to_command("Go to the red chair, please!") == "red chair"
    assert parse_go_to_command("please go to a fire extinguisher") == "fire extinguisher"
    assert parse_go_to_command("chair detected") is None
    assert parse_go_to_command("follow the chair") is None
    assert parse_go_to_command("go to") is None


def test_terminal_target_normalizes_a_missing_to_typo():
    from nero.interaction import _parse_bare_object_name

    assert _parse_bare_object_name("go the green can") == "green can"
    assert _parse_bare_object_name("please go to a fire extinguisher") == "fire extinguisher"


def test_direction_acknowledges_target_without_detection_confirmation():
    spoken = []
    events = []
    speaker = SimpleNamespace(speak=lambda text: (events.append("speak"), spoken.append(text)))
    responses = iter(["what can you see?", "go to", "go to the chair"])
    commands = SimpleNamespace(
        start_listening=lambda: events.append("start"),
        read_command=lambda _: next(responses),
        stop_listening=lambda: events.append("stop"),
    )

    assert request_navigation_target(speaker, commands) == "chair"
    assert spoken == ["Going to the chair."]
    assert events[:3] == ["start", "stop", "speak"]


def test_direction_rejects_unsupported_fixed_vocabulary_target():
    responses = iter(["fire extinguisher", "sofa"])
    acknowledgements = []
    commands = SimpleNamespace(
        accepts_bare_object_names=True,
        start_listening=lambda: None,
        read_command=lambda _: next(responses),
        acknowledge=acknowledgements.append,
        stop_listening=lambda: None,
    )

    target = request_navigation_target(
        SimpleNamespace(speak=lambda _: None),
        commands,
        target_validator=lambda name: name == "sofa",
    )

    assert target == "sofa"
    assert acknowledgements == ["unsupported", "accepted"]


def test_terminal_accepts_a_bare_object_name(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (prompts.append(prompt), "the red chair")[1],
    )

    assert (
        request_navigation_target(SimpleNamespace(speak=lambda _: None), TerminalCommandSource())
        == "red chair"
    )
    assert prompts == ["Object to follow (for example, 'chair'): "]


def test_unix_socket_command_source_accepts_object_and_is_private():
    socket_path = f"/tmp/nero-test-{os.getpid()}-{time.time_ns()}.sock"
    commands = UnixSocketCommandSource(socket_path)
    commands.start_listening()
    assert stat.S_IMODE(os.stat(socket_path).st_mode) == 0o600

    result = []
    listener = threading.Thread(
        target=lambda: result.append(
            request_navigation_target(SimpleNamespace(speak=lambda _: None), commands)
        )
    )
    listener.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    client.sendall(b"red chair\n")
    assert client.recv(128) == b"accepted\n"
    listener.join(timeout=2)
    client.close()
    commands.close()

    assert result == ["red chair"]
    assert not os.path.exists(socket_path)


def test_unix_socket_rejects_invalid_commands_and_closes_admission_while_busy():
    socket_path = f"/tmp/nero-admission-{os.getpid()}-{time.time_ns()}.sock"
    commands = UnixSocketCommandSource(socket_path)
    result = []
    listener = threading.Thread(
        target=lambda: result.append(
            request_navigation_target(SimpleNamespace(speak=lambda _: None), commands)
        )
    )
    listener.start()

    deadline = time.monotonic() + 1.0
    while not os.path.exists(socket_path) and time.monotonic() < deadline:
        time.sleep(0.001)

    invalid = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    invalid.connect(socket_path)
    invalid.sendall(b"go\n")
    assert invalid.recv(128) == b"rejected\n"
    invalid.close()

    valid = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    valid.connect(socket_path)
    valid.sendall(b"chair\n")
    assert valid.recv(128) == b"accepted\n"
    valid.close()
    listener.join(timeout=2)

    assert result == ["chair"]
    assert not os.path.exists(socket_path)
    queued = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with np.testing.assert_raises(OSError):
        queued.connect(socket_path)
    queued.close()

    # A later listening session starts with a new socket and no stale backlog.
    commands.start_listening()
    assert os.path.exists(socket_path)
    commands.stop_listening()
    commands.close()
    assert not os.path.exists(socket_path)


def test_unavailable_speaker_does_not_discard_terminal_target(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "chair")

    def unavailable_speaker(_: str) -> None:
        raise RuntimeError("LUI TTS unavailable")

    assert (
        request_navigation_target(
            SimpleNamespace(speak=unavailable_speaker), TerminalCommandSource()
        )
        == "chair"
    )


def test_direction_wait_can_be_cancelled_cleanly():
    events = []
    commands = SimpleNamespace(
        start_listening=lambda: events.append("start"),
        read_command=lambda _: "",
        stop_listening=lambda: events.append("stop"),
    )

    with np.testing.assert_raises(InterruptedError):
        request_navigation_target(
            SimpleNamespace(speak=lambda _: None),
            commands,
            cancelled=lambda: True,
        )
    assert events == ["start", "stop"]


def test_direction_parser_combines_split_asr_chunks():
    speaker = SimpleNamespace(speak=lambda _: None)
    responses = iter(["go to", "the coffee table"])
    commands = SimpleNamespace(
        start_listening=lambda: None,
        read_command=lambda _: next(responses),
        stop_listening=lambda: None,
    )

    assert request_navigation_target(speaker, commands) == "coffee table"


def test_navigation_target_listener_does_not_block_sensor_loop():
    released = threading.Event()
    commands = SimpleNamespace(
        start_listening=lambda: None,
        read_command=lambda _: released.wait(timeout=1.0) and "go to the chair",
        stop_listening=lambda: None,
        close=lambda: None,
    )
    listener = NavigationTargetListener(SimpleNamespace(speak=lambda _: None), commands)
    listener.start()
    assert listener.poll() is None
    released.set()
    for _ in range(100):
        target = listener.poll()
        if target is not None:
            break
        time.sleep(0.001)
    assert target == "chair"


def test_k1_voice_source_uses_official_lui_asr(monkeypatch):
    calls = []

    class FakeFactory:
        @classmethod
        def Instance(cls):
            return cls()

        def Init(self, domain, interface):
            calls.append(("channel_init", domain, interface))

    class FakeClient:
        def Init(self):
            calls.append("client_init")

        def StartAsr(self):
            calls.append("start_asr")

        def StopAsr(self):
            calls.append("stop_asr")

    class FakeSubscriber:
        def __init__(self, callback):
            self.callback = callback

        def InitChannel(self):
            calls.append("subscriber_init")

        def CloseChannel(self):
            calls.append("subscriber_close")

    sdk = SimpleNamespace(
        ChannelFactory=FakeFactory,
        LuiClient=FakeClient,
        LuiAsrChunkSubscriber=FakeSubscriber,
    )
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)

    source = K1VoiceCommandSource()
    source.start_listening()
    source._subscriber.callback(SimpleNamespace(text="go to the chair"))
    assert source.read_command("direction: ") == "go to the chair"
    source.stop_listening()
    source.close()
    source.close()
    assert calls == [
        ("channel_init", 0, "lo"),
        "client_init",
        "subscriber_init",
        "start_asr",
        "stop_asr",
        "subscriber_close",
    ]


def test_stand_off_is_internal_and_independent_of_initial_range():
    assert safe_stand_off_distance("chair") == 1.0
    assert safe_stand_off_distance("bottle") == 0.7
    assert safe_stand_off_distance("unknown") == 0.8


def test_real_agent_exits_immediately_on_terminal_policy_error(monkeypatch):
    import nero.agents.orb_slam_agent as agent

    events = []

    class FailedPolicy:
        last_sensor = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("start")

        def step(self):
            events.append("step")
            return SimpleNamespace(
                state=agent.PolicyState.ERROR,
                message="camera timed out",
            )

        def stop(self):
            events.append("policy stop")

    class Listener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("listen")

        def poll(self):
            return None

        def close(self):
            events.append("listener close")

    monkeypatch.setattr(agent, "NavigationPolicy", FailedPolicy)
    monkeypatch.setattr(agent, "NavigationTargetListener", Listener)
    monkeypatch.setattr(agent.signal, "signal", lambda *args: None)
    monkeypatch.setattr(agent.RosObservabilityPublisher, "try_create", lambda **kwargs: None)
    robot = SimpleNamespace(stop=lambda: events.append("robot stop"))

    agent.run_agent(
        robot,
        SimpleNamespace(no_ros_observability=True, no_display=True),
        command_source=SimpleNamespace(),
    )

    assert events == [
        "start",
        "listen",
        "step",
        "policy stop",
        "robot stop",
        "listener close",
    ]


def test_real_agent_announces_missing_object_once_per_command(monkeypatch):
    import nero.agents.orb_slam_agent as agent

    spoken = []

    class LostPolicy:
        slam = None
        map_navigator = None
        last_sensor = SimpleNamespace(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            timestamp=1.0,
            raw_state=None,
        )

        def __init__(self, **kwargs):
            self.calls = 0

        def start(self):
            pass

        def set_target(self, target):
            self.target = target

        def step(self):
            self.calls += 1
            state = agent.PolicyState.LOST if self.calls == 1 else agent.PolicyState.ERROR
            return SimpleNamespace(
                state=state,
                message="not found",
                velocity_command=None,
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

    monkeypatch.setattr(agent, "NavigationPolicy", LostPolicy)
    monkeypatch.setattr(agent, "NavigationTargetListener", Listener)
    monkeypatch.setattr(agent.signal, "signal", lambda *args: None)
    monkeypatch.setattr(agent.RosObservabilityPublisher, "try_create", lambda **kwargs: None)
    monkeypatch.setattr(
        agent.Visualization, "draw_navigation_info", lambda self, frame, **kwargs: frame
    )
    robot = SimpleNamespace(stop=lambda: None, speak=spoken.append)

    agent.run_agent(
        robot,
        SimpleNamespace(no_ros_observability=True, no_display=True),
        command_source=SimpleNamespace(),
    )

    assert spoken == ["I could not detect the green cup."]


def test_robot_speak_uses_booster_speaker_service():
    spoken = []
    robot = RobotInterface.__new__(RobotInterface)
    robot._robot = SimpleNamespace(speaker=SimpleNamespace(synthesize=spoken.append))
    robot.speak("chair detected")
    assert spoken == ["chair detected"]


def test_robot_speak_falls_back_to_flite_when_lui_is_unavailable(monkeypatch):
    class FailingLui:
        def StartTts(self, _):
            raise RuntimeError("service unavailable")

    class TtsParameter:
        text = ""

    monkeypatch.setitem(
        sys.modules,
        "booster_robotics_sdk_python",
        SimpleNamespace(
            LuiTtsConfig=lambda: object(),
            LuiTtsParameter=TtsParameter,
        ),
    )
    calls = []
    monkeypatch.setattr(
        "nero.robot.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    robot = RobotInterface.__new__(RobotInterface)
    robot._lui = FailingLui()
    robot._lui_tts_failed = False

    robot.speak("Going to the chair.")
    robot.speak("Arrived.")

    assert robot._lui_tts_failed is True
    assert len(calls) == 4
    assert calls[0][0][:4] == ["flite", "-t", "Going to the chair.", "-o"]
    assert calls[1][0][:3] == ["aplay", "-D", "plughw:0,0"]
    assert calls[2][0][:4] == ["flite", "-t", "Arrived.", "-o"]
    assert calls[3][0][:3] == ["aplay", "-D", "plughw:0,0"]
    assert all(kwargs == {"check": True, "timeout": 30} for _, kwargs in calls)
