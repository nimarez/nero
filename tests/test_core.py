import math
import sys
import threading
import time
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

    turn_in_place = controller.compute_goal_velocity(
        np.zeros(3), np.array([0.1, 0.0, 1.0])
    )
    assert turn_in_place.linear_x == 0.0
    assert turn_in_place.angular_z > 0.0
    assert not controller.has_reached_pose(np.zeros(3), np.array([0.1, 0.0, 1.0]))

    reverse = controller.compute_avoidance_velocity(
        {"has_obstacle": True, "min_distance": 0.2}
    )
    assert reverse.linear_x == -0.1


def test_robot_image_helpers_normalize_k1_images():
    image = np.arange(12).reshape(2, 2, 3)
    np.testing.assert_array_equal(
        RobotInterface.image_to_array(SimpleNamespace(data=image)), image
    )
    np.testing.assert_array_equal(RobotInterface.image_to_array(image), image)
    stamped = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000))
    )
    assert RobotInterface.image_timestamp(stamped) == 12.5


def test_robot_image_helpers_decode_production_k1_encodings():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    depth_message = SimpleNamespace(
        encoding="mono16", height=3, width=4, data=depth.tobytes()
    )
    np.testing.assert_array_equal(RobotInterface.image_to_array(depth_message), depth)

    # Neutral NV12 encodes a gray image and exercises the K1's exact wire layout.
    nv12 = np.concatenate(
        [np.full(4 * 4, 128, np.uint8), np.full(4 * 2, 128, np.uint8)]
    )
    rgb_message = SimpleNamespace(
        encoding="nv12", height=4, width=4, data=nv12.tobytes()
    )
    decoded = RobotInterface.image_to_array(rgb_message)
    assert decoded.shape == (4, 4, 3)
    assert decoded.dtype == np.uint8
    assert np.max(decoded) - np.min(decoded) == 0


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


def test_hardware_agent_clis_use_k1_sensors_implicitly(monkeypatch):
    from nero.agents import (
        booster_studio_agent,
        map_nav_agent,
        mapping_agent,
        orb_slam_agent,
    )

    monkeypatch.setattr(sys, "argv", ["nero-orb-slam"])
    orb_args = orb_slam_agent.parse_args()
    assert not hasattr(orb_args, "camera")
    assert not hasattr(orb_args, "depth_camera")
    assert not hasattr(orb_args, "robot_serial")
    assert not hasattr(orb_args, "object")
    assert not hasattr(orb_args, "target_distance")

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

    monkeypatch.setattr(sys, "argv", ["nero-booster-studio"])
    studio_args = booster_studio_agent.parse_args()
    assert not hasattr(studio_args, "object")
    assert not hasattr(studio_args, "target_distance")


def test_go_to_command_parser_accepts_natural_object_names():
    assert parse_go_to_command("go to chair") == "chair"
    assert parse_go_to_command("Go to the red chair, please!") == "red chair"
    assert (
        parse_go_to_command("please go to a fire extinguisher") == "fire extinguisher"
    )
    assert parse_go_to_command("chair detected") is None
    assert parse_go_to_command("follow the chair") is None
    assert parse_go_to_command("go to") is None


def test_direction_acknowledges_target_without_detection_confirmation():
    spoken = []
    events = []
    speaker = SimpleNamespace(
        speak=lambda text: (events.append("speak"), spoken.append(text))
    )
    responses = iter(["what can you see?", "go to", "go to the chair"])
    commands = SimpleNamespace(
        start_listening=lambda: events.append("start"),
        read_command=lambda _: next(responses),
        stop_listening=lambda: events.append("stop"),
    )

    assert request_navigation_target(speaker, commands) == "chair"
    assert spoken == ["Going to the chair."]
    assert events[:3] == ["start", "stop", "speak"]


def test_terminal_accepts_a_bare_object_name(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (prompts.append(prompt), "the red chair")[1],
    )

    assert request_navigation_target(
        SimpleNamespace(speak=lambda _: None), TerminalCommandSource()
    ) == "red chair"
    assert prompts == ["Object to follow (for example, 'chair'): "]


def test_unavailable_speaker_does_not_discard_terminal_target(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "chair")

    def unavailable_speaker(_: str) -> None:
        raise RuntimeError("LUI TTS unavailable")

    assert request_navigation_target(
        SimpleNamespace(speak=unavailable_speaker), TerminalCommandSource()
    ) == "chair"


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
    listener = NavigationTargetListener(
        SimpleNamespace(speak=lambda _: None), commands
    )
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
