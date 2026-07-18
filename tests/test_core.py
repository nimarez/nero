import math
import sys
from types import SimpleNamespace

import numpy as np

import nero.agents as agents
from nero.navigation.controller import VelocityController
from nero.perception.object_detector import ObjectDetection, ObjectDetector
from nero.robot import RobotInterface
from nero.interaction import announce_and_confirm, deduce_target_distance


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

    monkeypatch.setattr(sys, "argv", ["nero-map-nav", "--map", "map.npy"])
    map_nav_args = map_nav_agent.parse_args()
    assert not hasattr(map_nav_args, "camera")
    assert not hasattr(map_nav_args, "depth_camera")
    assert not hasattr(map_nav_args, "robot_serial")

    monkeypatch.setattr(sys, "argv", ["nero-booster-studio"])
    studio_args = booster_studio_agent.parse_args()
    assert not hasattr(studio_args, "object")
    assert not hasattr(studio_args, "target_distance")


def test_detection_announcement_requires_explicit_confirmation():
    spoken = []
    speaker = SimpleNamespace(speak=spoken.append)

    assert announce_and_confirm(speaker, "chair", lambda _: "yes")
    assert not announce_and_confirm(speaker, "bottle", lambda _: "no")
    assert spoken == [
        "chair detected. Should I follow it?",
        "bottle detected. Should I follow it?",
    ]


def test_target_distance_is_deduced_internally():
    assert deduce_target_distance("chair", 4.0) == 2.0
    assert deduce_target_distance("bottle", 4.0) == 1.2
    assert deduce_target_distance("unknown", 1.0) == 0.8


def test_robot_speak_uses_booster_speaker_service():
    spoken = []
    robot = RobotInterface.__new__(RobotInterface)
    robot._robot = SimpleNamespace(speaker=SimpleNamespace(synthesize=spoken.append))
    robot.speak("chair detected")
    assert spoken == ["chair detected"]
