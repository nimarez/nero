import math

import numpy as np

import nero.agents as agents
from nero.navigation.controller import VelocityController
from nero.perception.object_detector import ObjectDetection, ObjectDetector


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
