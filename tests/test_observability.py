from types import SimpleNamespace
import sys

import numpy as np
import pytest

from nero.observability.rerun_bridge import (
    RerunRosBridge,
    image_message_to_array,
    main as rerun_main,
    navigation_geometry_primitives,
    pointcloud2_to_xyz,
)
from nero.observability.ros_publisher import (
    RosObservabilityPublisher,
    _point_cloud_xyz,
    _seconds_to_stamp,
    navigation_geometry_payload,
)
from nero.observability.topics import ObservabilityTopics


def test_observability_topics_are_stable_and_namespaced():
    topics = ObservabilityTopics()
    assert topics.rgb == "/nero/sensors/rgb"
    assert topics.odometry == "/nero/sensors/odometry"
    assert topics.joint_states == "/nero/sensors/joint_states"
    assert topics.pose == "/nero/slam/pose"
    assert topics.command == "/nero/navigation/cmd_vel"
    assert topics.plan == "/nero/navigation/plan"
    assert topics.goal_pose == "/nero/navigation/goal_pose"
    assert topics.object_position == "/nero/navigation/object_position"
    assert topics.reference_map == "/nero/reference/map_points"
    values = list(vars(topics).values())
    assert len(values) == len(set(values))
    assert all(topic.startswith("/nero/") for topic in values)


def test_rerun_topic_contract_is_printable_without_ros_or_rerun(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["nero-rerun", "--print-topics"])
    rerun_main()
    output = capsys.readouterr().out
    assert "rgb: /nero/sensors/rgb" in output
    assert "odometry: /nero/sensors/odometry" in output
    assert "joint_states: /nero/sensors/joint_states" in output


def test_pure_pursuit_safety_geometry_is_camera_frame_and_color_coded():
    status = SimpleNamespace(
        state=SimpleNamespace(value="navigating"),
        target_position_camera=[0.2, 0.1, 2.0],
        stand_off_distance=0.8,
        stand_off_tolerance=0.12,
    )

    payload = navigation_geometry_payload(status)
    geometry = navigation_geometry_primitives(payload)

    assert payload["frame"] == "camera"
    assert geometry["root"].startswith("world/robot/camera/")
    assert geometry["condition"] == "approaching"
    assert geometry["color"] == [0, 200, 255]
    assert geometry["circle"].shape == (65, 3)
    radii = np.linalg.norm(
        geometry["circle"][:, [0, 2]] - np.asarray(payload["center"])[[0, 2]],
        axis=1,
    )
    np.testing.assert_allclose(radii, 0.8)


def test_slam_safety_geometry_uses_world_object_and_approach_pose():
    goal = SimpleNamespace(
        object_name="chair",
        object_position_world=np.array([2.0, 3.0, 0.7]),
        stand_off_distance=1.0,
        approach_pose=np.array([1.0, 3.0, 0.0]),
    )
    status = SimpleNamespace(
        state=SimpleNamespace(value="navigating"),
        current_goal=goal,
        current_pose=SimpleNamespace(position=np.array([0.0, 3.0, 0.0])),
    )

    payload = navigation_geometry_payload(status)
    geometry = navigation_geometry_primitives(payload)

    assert payload["frame"] == "map"
    assert geometry["root"] == "world/navigation/safety_geometry"
    np.testing.assert_allclose(geometry["approach"], [1.0, 3.0, 0.03])
    assert np.allclose(geometry["circle"][:, 2], 0.03)
    assert geometry["condition"] == "approaching"


@pytest.mark.parametrize(
    ("distance", "condition", "color"),
    [
        (0.5, "inside radius", [255, 60, 60]),
        (0.8, "holding radius", [80, 255, 80]),
        (1.5, "approaching", [0, 200, 255]),
    ],
)
def test_safety_radius_state_colors(distance, condition, color):
    geometry = navigation_geometry_primitives(
        {
            "frame": "camera",
            "center": [0.0, 0.0, distance],
            "radius": 0.8,
            "tolerance": 0.12,
        }
    )

    assert geometry["condition"] == condition
    assert geometry["color"] == color


def test_odometry_and_joint_callbacks_log_sensor_metrics():
    class FakeRerun:
        Scalar = float

    class FakeRecording:
        def __init__(self):
            self.entities = []

        def set_time_nanos(self, timeline, timestamp):
            pass

        def log(self, entity, value):
            self.entities.append(entity)

    bridge = RerunRosBridge.__new__(RerunRosBridge)
    bridge._rr = FakeRerun()
    bridge._recording = FakeRecording()
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    position = SimpleNamespace(x=1.0, y=2.0, z=0.0)
    bridge._on_odometry(
        SimpleNamespace(
            header=header,
            pose=SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation)),
        )
    )
    bridge._on_joint_states(
        SimpleNamespace(
            header=header,
            name=["left/hip"],
            position=[0.1],
            velocity=[0.2],
            effort=[0.3],
        )
    )
    assert "metrics/odometry/x" in bridge._recording.entities
    assert "metrics/odometry/yaw" in bridge._recording.entities
    assert "metrics/joints/left_hip/position" in bridge._recording.entities
    assert "metrics/joints/left_hip/velocity" in bridge._recording.entities


def test_robot_state_publishes_normalized_odometry_and_joints():
    def header():
        return SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")

    def vector():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def image_message():
        return SimpleNamespace(header=header())

    def imu_message():
        return SimpleNamespace(
            header=header(),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            angular_velocity=vector(),
            linear_acceleration=vector(),
        )

    def odometry_message():
        pose = SimpleNamespace(
            position=vector(),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        return SimpleNamespace(header=header(), child_frame_id="", pose=SimpleNamespace(pose=pose))

    def joint_message():
        return SimpleNamespace(header=header(), name=[], position=[], velocity=[], effort=[])

    class Capture:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    captures = {name: Capture() for name in ("rgb", "depth", "imu", "odometry", "joint_states")}
    publisher = RosObservabilityPublisher.__new__(RosObservabilityPublisher)
    publisher._last_sensor_timestamp = None
    publisher._camera_frame = "camera"
    publisher._types = {
        "Image": image_message,
        "Imu": imu_message,
        "Odometry": odometry_message,
        "JointState": joint_message,
    }
    publisher._publishers = captures

    raw_joints = SimpleNamespace(name=["left_hip"], position=[0.1], velocity=[0.2], effort=[0.3])
    state = SimpleNamespace(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.uint16),
        camera_info=None,
        imu=object(),
        odom=object(),
        joints=raw_joints,
        orientation_rpy=np.array([0.0, 0.0, 0.4]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
        linear_acceleration=np.array([1.0, 2.0, 3.0]),
        position_2d=np.array([4.0, 5.0, 0.6]),
    )
    robot = SimpleNamespace(
        image_timestamp=lambda image: 12.5,
        image_to_array=lambda image: image,
    )
    publisher.publish_robot_state(state, robot)

    odometry = captures["odometry"].messages[0]
    assert odometry.header.frame_id == "odom"
    assert odometry.child_frame_id == "base_link"
    assert odometry.pose.pose.position.x == 4.0
    assert odometry.pose.pose.position.y == 5.0
    joints = captures["joint_states"].messages[0]
    assert joints.name == ["left_hip"]
    assert joints.position == [0.1]


def test_seconds_to_stamp_normalizes_rounding():
    stamp = SimpleNamespace(sec=0, nanosec=0)
    _seconds_to_stamp(stamp, 12.9999999999)
    assert (stamp.sec, stamp.nanosec) == (13, 0)


def test_point_cloud_round_trip():
    points = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    message = SimpleNamespace(
        fields=[
            SimpleNamespace(name="x", offset=0),
            SimpleNamespace(name="y", offset=4),
            SimpleNamespace(name="z", offset=8),
        ]
    )
    _point_cloud_xyz(message, points)
    restored = pointcloud2_to_xyz(message)
    np.testing.assert_allclose(restored, points)


def test_ros_bgr_image_becomes_rerun_rgb():
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    message = SimpleNamespace(
        encoding="bgr8",
        data=bgr.tobytes(),
        height=1,
        width=2,
        is_bigendian=False,
    )
    rgb = image_message_to_array(message)
    np.testing.assert_array_equal(rgb, bgr[..., ::-1])


def test_depth_image_preserves_uint16_millimetres():
    depth = np.array([[500, 1000], [2000, 6000]], dtype=np.uint16)
    message = SimpleNamespace(
        encoding="16UC1",
        data=depth.tobytes(),
        height=2,
        width=2,
        is_bigendian=False,
    )
    restored = image_message_to_array(message)
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, depth)


def test_detection_telemetry_uses_firmware_safe_json():
    published = []
    publisher = RosObservabilityPublisher.__new__(RosObservabilityPublisher)
    publisher._types = {"String": lambda: SimpleNamespace(data="")}
    publisher._publishers = {
        "detections": SimpleNamespace(publish=published.append)
    }
    detection = SimpleNamespace(
        label="unusual brass umbrella stand",
        confidence=0.87,
        bbox=(1, 2, 11, 22),
        position_3d=np.array([0.1, 0.2, 1.3]),
        distance=1.32,
        coordinate_frame="camera",
    )

    publisher.publish_detections([detection], 12.5)

    payload = __import__("json").loads(published[0].data)
    assert payload["timestamp"] == 12.5
    assert payload["detections"][0]["label"] == "unusual brass umbrella stand"
    assert payload["detections"][0]["bbox"] == [1, 2, 11, 22]
    assert payload["detections"][0]["position_3d"] == [0.1, 0.2, 1.3]


def test_policy_plan_is_published_in_map_frame_and_can_be_cleared():
    def header():
        return SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")

    def pose_stamped():
        return SimpleNamespace(
            header=header(),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            ),
        )

    captured = []
    publisher = RosObservabilityPublisher.__new__(RosObservabilityPublisher)
    publisher._types = {
        "Path": lambda: SimpleNamespace(header=header(), poses=[]),
        "PoseStamped": pose_stamped,
    }
    publisher._map_frame = "map"
    publisher._publishers = {"plan": SimpleNamespace(publish=captured.append)}

    publisher.publish_plan(np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]), 12.5)
    publisher.publish_plan(np.empty((0, 3)), 13.0)

    assert captured[0].header.frame_id == "map"
    assert [(p.pose.position.x, p.pose.position.y) for p in captured[0].poses] == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]
    assert captured[1].header.frame_id == "map"
    assert captured[1].poses == []


def test_rerun_callbacks_create_a_real_recording():
    rr = pytest.importorskip("rerun")
    recording = rr.new_recording("nero_test", make_default=False)
    memory = rr.memory_recording(recording)
    bridge = RerunRosBridge.__new__(RerunRosBridge)
    bridge._rr = rr
    bridge._recording = recording

    stamp = SimpleNamespace(sec=1, nanosec=50)
    header = SimpleNamespace(stamp=stamp)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    bridge._on_rgb(
        SimpleNamespace(
            header=header,
            encoding="rgb8",
            data=rgb.tobytes(),
            height=2,
            width=3,
            is_bigendian=False,
        )
    )
    bridge._on_depth(
        SimpleNamespace(
            header=header,
            encoding="16UC1",
            data=depth.tobytes(),
            height=2,
            width=3,
            is_bigendian=False,
        )
    )
    bridge._on_camera_info(
        SimpleNamespace(
            header=header,
            width=3,
            height=2,
            k=[2.0, 0.0, 1.5, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        )
    )
    vector = SimpleNamespace(x=0.1, y=0.2, z=0.3)
    bridge._on_imu(
        SimpleNamespace(
            header=header,
            angular_velocity=vector,
            linear_acceleration=vector,
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    position = SimpleNamespace(x=1.0, y=2.0, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    pose = SimpleNamespace(position=position, orientation=orientation)
    pose_stamped = SimpleNamespace(header=header, pose=pose)
    bridge._on_odometry(SimpleNamespace(header=header, pose=SimpleNamespace(pose=pose)))
    bridge._on_joint_states(
        SimpleNamespace(
            header=header,
            name=["left_hip"],
            position=[0.1],
            velocity=[0.2],
            effort=[0.3],
        )
    )
    bridge._on_pose(pose_stamped)
    bridge._on_reference_pose(pose_stamped)
    bridge._on_goal_pose(pose_stamped)
    bridge._on_object_position(SimpleNamespace(header=header, point=position))
    bridge._on_path(SimpleNamespace(header=header, poses=[pose_stamped, pose_stamped]))
    bridge._on_plan(SimpleNamespace(header=header, poses=[pose_stamped, pose_stamped]))
    bridge._on_reference_path(SimpleNamespace(header=header, poses=[pose_stamped, pose_stamped]))

    cloud = SimpleNamespace(
        header=header,
        fields=[
            SimpleNamespace(name="x", offset=0),
            SimpleNamespace(name="y", offset=4),
            SimpleNamespace(name="z", offset=8),
        ],
    )
    _point_cloud_xyz(cloud, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
    bridge._on_map(cloud)
    bridge._on_reference_map(cloud)

    bridge._on_detections(
        SimpleNamespace(
            data=(
                '{"timestamp":1.0,"detections":[{"label":"chair",'
                '"confidence":0.9,"bbox":[0,0,2,2],'
                '"position_3d":[0.0,0.0,1.0],"distance":1.0,'
                '"coordinate_frame":"camera"}]}'
            )
        )
    )
    bridge._on_status(
        SimpleNamespace(
            data=(
                '{"state":"navigating","message":"ok","navigation_geometry":'
                '{"frame":"camera","center":[0.0,0.0,1.5],'
                '"radius":0.8,"tolerance":0.12,"robot":[0.0,0.0,0.0]}}'
            )
        )
    )
    bridge._on_tracking(SimpleNamespace(data='{"status":"OK","map_points":1}'))
    bridge._on_command(SimpleNamespace(linear=vector, angular=SimpleNamespace(z=0.1)))

    recording.flush()
    assert len(memory.drain_as_bytes()) > 1_000
