from types import SimpleNamespace

import numpy as np
import pytest

from nero.observability.rerun_bridge import (
    RerunRosBridge,
    image_message_to_array,
    pointcloud2_to_xyz,
)
from nero.observability.ros_publisher import _point_cloud_xyz, _seconds_to_stamp
from nero.observability.topics import ObservabilityTopics


def test_observability_topics_are_stable_and_namespaced():
    topics = ObservabilityTopics()
    assert topics.rgb == "/nero/sensors/rgb"
    assert topics.pose == "/nero/slam/pose"
    assert topics.command == "/nero/navigation/cmd_vel"
    assert topics.goal_pose == "/nero/navigation/goal_pose"
    assert topics.object_position == "/nero/navigation/object_position"
    assert topics.reference_map == "/nero/reference/map_points"


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
        )
    )
    position = SimpleNamespace(x=1.0, y=2.0, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    pose = SimpleNamespace(position=position, orientation=orientation)
    pose_stamped = SimpleNamespace(header=header, pose=pose)
    bridge._on_pose(pose_stamped)
    bridge._on_reference_pose(pose_stamped)
    bridge._on_goal_pose(pose_stamped)
    bridge._on_object_position(SimpleNamespace(header=header, point=position))
    bridge._on_path(SimpleNamespace(header=header, poses=[pose_stamped, pose_stamped]))
    bridge._on_reference_path(
        SimpleNamespace(header=header, poses=[pose_stamped, pose_stamped])
    )

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

    hypothesis = SimpleNamespace(class_id="chair")
    result = SimpleNamespace(hypothesis=hypothesis)
    center = SimpleNamespace(position=SimpleNamespace(x=1.0, y=1.0))
    box = SimpleNamespace(center=center, size_x=1.0, size_y=1.0)
    detection = SimpleNamespace(bbox=box, results=[result])
    bridge._on_detections(SimpleNamespace(header=header, detections=[detection]))
    bridge._on_status(SimpleNamespace(data='{"state":"navigating","message":"ok"}'))
    bridge._on_tracking(SimpleNamespace(data='{"status":"OK","map_points":1}'))
    bridge._on_command(SimpleNamespace(linear=vector, angular=SimpleNamespace(z=0.1)))

    recording.flush()
    assert len(memory.drain_as_bytes()) > 1_000
