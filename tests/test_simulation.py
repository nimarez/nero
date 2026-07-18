import time
import threading

import numpy as np
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from nero.navigation.policy import NavigationPolicy, PolicyState
from nero.simulation.booster_studio import (
    BoosterStudioRobotInterface,
    BoosterStudioTopics,
    write_booster_studio_calibration,
)
from nero.simulation.environment import SimEnvironment
from nero.simulation.mock_robot import MockRobot
from nero.simulation.sim_camera import CameraMode, SimCamera
from nero.simulation.booster_studio_room import (
    BACKUP_SUFFIX,
    SCENE_DIR,
    configure_ros_transport,
    find_sim_root,
    install_room,
    restore_default_scene,
)


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


def test_booster_studio_topics_match_installed_k1_simulator():
    topics = BoosterStudioTopics()
    assert topics.rgb == "/rgbd_camera/rgb/image_compressed"
    assert topics.depth == "/rgbd_camera/depth/image_raw"
    assert topics.imu is None
    assert BoosterStudioTopics.IMU_CANDIDATES == (
        "/imu/data",
        "/booster/ros2_k2_imu/robot1",
    )
    assert topics.pose == "/soccer/sim/localization/robot_pose"
    assert topics.detections == "/soccer/sim/vision/detections"


def test_booster_studio_calibration_uses_live_intrinsics(tmp_path):
    camera_info = SimpleNamespace(
        header=SimpleNamespace(frame_id="/rgbd_camera_frame"),
        width=320,
        height=240,
        k=np.array([[216.5, 0, 160], [0, 216.5, 120], [0, 0, 1]]),
        d=[0.0] * 5,
    )
    output = tmp_path / "calibration.json"
    calibration = write_booster_studio_calibration(camera_info, output)
    assert output.is_file()
    assert calibration.camera_rgb is False
    assert calibration.imu_frame == "imu_link"
    assert calibration.camera_matrix[0] == 216.5
    np.testing.assert_allclose(np.asarray(calibration.tbc)[:3, 3], [0.0669, 0, 0.3559])


def test_booster_studio_image_helpers():
    stamp = SimpleNamespace(sec=4, nanosec=250_000_000)
    image = SimpleNamespace(
        data=np.arange(12).reshape(2, 2, 3),
        header=SimpleNamespace(stamp=stamp),
    )
    np.testing.assert_array_equal(
        BoosterStudioRobotInterface.image_to_array(image), image.data
    )
    assert BoosterStudioRobotInterface.image_timestamp(image) == 4.25


def test_booster_studio_detection_coordinates_feed_shared_controller():
    robot = BoosterStudioRobotInterface.__new__(BoosterStudioRobotInterface)
    robot._lock = threading.Lock()
    robot._detections = []
    hypothesis = SimpleNamespace(class_id="Ball", score=0.95)
    position = SimpleNamespace(x=2.0, y=0.5, z=0.0)
    result = SimpleNamespace(
        hypothesis=hypothesis,
        pose=SimpleNamespace(pose=SimpleNamespace(position=position)),
    )
    bbox = SimpleNamespace(
        center=SimpleNamespace(position=SimpleNamespace(x=100.0, y=80.0)),
        size_x=20.0,
        size_y=10.0,
    )
    robot._on_detections(
        SimpleNamespace(detections=[SimpleNamespace(results=[result], bbox=bbox)])
    )
    detection = robot.get_detections()[0]
    assert detection.label == "Ball"
    assert detection.bbox == (90, 75, 110, 85)
    np.testing.assert_allclose(detection.position_3d, [0.5, 0.0, 2.0])
    assert detection.distance == np.hypot(2.0, 0.5)


def _fake_booster_sim(tmp_path):
    root = tmp_path / "robocup_sim_src"
    mjcf = root / "mjcf"
    mjcf.mkdir(parents=True)
    (mjcf / "assets").mkdir()
    (mjcf / "K1_22dof.xml").write_text("<mujoco/>")
    (mjcf / "default_pitch_K1.xml").write_text("<mujoco model='original'/>")
    (mjcf / "default_pitch_K1.extensions.xml").write_text("<extensions/>")
    return root


def test_living_room_scene_is_well_formed_and_has_collision_content():
    root = ET.parse(SCENE_DIR / "living_room_K1.xml").getroot()
    assert root.tag == "mujoco"
    assert root.find(".//model[@name='K1']") is not None
    assert root.find(".//body[@name='ball']/freejoint") is not None
    assert root.find(".//body[@name='couch']/geom[@type='mesh']") is not None
    assert len(root.findall(".//asset/mesh")) == 8
    assert len(root.findall(".//worldbody/geom[@type='box']")) == 3

    extension_root = ET.parse(SCENE_DIR / "living_room_K1.extensions.xml").getroot()
    extension_names = {
        item.attrib["name"] for item in extension_root.findall("extension_process")
    }
    assert extension_names == {
        "detection_extension",
        "rgb_publisher_extension",
        "depth_publisher_extension",
        "pose_publisher_extension",
    }


def test_room_installer_stages_assets_without_changing_default(tmp_path):
    sim_root = _fake_booster_sim(tmp_path)
    original = (sim_root / "mjcf" / "default_pitch_K1.xml").read_text()
    scene, extensions = install_room(sim_root)
    assert scene.is_file()
    assert extensions.is_file()
    assert len(list((sim_root / "mjcf" / "assets" / "nero_room").glob("*.obj"))) == 8
    assert (sim_root / "mjcf" / "default_pitch_K1.xml").read_text() == original
    assert find_sim_root(sim_root) == sim_root


def test_room_activation_is_reversible_and_does_not_overwrite_backup(tmp_path):
    sim_root = _fake_booster_sim(tmp_path)
    default_scene = sim_root / "mjcf" / "default_pitch_K1.xml"
    default_extensions = sim_root / "mjcf" / "default_pitch_K1.extensions.xml"
    original_scene = default_scene.read_text()
    original_extensions = default_extensions.read_text()

    install_room(sim_root, activate=True)
    scene_backup = default_scene.with_name(default_scene.name + BACKUP_SUFFIX)
    extensions_backup = default_extensions.with_name(
        default_extensions.name + BACKUP_SUFFIX
    )
    assert scene_backup.read_text() == original_scene
    assert extensions_backup.read_text() == original_extensions
    assert "Nero living room K1" in default_scene.read_text()

    # A second activation keeps the first pristine backup.
    install_room(sim_root, activate=True)
    assert scene_backup.read_text() == original_scene
    restore_default_scene(sim_root)
    assert default_scene.read_text() == original_scene
    assert default_extensions.read_text() == original_extensions


def test_room_activation_enables_ros_imu_transport_and_restores_it(tmp_path):
    sim_root = _fake_booster_sim(tmp_path)
    container_env = tmp_path / ".env"
    original = "MODEL_PATH=mjcf/default_pitch_K1.xml\nSIM_TRANSPORT=shm\n"
    container_env.write_text(original)

    install_room(sim_root, activate=True, container_env=container_env)
    assert "SIM_TRANSPORT=ros" in container_env.read_text()
    assert container_env.with_name(".env" + BACKUP_SUFFIX).read_text() == original

    # Reconfiguration is idempotent and never replaces the original backup.
    configure_ros_transport(container_env)
    assert container_env.read_text().count("SIM_TRANSPORT=ros") == 1
    restore_default_scene(sim_root, container_env=container_env)
    assert container_env.read_text() == original


def test_room_activation_refuses_to_modify_signed_app_bundle(tmp_path):
    sim_root = _fake_booster_sim(tmp_path / "Booster Studio.app" / "Contents")
    try:
        install_room(sim_root, activate=True)
    except PermissionError as exc:
        assert "signed macOS application bundle" in str(exc)
    else:
        raise AssertionError("signed app activation should have been rejected")
