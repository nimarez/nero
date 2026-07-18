import sys
import hashlib
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from nero.mapping.mapping_policy import MappingConfig, MappingPolicy
from nero.slam.imu_buffer import IMUBuffer, IMUMeasurement
from nero.slam.k1_calibration import (
    K1Calibration,
    _device_body,
    estimate_frequency,
    estimate_imu_noise,
)
from nero.slam.orb_slam3_node import ORBSLAM3Node
import nero.slam.setup_orbslam as setup_orbslam
from nero.slam.setup_orbslam import check_native_runtime


def calibration() -> K1Calibration:
    return K1Calibration(
        camera_frame="camera_color_optical_frame",
        imu_frame="body_imu",
        width=320,
        height=240,
        camera_fps=27.5,
        camera_matrix=[216.5, 0, 160, 0, 216.5, 120, 0, 0, 1],
        distortion=[0, 0, 0, 0, 0],
        depth_map_factor=1000.0,
        camera_rgb=True,
        tbc=np.eye(4).tolist(),
        imu_frequency=200.0,
        imu_noise_gyro=0.001,
        imu_noise_acc=0.01,
        imu_gyro_walk=0.0001,
        imu_acc_walk=0.001,
    )


def test_imu_buffer_orders_and_drains_each_interval_once():
    buffer = IMUBuffer()
    samples = [
        IMUMeasurement(1.0, (1, 2, 3), (4, 5, 6)),
        IMUMeasurement(2.0, (1, 2, 3), (4, 5, 6)),
        IMUMeasurement(3.0, (1, 2, 3), (4, 5, 6)),
    ]
    buffer.extend(samples)
    buffer.append(samples[1])
    assert buffer.between(None, 2.0) == samples[:2]
    assert buffer.between(2.0, 3.0) == samples[2:]
    assert len(buffer) == 0


def test_calibration_round_trip_and_orb_settings(tmp_path):
    path = tmp_path / "calibration.json"
    settings = tmp_path / "settings.yaml"
    expected = calibration()
    expected.save(path)
    assert K1Calibration.load(path) == expected
    expected.write_orbslam_settings(settings)
    text = settings.read_text()
    assert 'File.version: "1.0"' in text
    assert "Camera1.cx: 160.0" in text
    assert "RGBD.DepthMapFactor: 1000.0" in text
    assert "Camera.fps: 27.5" in text
    assert "IMU.Frequency: 200.0" in text
    assert "IMU.T_b_c1: !!opencv-matrix" in text
    assert "IMU.InsertKFsWhenLost: 0" in text


def test_stationary_imu_noise_estimator_returns_finite_positive_terms():
    rng = np.random.default_rng(7)
    samples = [
        (index / 200, rng.normal(0, 0.001, 3), rng.normal((0, 0, 9.81), 0.01, 3))
        for index in range(1000)
    ]
    result = estimate_imu_noise(samples)
    assert result["imu_frequency"] == pytest.approx(200)
    assert all(np.isfinite(value) and value > 0 for value in result.values())


def test_sensor_frequency_uses_live_timestamps():
    assert estimate_frequency([index / 25 for index in range(20)], "camera") == pytest.approx(25)


def test_booster_device_info_body_is_unwrapped():
    assert _device_body({"body": {"cameras": [1]}, "kind": 4}) == {"cameras": [1]}
    assert _device_body({"body": '{"imus": [1]}'}) == {"imus": [1]}


class FakeSystem:
    def __init__(self, vocabulary, settings, sensor):
        self.arguments = vocabulary, settings, sensor
        self.calls = []
        self.initialized = False
        self.closed = False

    def initialize(self):
        self.initialized = True

    def process_image_rgbd_inertial(self, rgb, depth, timestamp, imu):
        self.calls.append((rgb, depth, timestamp, imu))

    def get_frame_pose(self):
        tcw = np.eye(4)
        tcw[0, 3] = -2.0
        return tcw

    def get_tracking_state(self):
        return 2

    def get_tracked_mappoints(self):
        return [object(), object()]

    def reset(self):
        self.calls.clear()

    def shutdown(self):
        self.closed = True


class RejectingEnhancedSystem(FakeSystem):
    def process_rgbd_inertial_enhanced(self, rgb, depth, timestamp, imu):
        self.calls.append((rgb, depth, timestamp, imu))
        return SimpleNamespace(success=False, is_valid=False)


def native_node(tmp_path, monkeypatch):
    vocabulary = tmp_path / "ORBvoc.txt"
    vocabulary.write_text("vocabulary")
    calibration_path = tmp_path / "calibration.json"
    calibration().save(calibration_path)
    module = SimpleNamespace(Sensor=SimpleNamespace(IMU_RGBD="IMU_RGBD"), System=FakeSystem)
    monkeypatch.setitem(sys.modules, "orbslam3", module)
    node = ORBSLAM3Node(
        vocab_path=str(vocabulary),
        settings_path=str(tmp_path / "settings.yaml"),
        calibration_path=str(calibration_path),
        allow_fallback=False,
        start_imu_source=False,
    )
    node.initialize()
    return node


def test_native_backend_selects_imu_rgbd_and_inverts_tcw(tmp_path, monkeypatch):
    node = native_node(tmp_path, monkeypatch)
    assert node.backend_name == "orbslam3-imu-rgbd"
    assert node._slam_system.arguments[2] == "IMU_RGBD"
    imu = [IMUMeasurement(1.0, (0, 0, 9.81), (0, 0, 0))]
    pose = node.track_frame(
        np.zeros((240, 320, 3), np.uint8),
        np.ones((240, 320), np.float32),
        imu_data=imu,
        timestamp=1.0,
    )
    assert pose.tracking_status == "OK"
    np.testing.assert_allclose(pose.position, [2, 0, 0])
    assert pose.num_map_points == 2
    assert node._slam_system.calls[0][1].dtype == np.uint16
    node.shutdown()
    assert node._slam_system.closed


def test_native_backend_refuses_frame_without_imu(tmp_path, monkeypatch):
    node = native_node(tmp_path, monkeypatch)
    pose = node.track_frame(
        np.zeros((240, 320, 3), np.uint8),
        np.ones((240, 320), np.uint16),
        imu_data=[],
        timestamp=1.0,
    )
    assert pose.tracking_status == "LOST"
    assert node._slam_system.calls == []


def test_native_backend_does_not_resubmit_duplicate_camera_timestamp(tmp_path, monkeypatch):
    node = native_node(tmp_path, monkeypatch)
    rgb = np.zeros((240, 320, 3), np.uint8)
    depth = np.ones((240, 320), np.uint16)
    imu = [IMUMeasurement(1.0, (0, 0, 9.81), (0, 0, 0))]
    first = node.track_frame(rgb, depth, imu_data=imu, timestamp=1.0)
    duplicate = node.track_frame(rgb, depth, imu_data=imu, timestamp=1.0)
    assert duplicate is first
    assert len(node._slam_system.calls) == 1


def test_native_backend_honors_enhanced_result_failure(tmp_path, monkeypatch):
    vocabulary = tmp_path / "ORBvoc.txt"
    vocabulary.write_text("vocabulary")
    calibration_path = tmp_path / "calibration.json"
    calibration().save(calibration_path)
    monkeypatch.setitem(
        sys.modules,
        "orbslam3",
        SimpleNamespace(
            Sensor=SimpleNamespace(IMU_RGBD="IMU_RGBD"), System=RejectingEnhancedSystem
        ),
    )
    node = ORBSLAM3Node(
        vocab_path=str(vocabulary),
        settings_path=str(tmp_path / "settings.yaml"),
        calibration_path=str(calibration_path),
        allow_fallback=False,
        start_imu_source=False,
    )
    node.initialize()
    pose = node.track_frame(
        np.zeros((240, 320, 3), np.uint8),
        np.ones((240, 320), np.uint16),
        imu_data=[IMUMeasurement(1.0, (0, 0, 9.81), (0, 0, 0))],
        timestamp=1.0,
    )
    assert pose.tracking_status == "LOST"


def test_fallback_is_explicit_and_handles_featureless_frame():
    node = ORBSLAM3Node(allow_fallback=True)
    node.initialize()
    assert node.backend_name == "rgbd-fallback"
    pose = node.track_frame(
        np.zeros((240, 320, 3), np.uint8),
        np.ones((240, 320), np.float32),
        timestamp=0.0,
    )
    assert pose.timestamp == 0.0
    assert pose.tracking_status == "LOST"


def test_mapping_policy_accepts_current_slam_config(tmp_path):
    policy = MappingPolicy(
        robot=None,
        slam_config={
            "voc_path": "config/ORBvoc.txt",
            "settings_path": "config/k1_orbslam3_imu_rgbd.yaml",
        },
        mapping_config=MappingConfig(output_dir=str(tmp_path / "mapping")),
    )
    assert policy._slam.vocab_path.name == "ORBvoc.txt"


def test_native_runtime_check_is_a_noop_off_linux(monkeypatch):
    monkeypatch.setattr("nero.slam.setup_orbslam.sys.platform", "darwin")
    check_native_runtime()


def test_vocabulary_installer_verifies_and_extracts_atomically(tmp_path, monkeypatch):
    archive_file = io.BytesIO()
    payload = b"orb vocabulary\n"
    with tarfile.open(fileobj=archive_file, mode="w:gz") as bundle:
        info = tarfile.TarInfo("ORBvoc.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    archive = archive_file.getvalue()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return None

        def read(self):
            return archive

    monkeypatch.setattr(
        setup_orbslam.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(),
    )
    monkeypatch.setattr(setup_orbslam, "VOCAB_ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    destination = tmp_path / "nested" / "ORBvoc.txt"
    assert setup_orbslam.install_vocabulary(destination) == destination.resolve()
    assert destination.read_bytes() == payload
