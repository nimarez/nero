import hashlib
import io
import json
import socket
import struct
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nero.perception.qnn_artifact as qnn_artifact
import nero.perception.qnn_deploy as qnn_deploy
import nero.perception.qnn_worker as qnn_worker
import nero.perception.qnn_yolo_world as qnn_runtime
from nero.perception.object_detector import ObjectDetector
from nero.perception.object_detector import configure_qualcomm_cpu_partition


def test_qnn_preprocess_matches_yolo_letterbox_and_channel_order():
    image = np.zeros((448, 544, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)

    tensor, geometry = qnn_runtime.preprocess_yolo_world(image, 256)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    assert geometry.scale == pytest.approx(256 / 544)
    assert (geometry.left, geometry.top) == (0, 22)
    np.testing.assert_allclose(tensor[0, :, 100, 100], np.array([30, 20, 10]) / 255, atol=1e-6)
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([114, 114, 114]) / 255, atol=1e-6)


def test_qnn_retains_all_cpus_for_slam(monkeypatch):
    monkeypatch.setenv("NERO_OBJECT_BACKEND", "yolo-world-qnn")
    monkeypatch.setattr(
        qnn_runtime.os,
        "sched_getaffinity",
        lambda _pid: pytest.fail("QNN must not change process affinity"),
        raising=False,
    )

    assert configure_qualcomm_cpu_partition() is None


def test_qnn_decode_unletterboxes_filters_and_suppresses_overlaps():
    _, geometry = qnn_runtime.preprocess_yolo_world(np.zeros((448, 544, 3), dtype=np.uint8), 256)
    output = np.zeros((1, 5, 3), dtype=np.float32)
    output[0, :, 0] = [128, 128, 100, 50, 0.9]
    output[0, :, 1] = [129, 128, 100, 50, 0.8]
    output[0, :, 2] = [20, 20, 10, 10, 0.2]

    detections = qnn_runtime.decode_yolo_world(output, geometry, 0.5, 10)

    assert len(detections) == 1
    assert detections[0][0] == pytest.approx(0.9)
    assert detections[0][1] == (166, 172, 378, 278)


class _FakeSessionOptions:
    def __init__(self):
        self.config = {}

    def add_session_config_entry(self, name, value):
        self.config[name] = value


class _FakeSession:
    created = []

    def __init__(self, model, sess_options, providers, provider_options):
        self.model = model
        self.options = sess_options
        self.requested_providers = providers
        self.provider_options = provider_options
        self.fallback_disabled = False
        self.__class__.created.append(self)

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_providers(self):
        return [qnn_runtime.QNN_PROVIDER]

    def get_inputs(self):
        return [
            SimpleNamespace(name="images", shape=[1, 3, 256, 256]),
            SimpleNamespace(name="text_features", shape=[1, 1, 512]),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="output_0", shape=[1, 5, 1344])]

    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 256, 256)
        assert inputs["text_features"].shape == (1, 1, 512)
        return [np.zeros((1, 5, 1344), dtype=np.float32)]


def test_qnn_session_is_htp_only_and_disables_cpu_fallback(monkeypatch, tmp_path):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [qnn_runtime.QNN_PROVIDER],
        SessionOptions=_FakeSessionOptions,
        InferenceSession=_FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    _FakeSession.created.clear()

    runtime = qnn_runtime.QNNYoloWorldRuntime(tmp_path / "model.onnx")
    output, _ = runtime.infer(
        np.zeros((1, 3, 256, 256), np.float32),
        np.zeros((1, 1, 512), np.float32),
    )

    session = _FakeSession.created[-1]
    assert session.options.config["session.disable_cpu_ep_fallback"] == "1"
    assert session.requested_providers == [qnn_runtime.QNN_PROVIDER]
    assert session.provider_options[0]["backend_type"] == "htp"
    assert session.provider_options[0]["offload_graph_io_quantization"] == "0"
    assert session.fallback_disabled
    assert output.shape == (1, 5, 1344)


def test_qnn_session_refuses_cpu_only_onnxruntime(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    with pytest.raises(RuntimeError, match="does not expose QNNExecutionProvider"):
        qnn_runtime.QNNYoloWorldRuntime(tmp_path / "model.onnx")


def test_qnn_plugin_worker_protocol_round_trip(monkeypatch, tmp_path):
    class FakeSession:
        def run(self, _outputs, inputs, run_options):
            assert run_options == "burst"
            assert inputs["images"].shape == (1, 3, 256, 256)
            assert inputs["text_features"].shape == (1, 1, 512)
            return [np.full((1, 5, 1344), 7, dtype=np.float32)]

    monkeypatch.setattr(
        qnn_worker,
        "_create_session",
        lambda _model: (FakeSession(), "burst", "1.26.0", "2.4.0", 1.25),
    )
    parent, child = socket.socketpair()
    child_fd = child.detach()
    errors = []

    def run_worker():
        try:
            qnn_worker.serve(tmp_path / "model.onnx", child_fd)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=run_worker)
    thread.start()
    kind, payload = qnn_runtime._recv_message(parent)
    assert kind == b"R"
    assert json.loads(payload)["provider"] == qnn_runtime.QNN_PROVIDER

    images = np.zeros((1, 3, 256, 256), dtype=np.float32)
    features = np.zeros((1, 1, 512), dtype=np.float32)
    parent.sendall(b"I" + images.tobytes() + features.tobytes())
    assert qnn_runtime._recv_exact(parent, 1) == b"O"
    elapsed = struct.unpack("<d", qnn_runtime._recv_exact(parent, 8))[0]
    output = np.frombuffer(qnn_runtime._recv_exact(parent, 1 * 5 * 1344 * 4), dtype=np.float32)
    assert elapsed >= 0
    assert np.all(output == 7)
    parent.sendall(b"Q")
    parent.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []


def test_qnn_artifact_install_and_verification(monkeypatch, tmp_path):
    contents = {"model.onnx": b"onnx", "model.data": b"external weights"}
    expected = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in contents.items()
    }
    monkeypatch.setattr(qnn_artifact, "EXPECTED_FILES", expected)
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in contents.items():
            bundle.writestr(f"job_123/{name}", data)

    model = qnn_artifact.install_qnn_artifact(archive, tmp_path / "installed")

    assert model.read_bytes() == b"onnx"
    manifest = json.loads((model.parent / "manifest.json").read_text())
    assert manifest["target_model"] == qnn_artifact.DEFAULT_TARGET_MODEL
    assert qnn_artifact.verify_qnn_artifact(model.parent) == model
    (model.parent / "model.data").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        qnn_artifact.verify_qnn_artifact(model.parent)


def test_object_detector_qnn_path_is_open_vocab_async(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    class FakeRuntime:
        providers = (qnn_runtime.QNN_PROVIDER,)

        def __init__(self, path, inference_size):
            assert Path(path) == model_path
            assert inference_size == 256

        def infer(self, images, text_features):
            assert images.shape == (1, 3, 256, 256)
            assert text_features[0, 0, 0] == 7
            output = np.zeros((1, 5, 1344), dtype=np.float32)
            output[0, :, 0] = [128, 128, 100, 50, 0.9]
            return output, 0.001

    class FakeEncoder:
        def encode(self, target):
            value = 1 if target == "object" else 7
            return np.full((1, 1, 512), value, dtype=np.float32)

    monkeypatch.setattr(qnn_artifact, "verify_qnn_artifact", lambda _: model_path)
    monkeypatch.setattr(qnn_runtime, "QNNYoloWorldRuntime", FakeRuntime)
    monkeypatch.setattr(qnn_runtime, "YoloWorldPromptEncoder", FakeEncoder)
    detector = ObjectDetector(backend="qnn", model_path=model_path)

    assert detector.initialize()
    assert detector.supported_targets is None
    detector.set_target("green can")
    image = np.zeros((448, 544, 3), dtype=np.uint8)
    depth = np.full((448, 544), 1000, dtype=np.uint16)
    assert detector.detect(image, depth) == []
    for _ in range(100):
        detections = detector.detect(image, depth)
        if detector.result_revision:
            break
        time.sleep(0.001)

    assert detector.result_revision == 1
    assert detections[0].label == "green can"
    assert detections[0].bbox == (166, 172, 378, 278)
    assert detections[0].position_3d[2] == 1.0
    detector.close()


def test_qnn_detector_rejects_an_input_size_that_does_not_match_graph():
    with pytest.raises(ValueError, match="requires inference size 256"):
        ObjectDetector(backend="qnn", inference_size=320)


def test_ai_hub_download_uses_the_actual_suffixed_filename(monkeypatch, tmp_path):
    requested = tmp_path / "model.zip"
    actual = tmp_path / "model.zip.onnx.zip"
    actual.write_bytes(b"zip")
    fake_model = SimpleNamespace(download=lambda filename: str(actual))
    monkeypatch.setitem(
        sys.modules,
        "qai_hub",
        SimpleNamespace(get_model=lambda model_id: fake_model),
    )

    result = qnn_artifact._download_target_model(qnn_artifact.DEFAULT_TARGET_MODEL, requested)

    assert result == actual


def test_qnn_deploy_streams_binary_without_a_remote_tty(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    model = artifact / "model.onnx"
    model.write_bytes(b"model")
    archive = SimpleNamespace(stdout=io.BytesIO(b"archive"), wait=lambda: 0)
    calls = []
    monkeypatch.setattr(qnn_deploy, "verify_qnn_artifact", lambda _: model)
    monkeypatch.setattr(qnn_deploy.subprocess, "Popen", lambda command, **kwargs: archive)
    monkeypatch.setattr(
        qnn_deploy.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nero-deploy-qnn-model",
            "--host",
            "robot.local",
            "--repo",
            "/tmp/repo with space",
        ],
    )

    qnn_deploy.main()

    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "-T"]
    assert command[2] == "booster@robot.local"
    assert "'/tmp/repo with space'" in command[3]
    assert "/home/booster/.local/bin/uv" in command[3]
    assert "nero-install-qnn-model" in command[3]
    assert kwargs["stdin"] is archive.stdout
