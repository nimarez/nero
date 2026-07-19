import io
import json
import time

import numpy as np
import pytest

import nero.perception.modal_yolo_world as modal_runtime
from nero.perception.object_detector import ObjectDetector


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_modal_client_sends_proxy_auth_and_decodes_detections(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(
            json.dumps(
                {
                    "detections": [
                        {
                            "label": "green can",
                            "confidence": 0.91,
                            "bbox": [10, 20, 30, 40],
                        }
                    ],
                    "inference_seconds": 0.012,
                }
            ).encode()
        )

    monkeypatch.setattr(modal_runtime.urllib.request, "urlopen", fake_urlopen)
    client = modal_runtime.ModalYoloWorldClient(
        url="https://example--detect.modal.run",
        key="wk-test",
        secret="ws-test",
        timeout=7,
    )

    detections, elapsed = client.detect(
        np.zeros((32, 48, 3), dtype=np.uint8),
        "green can",
        0.5,
        256,
        10,
    )

    request = captured["request"]
    headers = {name.lower(): value for name, value in request.header_items()}
    body = json.loads(request.data)
    assert request.full_url == "https://example--detect.modal.run"
    assert headers["modal-key"] == "wk-test"
    assert headers["modal-secret"] == "ws-test"
    assert body["target"] == "green can"
    assert body["image_b64"]
    assert captured["timeout"] == 7
    assert detections == [
        modal_runtime.ModalDetection("green can", 0.91, (10, 20, 30, 40))
    ]
    assert elapsed == pytest.approx(0.012)


def test_modal_client_requires_https_and_proxy_credentials(monkeypatch):
    monkeypatch.delenv("NERO_MODAL_ALLOW_HTTP", raising=False)
    with pytest.raises(ValueError, match="must use HTTPS"):
        modal_runtime.ModalYoloWorldClient(
            url="http://localhost:8000", key="key", secret="secret"
        )
    with pytest.raises(ValueError, match="proxy auth"):
        modal_runtime.ModalYoloWorldClient(url="https://example.modal.run")


def test_object_detector_modal_path_is_open_vocab_async(monkeypatch):
    class FakeClient:
        url = "https://example--detect.modal.run"

        def __init__(self):
            self.calls = []

        def detect(
            self,
            image,
            target,
            confidence_threshold,
            inference_size,
            max_detections,
        ):
            self.calls.append(target)
            return [
                modal_runtime.ModalDetection(target, 0.9, (10, 20, 30, 40))
            ], 0.001

    monkeypatch.setattr(modal_runtime, "ModalYoloWorldClient", FakeClient)
    detector = ObjectDetector(backend="modal")

    assert detector.initialize()
    assert detector.supported_targets is None
    detector.set_target("green can")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.detect(image) == []
    for _ in range(100):
        detections = detector.detect(image)
        if detector.result_revision:
            break
        time.sleep(0.001)

    assert detector.result_revision == 1
    assert detections[0].label == "green can"
    assert detections[0].bbox == (10, 20, 30, 40)
    assert detector._modal_client.calls == ["object", "green can"]
    detector.close()
