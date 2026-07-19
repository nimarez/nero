import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nero.agents import orb_slam_agent
from nero.perception.aruco_detector import ArucoObjectDetector, load_marker_map


def test_aruco_detector_projects_mapped_marker_with_live_depth_and_intrinsics():
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 100
    )
    rgb = np.full((160, 200, 3), 255, dtype=np.uint8)
    rgb[30:130, 50:150] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    depth = np.full((160, 200), 2000, dtype=np.uint16)
    camera_info = SimpleNamespace(k=[100.0, 0.0, 100.0, 0.0, 100.0, 80.0, 0.0, 0.0, 1.0])
    detector = ArucoObjectDetector({7: "Green Cup"})

    assert detector.initialize()
    assert detector.supports_target(" green   cup ")
    assert not detector.supports_target("chair")
    detector.set_target("green cup")
    detections = detector.detect(rgb, depth, camera_info)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "green cup"
    assert detection.confidence == 1.0
    assert 48 <= detection.bbox[0] <= 52
    assert 148 <= detection.bbox[2] <= 152
    np.testing.assert_allclose(detection.position_3d, [0.0, 0.0, 2.0], atol=0.03)
    assert detection.distance == pytest.approx(2.0, abs=0.03)
    assert detector.find_object(detections, "GREEN CUP") is detection


def test_aruco_mapping_file_and_agent_backend_selection(tmp_path, monkeypatch):
    path = tmp_path / "markers.json"
    path.write_text(json.dumps({"3": "toolbox", "9": "charging station"}))

    assert load_marker_map(path) == {3: "toolbox", 9: "charging station"}
    args = SimpleNamespace(
        object_backend="aruco", aruco_map=str(path), aruco_dictionary="DICT_5X5_50"
    )
    detector = orb_slam_agent.build_object_detector(args)
    assert isinstance(detector, ArucoObjectDetector)
    assert detector.dictionary_name == "DICT_5X5_50"
    assert detector.supported_targets == ("charging station", "toolbox")

    monkeypatch.setenv("NERO_OBJECT_BACKEND", "aruco")
    monkeypatch.setenv("NERO_ARUCO_MAP", str(path))
    detector = orb_slam_agent.build_object_detector(SimpleNamespace(object_backend=None))
    assert isinstance(detector, ArucoObjectDetector)


def test_aruco_rejects_invalid_or_ambiguous_mappings(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="non-empty"):
        load_marker_map(empty)
    with pytest.raises(ValueError, match="unique"):
        ArucoObjectDetector({1: "cup", 2: " CUP "})
    assert not ArucoObjectDetector().initialize()


def test_aruco_cli_keeps_k1_sensors_implicit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nero-orb-slam",
            "--object-backend",
            "aruco",
            "--aruco-map",
            "markers.json",
        ],
    )
    args = orb_slam_agent.parse_args()
    assert args.object_backend == "aruco"
    assert args.aruco_map == "markers.json"
    assert not hasattr(args, "camera")
    assert not hasattr(args, "depth_camera")
