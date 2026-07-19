import json

import cv2
import numpy as np
import pytest

from nero.projector.calibration import CalibrationState, ProjectorCalibration
from nero.projector.camera import annotate_aruco
from nero.projector.render import render_projector_grid


def test_calibration_round_trip_and_atomic_save(tmp_path):
    calibration = ProjectorCalibration(
        handles=((120, 90), (1720, 150), (1580, 980), (180, 880)),
        grid_divisions=16,
        line_thickness=3,
    )
    path = calibration.save(tmp_path / "projector.json")

    assert ProjectorCalibration.load(path) == calibration
    assert json.loads(path.read_text())["markers"] == {
        "dictionary": "DICT_4X4_50",
        "ids": [1, 2, 3, 4],
        "size_m": 0.13,
    }


def test_degenerate_handles_fail_closed():
    with pytest.raises(ValueError, match="degenerate"):
        ProjectorCalibration(handles=((1, 1), (2, 2), (3, 3), (4, 4)))


def test_calibration_state_is_latest_wins():
    state = CalibrationState(ProjectorCalibration())
    state.update_handles(((100, 100), (1800, 100), (1700, 950), (200, 950)))
    state.update_handles(((120, 120), (1750, 140), (1680, 930), (220, 910)))
    calibration, version, _ = state.snapshot()

    assert version == 2
    assert calibration.handles[0] == (120.0, 120.0)


def test_render_grid_has_green_grid_white_corners_and_orange_center():
    calibration = ProjectorCalibration()
    frame = render_projector_grid(calibration)

    assert frame.shape == (1080, 1920, 3)
    assert int(frame[:, :, 1].max()) == 255
    assert np.count_nonzero(np.all(frame == (255, 255, 255), axis=2)) > 100
    assert np.count_nonzero(np.all(frame == (0, 142, 255), axis=2)) > 20


def test_aruco_overlay_detects_only_expected_ids():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    placements = {1: (80, 90), 2: (420, 90), 3: (760, 380), 4: (1030, 380), 8: (610, 90)}
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 170)
        canvas[y : y + 170, x : x + 170] = marker
    image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    annotated, detections = annotate_aruco(image)

    assert [item.marker_id for item in detections] == [1, 2, 3, 4]
    assert annotated.shape == image.shape
