import json

import cv2
import numpy as np
import pytest

from nero.projector.calibration import CalibrationState, ProjectorCalibration
from nero.projector.camera import annotate_aruco
from nero.projector.motion import MotionPose, MotionTracker, map_floor_position
from nero.projector.navigation import ProjectorNavigationState, centered_bezier_waypoints
from nero.projector.operator_display import OPERATOR_HTML, RERUN_URL
from nero.projector.render import (
    render_motion_circle,
    render_navigation_overlay,
    render_projector_grid,
)
from nero.projector.visual_editor import VISUAL_EDITOR_HTML


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


def test_centered_vive_pose_maps_to_projector_center_and_draws_circle():
    origin = (-0.096, -0.058, 0.327)
    uv = map_floor_position(origin, origin)
    calibration = ProjectorCalibration()
    grid = render_projector_grid(calibration)
    frame = render_motion_circle(grid, calibration, uv)
    center = tuple(np.rint(calibration.transform(((0.5, 0.5),))[0]).astype(int))

    assert uv == (0.5, 0.5)
    assert not np.array_equal(frame, grid)
    assert np.any(frame[center[1], center[0]] != grid[center[1], center[0]])


def test_controller_height_does_not_change_floor_mapping():
    matrix = ((0.2, 0.1, 0.5), (-0.1, 0.3, 0.4))

    floor_pose = MotionTracker._apply_mapping((1.2, -0.4, 0.1), matrix)
    raised_pose = MotionTracker._apply_mapping((1.2, -0.4, 1.8), matrix)

    assert raised_pose == floor_pose


def test_floor_calibration_recaptures_center_instead_of_reusing_saved_origin(tmp_path):
    center_path = tmp_path / "center.json"
    center_path.write_text('{"position": [9.0, 8.0, 7.0]}')
    tracker = MotionTracker(
        pose_path=tmp_path / "pose.json",
        center_path=center_path,
        mapping_path=tmp_path / "mapping.json",
    )

    status = tracker.begin_floor_calibration()

    assert status["captured"] == 0
    assert status["target_uv"] == [0.5, 0.5]


def test_floor_mapping_defines_metric_room_frame_and_ignores_height():
    mapping = ((0.25, 0.0, 0.0), (0.0, -0.5, 2.0))
    origin = (2.0, 3.0, 0.1)
    floor_pose = MotionPose(1, "WW0", (4.0, 2.0, 0.1), (0.0, 0.0, 0.0, 1.0), 10.0, True)
    raised_pose = MotionPose(2, "WW0", (4.0, 2.0, 1.8), (0.0, 0.0, 0.0, 1.0), 10.0, True)

    floor = MotionTracker._room_pose(floor_pose, origin, mapping, None)
    raised = MotionTracker._room_pose(raised_pose, origin, mapping, None)

    assert floor["x"] == pytest.approx(2.0)
    assert floor["y"] == pytest.approx(-1.0)
    assert raised["x"] == floor["x"]
    assert raised["y"] == floor["y"]
    assert MotionTracker._room_points_to_floor_uv([[2.0, -1.0]], origin, mapping)[0] == pytest.approx(
        [1.0, 1.0]
    )


def test_navigation_contract_falls_back_to_direct_preview_and_accepts_nima_path(tmp_path):
    state = ProjectorNavigationState(tmp_path / "goal.json")
    state.set_goal({"x": 2.0, "y": -1.0, "yaw": 1.57, "source": "operator"})
    robot = {"x": 0.25, "y": 0.5, "yaw": 0.0, "valid": True}

    preview = state.snapshot(robot)
    assert preview["frame_id"] == "room_floor"
    assert preview["control_authority"] == "none"
    assert preview["trajectory"]["source"] == "direct-preview"
    assert preview["trajectory"]["waypoints"] == [[0.25, 0.5], [2.0, -1.0]]

    state.set_trajectory(
        {"waypoints": [[0.25, 0.5], [1.0, 0.2], [2.0, -1.0]], "source": "nima-a-star"}
    )
    planned = state.snapshot(robot)
    assert planned["trajectory"]["source"] == "nima-a-star"
    assert len(planned["trajectory"]["waypoints"]) == 3


def test_direct_preview_bezier_passes_through_room_center():
    route = centered_bezier_waypoints([-1.4, 0.8], [1.6, 0.4])

    assert route[0] == [-1.4, 0.8]
    assert route[-1] == [1.6, 0.4]
    assert [0.0, 0.0] in route


def test_navigation_overlay_draws_robot_axes_path_and_goal():
    calibration = ProjectorCalibration()
    base = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
    robot_frame = {
        "grid_lines": [[[0.4, 0.4], [0.6, 0.4]], [[0.5, 0.3], [0.5, 0.6]]],
        "footprint": [[0.46, 0.46], [0.54, 0.46], [0.54, 0.54], [0.46, 0.54]],
        "x_axis": [[0.5, 0.5], [0.65, 0.5]],
        "y_axis": [[0.5, 0.5], [0.5, 0.35]],
    }

    frame = render_navigation_overlay(
        base,
        calibration,
        robot_frame_uv=robot_frame,
        trajectory_uv=[[0.5, 0.5], [0.75, 0.35]],
        goal_uv=[0.75, 0.35],
        goal_heading_uv=[0.82, 0.35],
        animation_phase=0.5,
    )

    assert np.count_nonzero(frame) > 1000


@pytest.mark.parametrize("preset", ["intent_field", "beacon_trail", "safety_corridor"])
def test_mission_visualizations_are_distinct_from_preserved_engineering_mode(preset):
    base_calibration = ProjectorCalibration()
    mission_calibration = base_calibration.with_visualization(
        {"mode": "mission", "preset": preset}
    )
    trajectory = [[0.22, 0.74], [0.48, 0.58], [0.78, 0.27]]
    goal = [0.84, 0.20]

    engineering = render_navigation_overlay(
        render_projector_grid(base_calibration),
        base_calibration,
        robot_frame_uv=None,
        trajectory_uv=trajectory,
        goal_uv=goal,
        goal_heading_uv=None,
        animation_phase=0.25,
    )
    mission = render_navigation_overlay(
        render_projector_grid(mission_calibration),
        mission_calibration,
        robot_frame_uv=None,
        trajectory_uv=trajectory,
        goal_uv=goal,
        goal_heading_uv=None,
        animation_phase=0.25,
    )

    assert mission_calibration.visualization_mode == "mission"
    assert mission_calibration.mission_preset == preset
    assert np.count_nonzero(mission) > 1000
    assert not np.array_equal(mission, engineering)


def test_mission_completion_adds_high_contrast_ready_animation():
    calibration = ProjectorCalibration().with_visualization({"mode": "mission"})
    base = render_projector_grid(calibration)
    common = {
        "robot_frame_uv": None,
        "trajectory_uv": [[0.25, 0.7], [0.5, 0.5], [0.76, 0.28]],
        "goal_uv": [0.8, 0.24],
        "goal_heading_uv": None,
        "animation_phase": 0.3,
    }

    approaching = render_navigation_overlay(
        base, calibration, **common, distance_to_goal_m=1.2
    )
    complete = render_navigation_overlay(
        base, calibration, **common, distance_to_goal_m=0.2
    )

    assert not np.array_equal(approaching, complete)
    assert np.count_nonzero(complete[:, :, 2] > complete[:, :, 1] * 1.2) > 100


def test_visualization_settings_validate_and_round_trip(tmp_path):
    calibration = ProjectorCalibration().with_visualization(
        {
            "mode": "mission",
            "preset": "intent_field",
            "route_color": "#92FFD4",
            "target_color": "#FFD94F",
            "route_width": 0.08,
            "goal_gap": 0.1,
            "target_radius": 0.07,
            "glow_strength": 1.0,
            "animation_speed": 1.2,
            "grid_intensity": 0.45,
        }
    )

    restored = ProjectorCalibration.load(calibration.save(tmp_path / "projector.json"))

    assert restored == calibration
    assert restored.visualization_dict()["mode"] == "mission"
    with pytest.raises(ValueError, match="unsupported visualization setting"):
        calibration.with_visualization({"fake_control": 1})


def test_visual_editor_exposes_both_modes_and_factory_presets():
    assert 'data-mode="engineering"' in VISUAL_EDITOR_HTML
    assert 'data-mode="mission"' in VISUAL_EDITOR_HTML
    assert 'data-preset="intent_field"' in VISUAL_EDITOR_HTML
    assert 'data-preset="beacon_trail"' in VISUAL_EDITOR_HTML
    assert 'data-preset="safety_corridor"' in VISUAL_EDITOR_HTML
    assert "/api/visualization" in VISUAL_EDITOR_HTML


def test_operator_display_combines_camera_rerun_and_floor_telemetry():
    assert 'src="/stream.mjpg"' in OPERATOR_HTML
    assert f'src="{RERUN_URL}"' in OPERATOR_HTML
    assert "Perspective" in OPERATOR_HTML
    assert "Data frame" in OPERATOR_HTML
    assert "Marker boxes" in OPERATOR_HTML
    assert "vertical dropped" in OPERATOR_HTML


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
