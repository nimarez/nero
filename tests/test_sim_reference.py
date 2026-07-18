import numpy as np
import pytest

from nero.evaluation.sim_reference import (
    align_se2,
    depth_to_world_points,
    localization_metrics,
    map_metrics,
)


def _trajectory(scale=1.0):
    poses = []
    for x, y, yaw in ((0, 0, 0), (1, 0, 0.1), (2, 0.5, 0.2), (3, 1, 0.3)):
        pose = np.eye(4)
        pose[:2, :2] = [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
        pose[:2, 3] = np.array([x, y]) * scale
        poses.append(pose)
    return np.asarray(poses)


def test_localization_alignment_removes_origin_but_not_scale():
    reference = _trajectory()
    transform = np.eye(4)
    angle = 0.7
    transform[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    transform[:2, 3] = [4, -2]
    estimated = np.linalg.inv(transform) @ reference
    metrics = localization_metrics(estimated, reference, total_frames=5)
    assert metrics["ate_rmse_m"] < 1e-12
    assert metrics["yaw_rmse_deg"] < 1e-10
    assert metrics["tracking_valid_ratio"] == pytest.approx(0.8)

    scaled_metrics = localization_metrics(_trajectory(scale=1.2), reference)
    assert scaled_metrics["scale_error_percent"] == pytest.approx(20.0)
    assert scaled_metrics["ate_rmse_m"] > 0.1


def test_depth_backprojection_and_map_metrics_are_metric():
    depth = np.array([[1000, 0], [2000, 7000]], dtype=np.uint16)
    intrinsic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    points = depth_to_world_points(depth, np.eye(4), intrinsic, stride=1)
    np.testing.assert_allclose(points, [[0, 0, 1], [0, 2, 2]])

    reference = np.array([[0, 0, 1], [0, 2, 2], [1, 1, 1]], dtype=float)
    alignment = np.eye(4)
    alignment[0, 3] = -3
    metrics = map_metrics(reference + [3, 0, 0], reference, alignment=alignment)
    assert metrics["symmetric_rmse_m"] < 1e-12
    assert metrics["f_score"] == 1.0


def test_alignment_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="counts"):
        align_se2(_trajectory(), _trajectory()[:3])
