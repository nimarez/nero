"""Reference metrics for measuring sim-to-real localization and map drift.

The alignment intentionally estimates only a rigid planar transform and a
vertical offset. It never estimates scale: RGB-D localization is metric, so a
scale error must remain visible in the score.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def _pose_array(poses: Iterable[np.ndarray]) -> np.ndarray:
    values = np.asarray(list(poses), dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    if len(values) < 2:
        raise ValueError("at least two paired poses are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("poses contain non-finite values")
    return values


def align_se2(
    estimated_poses: Iterable[np.ndarray], reference_poses: Iterable[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align estimated poses to references without changing scale."""
    estimated = _pose_array(estimated_poses)
    reference = _pose_array(reference_poses)
    if len(estimated) != len(reference):
        raise ValueError("estimated and reference pose counts must match")

    estimated_xy = estimated[:, :2, 3]
    reference_xy = reference[:, :2, 3]
    estimated_center = estimated_xy.mean(axis=0)
    reference_center = reference_xy.mean(axis=0)
    covariance = (estimated_xy - estimated_center).T @ (
        reference_xy - reference_center
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation_2d = vt.T @ u.T
    if np.linalg.det(rotation_2d) < 0:
        vt[-1] *= -1
        rotation_2d = vt.T @ u.T
    translation = reference_center - rotation_2d @ estimated_center

    alignment = np.eye(4)
    alignment[:2, :2] = rotation_2d
    alignment[:2, 3] = translation
    planar_aligned = np.einsum("ij,njk->nik", alignment, estimated)
    alignment[2, 3] = float(
        np.median(reference[:, 2, 3] - planar_aligned[:, 2, 3])
    )
    return np.einsum("ij,njk->nik", alignment, estimated), alignment


def _yaw(poses: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz")[:, 2]


def localization_metrics(
    estimated_poses: Iterable[np.ndarray],
    reference_poses: Iterable[np.ndarray],
    *,
    total_frames: int | None = None,
) -> dict[str, float | int]:
    """Compute ATE, orientation, RPE, scale, and tracking coverage metrics."""
    estimated = _pose_array(estimated_poses)
    reference = _pose_array(reference_poses)
    aligned, _ = align_se2(estimated, reference)
    count = len(aligned)
    total = count if total_frames is None else int(total_frames)
    if total < count or total <= 0:
        raise ValueError("total_frames must be at least the valid pose count")

    errors = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    planar_errors = np.linalg.norm(aligned[:, :2, 3] - reference[:, :2, 3], axis=1)
    yaw_errors = np.asarray(wrap_angle(_yaw(aligned) - _yaw(reference)))

    estimated_relative = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    reference_relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    relative_error = np.linalg.inv(reference_relative) @ estimated_relative
    relative_translation = np.linalg.norm(relative_error[:, :3, 3], axis=1)
    relative_yaw = np.asarray(wrap_angle(_yaw(relative_error)))

    estimated_length = float(
        np.linalg.norm(np.diff(aligned[:, :3, 3], axis=0), axis=1).sum()
    )
    reference_length = float(
        np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1).sum()
    )
    scale_ratio = estimated_length / reference_length if reference_length > 0 else 1.0

    return {
        "paired_poses": count,
        "total_frames": total,
        "tracking_valid_ratio": count / total,
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "planar_ate_rmse_m": float(np.sqrt(np.mean(planar_errors**2))),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_errors**2)))),
        "rpe_translation_rmse_m": float(
            np.sqrt(np.mean(relative_translation**2))
        ),
        "rpe_yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(relative_yaw**2)))),
        "estimated_path_length_m": estimated_length,
        "reference_path_length_m": reference_length,
        "path_length_ratio": scale_ratio,
        "scale_error_percent": abs(scale_ratio - 1.0) * 100.0,
    }


def depth_to_world_points(
    depth: np.ndarray,
    camera_pose: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    depth_map_factor: float = 1000.0,
    depth_min_m: float = 0.5,
    depth_max_m: float = 6.0,
    stride: int = 8,
) -> np.ndarray:
    """Back-project a depth image into a downsampled world-frame point cloud."""
    if stride <= 0 or depth_map_factor <= 0:
        raise ValueError("stride and depth_map_factor must be positive")
    values = np.asarray(depth, dtype=float)[::stride, ::stride] / depth_map_factor
    rows, columns = np.indices(values.shape)
    rows = rows * stride
    columns = columns * stride
    valid = np.isfinite(values) & (values >= depth_min_m) & (values <= depth_max_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)
    intrinsic = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    z = values[valid]
    x = (columns[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    return (np.asarray(camera_pose) @ camera_points.T).T[:, :3]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if not len(points):
        return points.reshape(0, 3)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def map_metrics(
    estimated_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    alignment: np.ndarray | None = None,
    voxel_size_m: float = 0.05,
    tolerance_m: float = 0.10,
) -> dict[str, float | int]:
    """Compute symmetric geometry metrics after trajectory-derived alignment."""
    estimated = np.asarray(estimated_points, dtype=float).reshape(-1, 3)
    reference = np.asarray(reference_points, dtype=float).reshape(-1, 3)
    if alignment is not None and len(estimated):
        homogeneous = np.column_stack((estimated, np.ones(len(estimated))))
        estimated = (np.asarray(alignment, dtype=float) @ homogeneous.T).T[:, :3]
    estimated = _voxel_downsample(estimated, voxel_size_m)
    reference = _voxel_downsample(reference, voxel_size_m)
    if not len(estimated) or not len(reference):
        raise ValueError("both point clouds must contain valid points")

    estimated_to_reference = cKDTree(reference).query(estimated, workers=-1)[0]
    reference_to_estimated = cKDTree(estimated).query(reference, workers=-1)[0]
    precision = float(np.mean(estimated_to_reference <= tolerance_m))
    recall = float(np.mean(reference_to_estimated <= tolerance_m))
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    squared = np.concatenate((estimated_to_reference**2, reference_to_estimated**2))
    return {
        "estimated_points": len(estimated),
        "reference_points": len(reference),
        "symmetric_rmse_m": float(np.sqrt(np.mean(squared))),
        "chamfer_m2": float(np.mean(squared)),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "tolerance_m": tolerance_m,
    }
