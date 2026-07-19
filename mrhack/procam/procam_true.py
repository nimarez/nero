#!/usr/bin/env python3
"""Project a true metric circle inside the rectangle formed by four ArUco tags."""

import sys

import cv2
import numpy as np

from context_snippets import (
    PROJ_H,
    PROJ_W,
    capture_color,
    detect_tags,
    load_handles,
    project_png,
)

EXPECTED_IDS = (0, 1, 2, 3)
TAG_SIDE = 1.0
CIRCLE_SAMPLES = 160
CIRCLE_THICKNESS = 34


def transform_points(points, homography):
    """Apply a homography to an Nx2 array."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)


def square_consistency_score(homography, tags):
    """
    Score how square all detected tags look after rectification.

    A homography fitted from one reference tag fixes the plane's Euclidean
    coordinate system. The other equal-size tags should then have four equal
    sides, perpendicular neighboring sides, and equal diagonals.
    """
    errors = []

    for corners in tags.values():
        rectified = transform_points(corners, homography)
        if not np.all(np.isfinite(rectified)):
            return float("inf")

        edges = np.roll(rectified, -1, axis=0) - rectified
        lengths = np.linalg.norm(edges, axis=1)
        mean_length = float(np.mean(lengths))

        if mean_length <= 1e-9:
            return float("inf")

        side_error = float(np.std(lengths) / mean_length)

        perpendicular_errors = []
        for i in range(4):
            a = edges[i]
            b = edges[(i + 1) % 4]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-12:
                return float("inf")
            perpendicular_errors.append(abs(float(np.dot(a, b))) / denom)

        diagonal_0 = float(np.linalg.norm(rectified[2] - rectified[0]))
        diagonal_1 = float(np.linalg.norm(rectified[3] - rectified[1]))
        diagonal_mean = 0.5 * (diagonal_0 + diagonal_1)
        diagonal_error = (
            abs(diagonal_0 - diagonal_1) / diagonal_mean
            if diagonal_mean > 1e-9
            else float("inf")
        )

        # Tags have the same physical side length, so their rectified side
        # length should also be one metric unit.
        scale_error = abs(mean_length - TAG_SIDE) / TAG_SIDE

        errors.append(
            side_error
            + float(np.mean(perpendicular_errors))
            + diagonal_error
            + scale_error
        )

    return float(np.median(errors))


def build_cam_to_floor(tags):
    """
    Construct camera-pixel -> metric-floor coordinates from a tag square.

    Assumption: all four tags are identical, planar, and their reported
    TL/TR/BR/BL corners describe physical squares. One square is sufficient
    to establish a Euclidean coordinate system over the entire plane. We try
    every tag as that reference and retain the rectification under which all
    four observed tags are most consistently unit squares. This avoids needing
    to know the tags' translations or rotations in advance.
    """
    metric_square = np.float32(
        [
            [0.0, 0.0],
            [TAG_SIDE, 0.0],
            [TAG_SIDE, TAG_SIDE],
            [0.0, TAG_SIDE],
        ]
    )

    candidates = []
    for tag_id, corners in tags.items():
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        homography = cv2.getPerspectiveTransform(corners, metric_square)

        if not np.all(np.isfinite(homography)):
            continue

        score = square_consistency_score(homography, tags)
        image_area = abs(float(cv2.contourArea(corners)))
        candidates.append((score, -image_area, tag_id, homography))

    if not candidates:
        raise RuntimeError("Could not estimate a floor rectification")

    # Consistency is primary; image area breaks ties in favor of the tag with
    # the most pixel support and therefore usually the best numerical accuracy.
    score, _, reference_id, homography = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return homography, reference_id, score


def order_clockwise(points):
    """
    Return four points clockwise, with a stable upper-left starting point.

    Both metric floor coordinates and projector pixels use a downward-positive
    y convention. The calibration assumes the dragged handles occupy the same
    physical rectangle corners as the tags, so applying this same ordering to
    both sets establishes their correspondence.
    """
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # With y increasing downward, increasing atan2 angle is visually clockwise.
    ordered = points[np.argsort(angles)]

    # Resolve the cyclic ambiguity consistently at the upper-left point.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def main():
    image = capture_color()
    detected = detect_tags(image)
    found_ids = sorted(detected)
    print("Detected ArUco tag IDs:", found_ids)

    missing_ids = [tag_id for tag_id in EXPECTED_IDS if tag_id not in detected]
    if missing_ids:
        print(
            "Need tags 0, 1, 2, and 3; missing:",
            missing_ids,
            file=sys.stderr,
        )
        return 1

    tags = {
        tag_id: np.asarray(detected[tag_id], dtype=np.float32).reshape(4, 2)
        for tag_id in EXPECTED_IDS
    }

    try:
        handles = np.asarray(load_handles(), dtype=np.float64).reshape(-1, 2)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Could not load four projector handles: {exc}", file=sys.stderr)
        return 1

    if len(handles) != 4 or not np.all(np.isfinite(handles)):
        print(
            f"Expected four finite projector handles; got {handles.tolist()}",
            file=sys.stderr,
        )
        return 1

    try:
        h_cam2floor, reference_id, score = build_cam_to_floor(tags)
    except (cv2.error, RuntimeError) as exc:
        print(f"Floor rectification failed: {exc}", file=sys.stderr)
        return 1

    camera_centers = np.array(
        [np.mean(tags[tag_id], axis=0) for tag_id in EXPECTED_IDS],
        dtype=np.float64,
    )
    metric_centers = transform_points(camera_centers, h_cam2floor)

    if not np.all(np.isfinite(metric_centers)):
        print("Rectified tag centers are not finite", file=sys.stderr)
        return 1

    metric_centers = order_clockwise(metric_centers)
    handles = order_clockwise(handles)

    h_floor2proj, _ = cv2.findHomography(
        metric_centers.astype(np.float32),
        handles.astype(np.float32),
        method=0,
    )
    if h_floor2proj is None or not np.all(np.isfinite(h_floor2proj)):
        print("Could not estimate floor-to-projector homography", file=sys.stderr)
        return 1

    edge_lengths = np.linalg.norm(
        np.roll(metric_centers, -1, axis=0) - metric_centers,
        axis=1,
    )
    rect_width = 0.5 * (edge_lengths[0] + edge_lengths[2])
    rect_height = 0.5 * (edge_lengths[1] + edge_lengths[3])
    radius = 0.4 * min(rect_width, rect_height)
    circle_center = np.mean(metric_centers, axis=0)

    if not np.isfinite(radius) or radius <= 0.0:
        print(f"Invalid metric rectangle dimensions: {edge_lengths}", file=sys.stderr)
        return 1

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        CIRCLE_SAMPLES,
        endpoint=False,
    )
    metric_circle = circle_center + radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    projector_circle = transform_points(metric_circle, h_floor2proj)

    if not np.all(np.isfinite(projector_circle)):
        print("Projected circle contains non-finite points", file=sys.stderr)
        return 1

    canvas = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
    polyline = np.rint(projector_circle).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas,
        [polyline],
        isClosed=True,
        color=(255, 255, 255),
        thickness=CIRCLE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    project_png(canvas)

    print(
        f"Using tag {reference_id} as the metric reference "
        f"(consistency score {score:.4f})"
    )
    print(
        f"Metric rectangle dimensions: {rect_width:.3f} x "
        f"{rect_height:.3f} tag sides"
    )
    print(
        f"Projected true metric circle: center={circle_center.tolist()}, "
        f"radius={radius:.3f}, thickness={CIRCLE_THICKNESS}px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
