"""Visual odometry for localization without SLAM.

Tracks camera motion using feature matching between consecutive frames.
Does not perform loop closure or global optimization (that would be SLAM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Pose2D:
    """2D pose (x, y, theta)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians

    def __add__(self, other: "Pose2D") -> "Pose2D":
        return Pose2D(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=self.theta + other.theta,
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def to_transform_matrix(self) -> np.ndarray:
        """Convert to 3x3 transform matrix."""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        return np.array(
            [
                [cos_t, -sin_t, self.x],
                [sin_t, cos_t, self.y],
                [0, 0, 1],
            ]
        )


class VisualOdometry:
    """Monocular visual odometry using ORB features.

    Tracks relative motion between consecutive frames.
    Requires known camera intrinsics and assumes planar motion.
    """

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 320.0,
        cy: float = 240.0,
        max_features: int = 2000,
        min_matches: int = 20,
    ):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._max_features = max_features
        self._min_matches = min_matches

        # ORB detector
        self._orb = cv2.ORB_create(nfeatures=max_features)

        # BFMatcher
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # State
        self._pose = Pose2D()
        self._prev_kp: Optional[np.ndarray] = None
        self._prev_desc: Optional[np.ndarray] = None
        self._initialized = False

        # For depth-assisted VO
        self._use_depth = False

    def set_use_depth(self, use_depth: bool) -> None:
        """Enable depth-assisted odometry for scale recovery."""
        self._use_depth = use_depth

    def initialize(self, frame: np.ndarray) -> None:
        """Initialize with first frame."""
        gray = self._to_gray(frame)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if kp is not None and desc is not None:
            self._prev_kp = np.array([p.pt for p in kp], dtype=np.float32)
            self._prev_desc = desc
            self._initialized = True
            logger.info(f"Visual odometry initialized with {len(kp)} features")

    def update(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Optional[Pose2D]:
        """Update odometry with new frame.

        Args:
            frame: Current RGB or grayscale frame
            depth: Optional depth image (for scale recovery)

        Returns:
            Current pose or None if tracking lost
        """
        if not self._initialized:
            self.initialize(frame)
            return self._pose

        gray = self._to_gray(frame)
        kp, desc = self._orb.detectAndCompute(gray, None)

        if (
            kp is None
            or desc is None
            or self._prev_kp is None
            or self._prev_desc is None
        ):
            self._prev_kp = (
                np.array([p.pt for p in kp], dtype=np.float32)
                if kp is not None
                else None
            )
            self._prev_desc = desc
            return None

        # Match features
        matches = self._matcher.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < self._min_matches:
            logger.warning(f"Too few matches: {len(matches)} < {self._min_matches}")
            self._prev_kp = np.array([p.pt for p in kp], dtype=np.float32)
            self._prev_desc = desc
            return None

        # Get matched points
        pts1 = np.array([self._prev_kp[m.queryIdx] for m in matches], dtype=np.float32)
        pts2 = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Estimate motion
        delta = self._estimate_motion(pts1, pts2, depth)
        if delta is None:
            return None

        # Update pose
        self._pose = self._pose + delta

        # Update previous frame
        self._prev_kp = np.array([p.pt for p in kp], dtype=np.float32)
        self._prev_desc = desc

        return self._pose

    def reset(self) -> None:
        """Reset odometry to origin."""
        self._pose = Pose2D()
        self._prev_kp = None
        self._prev_desc = None
        self._initialized = False

    @property
    def pose(self) -> Pose2D:
        return self._pose

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _estimate_motion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Optional[Pose2D]:
        """Estimate 2D motion between point sets."""
        # Undistort points
        # Try essential matrix for general motion
        if len(pts1) >= 8:
            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                focal=self._fx,
                pp=(self._cx, self._cy),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if E is not None:
                _, R, t, mask = cv2.recoverPose(
                    E,
                    pts1,
                    pts2,
                    focal=self._fx,
                    pp=(self._cx, self._cy),
                )

                # Extract yaw from rotation
                yaw = np.arctan2(R[1, 0], R[0, 0])

                # Get scale from depth if available
                scale = 1.0
                if depth is not None and len(mask) == len(pts1):
                    valid_pts = pts1[mask.ravel() == 1]
                    depths = []
                    for pt in valid_pts:
                        px, py = int(pt[0]), int(pt[1])
                        if 0 <= px < depth.shape[1] and 0 <= py < depth.shape[0]:
                            d = depth[py, px]
                            if d > 0:
                                depths.append(d)
                    if depths:
                        scale = np.median(depths)

                return Pose2D(
                    x=float(t[0]) * scale,
                    y=float(t[1]) * scale,
                    theta=float(yaw),
                )

        # Fallback: simple translation estimate
        dx = np.median(pts2[:, 0] - pts1[:, 0])
        dy = np.median(pts2[:, 1] - pts1[:, 1])

        # Convert pixels to meters (approximate)
        scale = 0.001  # rough pixel-to-meter ratio
        return Pose2D(
            x=dx * scale,
            y=dy * scale,
            theta=0.0,
        )

    def draw_matches(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """Draw feature matches between two frames for visualization."""
        if self._prev_kp is None or self._prev_desc is None:
            return frame2

        gray2 = self._to_gray(frame2)

        kp1 = [cv2.KeyPoint(x, y, 1) for x, y in self._prev_kp]
        kp2, desc2 = self._orb.detectAndCompute(gray2, None)

        if kp2 is None or desc2 is None:
            return frame2

        matches = self._matcher.match(self._prev_desc, desc2)
        matches = sorted(matches, key=lambda m: m.distance)[:50]

        return cv2.drawMatches(
            frame1,
            kp1,
            frame2,
            kp2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
