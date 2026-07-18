"""Map manager for SLAM map persistence and visualization."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapMetadata:
    """Metadata for a saved SLAM map."""

    name: str
    created_at: float = 0.0
    updated_at: float = 0.0
    keyframe_count: int = 0
    map_point_count: int = 0
    area_covered_m2: float = 0.0
    description: str = ""


@dataclass
class Keyframe:
    """A keyframe in the SLAM map."""

    id: int
    position: list[float]  # [x, y, z]
    orientation: list[float]  # [x, y, z, w] quaternion
    timestamp: float
    rgb_path: str = ""
    depth_path: str = ""


@dataclass
class SavedMap:
    """Complete saved map data."""

    metadata: MapMetadata
    keyframes: list[Keyframe] = field(default_factory=list)
    trajectory: list[list[float]] = field(default_factory=list)  # [[x, y, yaw], ...]


class MapManager:
    """Manages SLAM map saving, loading, and listing."""

    def __init__(self, maps_dir: str = "maps"):
        self.maps_dir = Path(maps_dir)
        self.maps_dir.mkdir(parents=True, exist_ok=True)

    def save_map(
        self,
        name: str,
        trajectory: list[list[float]],
        keyframes: Optional[list[Keyframe]] = None,
        description: str = "",
    ) -> str:
        """Save current map to disk.

        Args:
            name: Map name
            trajectory: List of [x, y, yaw] poses
            keyframes: Optional list of keyframe data
            description: Optional description

        Returns:
            Path to saved map file
        """
        # Compute metadata
        if len(trajectory) > 1:
            positions = np.array([t[:2] for t in trajectory])
            # Approximate area as bounding box
            dx = positions[:, 0].max() - positions[:, 0].min()
            dy = positions[:, 1].max() - positions[:, 1].min()
            area = dx * dy
        else:
            area = 0.0

        metadata = MapMetadata(
            name=name,
            created_at=time.time(),
            updated_at=time.time(),
            keyframe_count=len(keyframes) if keyframes else 0,
            map_point_count=0,  # Would come from SLAM system
            area_covered_m2=round(area, 2),
            description=description,
        )

        saved_map = SavedMap(
            metadata=metadata,
            keyframes=keyframes or [],
            trajectory=trajectory,
        )

        # Save to JSON
        map_path = self.maps_dir / f"{name}.json"
        with open(map_path, "w") as f:
            json.dump(asdict(saved_map), f, indent=2)

        logger.info(f"Map '{name}' saved to {map_path} ({len(trajectory)} poses, {area:.1f} m²)")
        return str(map_path)

    def load_map(self, name: str) -> Optional[SavedMap]:
        """Load a map from disk.

        Args:
            name: Map name (without .json extension)

        Returns:
            SavedMap or None if not found
        """
        map_path = self.maps_dir / f"{name}.json"
        if not map_path.exists():
            logger.warning(f"Map '{name}' not found at {map_path}")
            return None

        try:
            with open(map_path) as f:
                data = json.load(f)

            # Reconstruct objects
            metadata = MapMetadata(**data["metadata"])
            keyframes = [Keyframe(**kf) for kf in data.get("keyframes", [])]
            trajectory = data.get("trajectory", [])

            return SavedMap(
                metadata=metadata,
                keyframes=keyframes,
                trajectory=trajectory,
            )
        except Exception as e:
            logger.error(f"Failed to load map '{name}': {e}")
            return None

    def list_maps(self) -> list[MapMetadata]:
        """List all saved maps.

        Returns:
            List of map metadata
        """
        maps = []
        for map_file in self.maps_dir.glob("*.json"):
            try:
                with open(map_file) as f:
                    data = json.load(f)
                maps.append(MapMetadata(**data["metadata"]))
            except Exception as e:
                logger.warning(f"Failed to read map {map_file}: {e}")
        return sorted(maps, key=lambda m: m.updated_at, reverse=True)

    def delete_map(self, name: str) -> bool:
        """Delete a saved map.

        Args:
            name: Map name

        Returns:
            True if deleted
        """
        map_path = self.maps_dir / f"{name}.json"
        if map_path.exists():
            map_path.unlink()
            logger.info(f"Map '{name}' deleted")
            return True
        logger.warning(f"Map '{name}' not found")
        return False

    def get_trajectory(self, name: str) -> Optional[np.ndarray]:
        """Get trajectory as numpy array.

        Args:
            name: Map name

        Returns:
            Nx3 array of [x, y, yaw] or None
        """
        saved_map = self.load_map(name)
        if saved_map is None:
            return None
        return np.array(saved_map.trajectory)