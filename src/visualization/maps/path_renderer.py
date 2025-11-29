"""
Flight path rendering utilities.

Provides path styling, segmentation, and color mapping for map display.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PathSegment:
    """Represents a colored path segment."""
    lat: np.ndarray
    lon: np.ndarray
    color: str
    value: Optional[float] = None


class PathRenderer:
    """Renders flight paths with various styling options."""

    COLORMAPS = {
        "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
        "plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
        "altitude": ["#2ecc71", "#f1c40f", "#e74c3c"],
        "speed": ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"],
    }

    def __init__(self, colormap: str = "viridis"):
        self.colormap = colormap

    def segment_by_value(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        values: np.ndarray,
        n_segments: int = 10
    ) -> List[PathSegment]:
        """Segment path by value ranges."""
        if len(lat) == 0:
            return []

        v_min, v_max = np.nanmin(values), np.nanmax(values)
        if v_max == v_min:
            return [PathSegment(lat, lon, self._get_color(0.5))]

        segments = []
        boundaries = np.linspace(v_min, v_max, n_segments + 1)

        for i in range(n_segments):
            mask = (values >= boundaries[i]) & (values < boundaries[i + 1])
            if i == n_segments - 1:
                mask |= (values == v_max)

            if mask.any():
                color = self._get_color(i / (n_segments - 1))
                segments.append(PathSegment(
                    lat=lat[mask],
                    lon=lon[mask],
                    color=color,
                    value=(boundaries[i] + boundaries[i + 1]) / 2
                ))

        return segments

    def _get_color(self, t: float) -> str:
        """Get color from colormap at position t (0-1)."""
        colors = self.COLORMAPS.get(self.colormap, self.COLORMAPS["viridis"])
        n = len(colors)
        idx = min(int(t * (n - 1)), n - 2)
        return colors[idx]

    def create_time_gradient(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        n_segments: int = 20
    ) -> List[PathSegment]:
        """Create time-based gradient coloring."""
        n = len(lat)
        if n == 0:
            return []

        segment_size = max(1, n // n_segments)
        segments = []

        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size + 1, n)
            color = self._get_color(i / (n_segments - 1))
            segments.append(PathSegment(
                lat=lat[start:end],
                lon=lon[start:end],
                color=color
            ))

        return segments

