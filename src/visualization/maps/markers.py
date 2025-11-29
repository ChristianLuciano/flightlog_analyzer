"""
Map marker management.

Provides marker creation and styling for map visualizations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class MarkerType(Enum):
    """Marker type options."""
    START = "start"
    END = "end"
    CURRENT = "current"
    EVENT = "event"
    WAYPOINT = "waypoint"
    CUSTOM = "custom"


@dataclass
class Marker:
    """Map marker definition."""
    lat: float
    lon: float
    marker_type: MarkerType
    label: str = ""
    color: str = "#3498db"
    size: int = 10
    symbol: str = "circle"
    heading: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MarkerManager:
    """Manages map markers."""

    DEFAULT_STYLES = {
        MarkerType.START: {"color": "#2ecc71", "symbol": "circle", "size": 15},
        MarkerType.END: {"color": "#e74c3c", "symbol": "circle", "size": 15},
        MarkerType.CURRENT: {"color": "#3498db", "symbol": "triangle-up", "size": 18},
        MarkerType.EVENT: {"color": "#f39c12", "symbol": "star", "size": 12},
        MarkerType.WAYPOINT: {"color": "#9b59b6", "symbol": "diamond", "size": 10},
    }

    def __init__(self):
        self._markers: List[Marker] = []

    def add_marker(self, marker: Marker) -> None:
        """Add a marker."""
        self._markers.append(marker)

    def remove_marker(self, index: int) -> None:
        """Remove marker by index."""
        if 0 <= index < len(self._markers):
            del self._markers[index]

    def clear(self) -> None:
        """Clear all markers."""
        self._markers.clear()

    def get_markers(self, marker_type: Optional[MarkerType] = None) -> List[Marker]:
        """Get markers, optionally filtered by type."""
        if marker_type:
            return [m for m in self._markers if m.marker_type == marker_type]
        return self._markers.copy()

    def to_plotly_traces(self) -> List[Dict[str, Any]]:
        """Convert markers to Plotly trace format."""
        traces = []
        for marker in self._markers:
            style = self.DEFAULT_STYLES.get(marker.marker_type, {})
            traces.append({
                "lat": [marker.lat],
                "lon": [marker.lon],
                "mode": "markers+text",
                "marker": {
                    "size": marker.size or style.get("size", 10),
                    "color": marker.color or style.get("color", "#3498db"),
                    "symbol": marker.symbol or style.get("symbol", "circle"),
                },
                "text": [marker.label],
                "name": marker.marker_type.value,
            })
        return traces

