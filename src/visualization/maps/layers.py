"""
Map layer management.

Handles overlay layers for maps including geofences, waypoints, and custom shapes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class LayerType(Enum):
    """Layer type options."""
    GEOFENCE = "geofence"
    WAYPOINTS = "waypoints"
    GRID = "grid"
    HEATMAP = "heatmap"
    CUSTOM = "custom"


@dataclass
class GeoFence:
    """Geofence definition."""
    name: str
    points: List[tuple]  # List of (lat, lon) tuples
    color: str = "#e74c3c"
    fill_opacity: float = 0.2
    line_width: int = 2


@dataclass
class Layer:
    """Map layer definition."""
    id: str
    layer_type: LayerType
    visible: bool = True
    data: Any = None
    style: Dict[str, Any] = field(default_factory=dict)


class LayerManager:
    """Manages map overlay layers."""

    def __init__(self):
        self._layers: Dict[str, Layer] = {}

    def add_layer(self, layer: Layer) -> None:
        """Add a layer."""
        self._layers[layer.id] = layer

    def remove_layer(self, layer_id: str) -> bool:
        """Remove layer by ID."""
        if layer_id in self._layers:
            del self._layers[layer_id]
            return True
        return False

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """Get layer by ID."""
        return self._layers.get(layer_id)

    def set_visibility(self, layer_id: str, visible: bool) -> None:
        """Set layer visibility."""
        if layer_id in self._layers:
            self._layers[layer_id].visible = visible

    def get_visible_layers(self) -> List[Layer]:
        """Get all visible layers."""
        return [l for l in self._layers.values() if l.visible]

    def add_geofence(
        self,
        name: str,
        points: List[tuple],
        **style_kwargs
    ) -> str:
        """Add a geofence layer."""
        layer_id = f"geofence_{name}"
        geofence = GeoFence(name=name, points=points, **style_kwargs)
        self.add_layer(Layer(
            id=layer_id,
            layer_type=LayerType.GEOFENCE,
            data=geofence
        ))
        return layer_id

    def check_geofence_violation(
        self,
        lat: float,
        lon: float,
        geofence: GeoFence
    ) -> bool:
        """Check if point is outside geofence."""
        # Simple point-in-polygon test
        points = geofence.points
        n = len(points)
        inside = False

        j = n - 1
        for i in range(n):
            if ((points[i][0] > lat) != (points[j][0] > lat) and
                lon < (points[j][1] - points[i][1]) * (lat - points[i][0]) /
                      (points[j][0] - points[i][0]) + points[i][1]):
                inside = not inside
            j = i

        return not inside

    def clear(self) -> None:
        """Clear all layers."""
        self._layers.clear()

