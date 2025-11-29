"""
Geographic utility functions.

Provides calculations for distance, bearing, and coordinate transformations.
"""

import numpy as np
from typing import Tuple, Union, Optional

# Earth radius in meters
EARTH_RADIUS = 6371000


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate haversine distance between points.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS * c


def cumulative_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative distance along path.

    Args:
        lat: Latitude array in degrees.
        lon: Longitude array in degrees.

    Returns:
        Cumulative distance array in meters.
    """
    if len(lat) < 2:
        return np.zeros_like(lat)

    distances = np.zeros(len(lat))
    segment_dist = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    distances[1:] = np.cumsum(segment_dist)

    return distances


def bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate initial bearing between two points.

    Args:
        lat1, lon1: Start point in degrees.
        lat2, lon2: End point in degrees.

    Returns:
        Bearing in degrees (0-360).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing_rad = np.arctan2(x, y)
    return (np.degrees(bearing_rad) + 360) % 360


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """
    Convert lat/lon to UTM coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Tuple of (easting, northing, zone_number, zone_letter).
    """
    # Simplified UTM conversion
    zone_number = int((lon + 180) / 6) + 1

    # Zone letter
    if lat >= 72:
        zone_letter = 'X'
    elif lat >= 64:
        zone_letter = 'W'
    elif lat >= 56:
        zone_letter = 'V'
    elif lat >= 48:
        zone_letter = 'U'
    elif lat >= 40:
        zone_letter = 'T'
    elif lat >= 32:
        zone_letter = 'S'
    elif lat >= 24:
        zone_letter = 'R'
    elif lat >= 16:
        zone_letter = 'Q'
    elif lat >= 8:
        zone_letter = 'P'
    elif lat >= 0:
        zone_letter = 'N'
    else:
        zone_letter = 'M' if lat >= -8 else 'L'

    # Simplified conversion (for full accuracy use pyproj)
    lon_origin = (zone_number - 1) * 6 - 180 + 3
    easting = 500000 + (lon - lon_origin) * 111320 * np.cos(np.radians(lat))
    northing = lat * 110540

    if lat < 0:
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def bounding_box(
    lat: np.ndarray,
    lon: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box.

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon).
    """
    return (
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
        float(np.nanmin(lon)),
        float(np.nanmax(lon))
    )


def point_in_polygon(
    lat: float,
    lon: float,
    polygon: list
) -> bool:
    """
    Check if point is inside polygon.

    Args:
        lat, lon: Point coordinates.
        polygon: List of (lat, lon) tuples.

    Returns:
        True if point is inside polygon.
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        if ((polygon[i][0] > lat) != (polygon[j][0] > lat) and
            lon < (polygon[j][1] - polygon[i][1]) * (lat - polygon[i][0]) /
                  (polygon[j][0] - polygon[i][0]) + polygon[i][1]):
            inside = not inside
        j = i

    return inside

