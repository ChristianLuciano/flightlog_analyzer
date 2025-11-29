"""Tests for geographic utilities."""

import pytest
import numpy as np

from src.utils.geo import (
    haversine_distance, cumulative_distance, bearing,
    latlon_to_utm, bounding_box, point_in_polygon
)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point(self):
        """Test distance to same point is zero."""
        dist = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
        assert dist == 0

    def test_known_distance(self):
        """Test against known distance."""
        # San Francisco to Los Angeles ~560km
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437

        dist = haversine_distance(sf_lat, sf_lon, la_lat, la_lon)
        assert 550000 < dist < 570000  # meters

    def test_vectorized(self):
        """Test vectorized computation."""
        lat1 = np.array([0, 0, 0])
        lon1 = np.array([0, 0, 0])
        lat2 = np.array([0, 0, 0])
        lon2 = np.array([1, 2, 3])

        distances = haversine_distance(lat1, lon1, lat2, lon2)

        assert len(distances) == 3
        assert distances[0] < distances[1] < distances[2]


class TestCumulativeDistance:
    """Tests for cumulative distance calculation."""

    def test_single_point(self):
        """Test with single point."""
        lat = np.array([37.7749])
        lon = np.array([-122.4194])

        distances = cumulative_distance(lat, lon)
        assert len(distances) == 1
        assert distances[0] == 0

    def test_increasing(self):
        """Test that cumulative distance increases."""
        lat = np.linspace(37.0, 38.0, 10)
        lon = np.linspace(-122.0, -121.0, 10)

        distances = cumulative_distance(lat, lon)

        assert distances[0] == 0
        assert all(np.diff(distances) >= 0)


class TestBearing:
    """Tests for bearing calculation."""

    def test_north(self):
        """Test bearing due north."""
        b = bearing(0, 0, 1, 0)
        assert np.isclose(b, 0, atol=1)

    def test_east(self):
        """Test bearing due east."""
        b = bearing(0, 0, 0, 1)
        assert np.isclose(b, 90, atol=1)

    def test_south(self):
        """Test bearing due south."""
        b = bearing(1, 0, 0, 0)
        assert np.isclose(b, 180, atol=1)


class TestLatLonToUTM:
    """Tests for UTM conversion."""

    def test_zone_calculation(self):
        """Test UTM zone calculation."""
        # San Francisco should be in zone 10
        easting, northing, zone, letter = latlon_to_utm(37.7749, -122.4194)

        assert zone == 10
        assert letter in ['S', 'T']
        assert easting > 0
        assert northing > 0


class TestBoundingBox:
    """Tests for bounding box calculation."""

    def test_bounding_box(self):
        """Test bounding box calculation."""
        lat = np.array([1, 2, 3, 4, 5])
        lon = np.array([10, 20, 30, 40, 50])

        min_lat, max_lat, min_lon, max_lon = bounding_box(lat, lon)

        assert min_lat == 1
        assert max_lat == 5
        assert min_lon == 10
        assert max_lon == 50


class TestPointInPolygon:
    """Tests for point-in-polygon test."""

    def test_inside_square(self):
        """Test point inside square."""
        polygon = [(0, 0), (0, 10), (10, 10), (10, 0)]

        assert point_in_polygon(5, 5, polygon)
        assert not point_in_polygon(15, 15, polygon)

    def test_on_edge(self):
        """Test point on edge (may be inside or outside)."""
        polygon = [(0, 0), (0, 10), (10, 10), (10, 0)]

        # Edge cases may vary by implementation
        result = point_in_polygon(0, 5, polygon)
        assert isinstance(result, bool)

