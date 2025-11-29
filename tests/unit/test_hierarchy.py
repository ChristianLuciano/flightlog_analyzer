"""Tests for hierarchy navigation."""

import pytest
import pandas as pd

from src.data.hierarchy import HierarchyNavigator, resolve_path
from src.core.exceptions import HierarchyError


class TestHierarchyNavigator:
    """Tests for HierarchyNavigator class."""

    def test_resolve_path(self, sample_flight_data):
        """Test path resolution."""
        nav = HierarchyNavigator(sample_flight_data)

        # Resolve to DataFrame
        gps = nav.resolve('Sensors.GPS')
        assert isinstance(gps, pd.DataFrame)

        # Resolve to nested dict
        imu = nav.resolve('Sensors.IMU')
        assert isinstance(imu, dict)

    def test_resolve_signal(self, sample_flight_data):
        """Test resolving to signal level."""
        nav = HierarchyNavigator(sample_flight_data)

        lat = nav.resolve('Sensors.GPS.lat')
        assert isinstance(lat, pd.Series)

    def test_invalid_path_error(self, sample_flight_data):
        """Test error on invalid path."""
        nav = HierarchyNavigator(sample_flight_data)

        with pytest.raises(HierarchyError):
            nav.resolve('Invalid.Path')

    def test_exists(self, sample_flight_data):
        """Test path existence check."""
        nav = HierarchyNavigator(sample_flight_data)

        assert nav.exists('Sensors.GPS')
        assert nav.exists('Sensors.IMU.Accelerometer')
        assert not nav.exists('Nonexistent')

    def test_list_children(self, sample_flight_data):
        """Test listing children."""
        nav = HierarchyNavigator(sample_flight_data)

        # Root level
        root_children = nav.list_children('')
        assert 'Sensors' in root_children
        assert 'Events' in root_children

        # Nested level
        imu_children = nav.list_children('Sensors.IMU')
        assert 'Accelerometer' in imu_children
        assert 'Gyroscope' in imu_children

    def test_search(self, sample_flight_data):
        """Test path search."""
        nav = HierarchyNavigator(sample_flight_data)

        # Fuzzy search
        results = nav.search('accel')
        assert any('accel' in r.lower() for r in results)

        # Exact search
        results = nav.search('GPS', search_type='exact')
        assert any('GPS' in r for r in results)

    def test_build_tree(self, sample_flight_data):
        """Test tree building."""
        nav = HierarchyNavigator(sample_flight_data)
        tree = nav.build_tree()

        assert tree.name == 'root'
        assert len(tree.children) > 0

    def test_get_dataframe_paths(self, sample_flight_data):
        """Test getting all DataFrame paths."""
        nav = HierarchyNavigator(sample_flight_data)
        paths = nav.get_dataframe_paths()

        assert 'Sensors.GPS' in paths
        assert 'Events' in paths


class TestResolvePathFunction:
    """Tests for resolve_path convenience function."""

    def test_resolve_path_function(self, sample_flight_data):
        """Test resolve_path function."""
        result = resolve_path(sample_flight_data, 'Sensors.GPS')
        assert isinstance(result, pd.DataFrame)

