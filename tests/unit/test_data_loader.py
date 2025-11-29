"""Tests for data loading functionality."""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import DataLoader, load_flight_data
from src.core.exceptions import InvalidDataStructure, TimestampError


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_flat_structure(self, sample_dataframe):
        """Test loading flat data structure."""
        data = {'TestData': sample_dataframe}
        loader = DataLoader().load(data)

        assert len(loader.list_dataframes()) == 1
        assert 'TestData' in loader.list_dataframes()

    def test_load_hierarchical_structure(self, sample_flight_data):
        """Test loading hierarchical data structure."""
        loader = DataLoader().load(sample_flight_data)

        df_paths = loader.list_dataframes()
        assert 'Sensors.IMU.Accelerometer' in df_paths
        assert 'Sensors.IMU.Gyroscope' in df_paths
        assert 'Sensors.GPS' in df_paths
        assert 'Events' in df_paths

    def test_get_dataframe(self, sample_flight_data):
        """Test getting DataFrame by path."""
        loader = DataLoader().load(sample_flight_data)

        gps_df = loader.get_dataframe('Sensors.GPS')
        assert isinstance(gps_df, pd.DataFrame)
        assert 'lat' in gps_df.columns

    def test_get_signal(self, sample_flight_data):
        """Test getting signal by path."""
        loader = DataLoader().load(sample_flight_data)

        lat_signal = loader.get_signal('Sensors.GPS.lat')
        assert isinstance(lat_signal, pd.Series)

    def test_missing_timestamp_error(self):
        """Test error on missing timestamp column."""
        data = {'Bad': pd.DataFrame({'x': [1, 2, 3]})}

        with pytest.raises(TimestampError):
            DataLoader().load(data)

    def test_invalid_structure_error(self):
        """Test error on invalid structure."""
        data = {'Bad': [1, 2, 3]}  # List instead of DataFrame

        with pytest.raises(InvalidDataStructure):
            DataLoader().load(data)

    def test_get_time_range(self, sample_flight_data):
        """Test getting overall time range."""
        loader = DataLoader().load(sample_flight_data)

        t_min, t_max = loader.get_time_range()
        assert t_min == 0.0
        assert t_max > 0

    def test_list_signals(self, sample_flight_data):
        """Test listing all signals."""
        loader = DataLoader().load(sample_flight_data)

        signals = loader.list_signals()
        assert len(signals) > 0

        signal_names = [s.name for s in signals]
        assert 'lat' in signal_names
        assert 'accel_x' in signal_names


class TestLoadFlightData:
    """Tests for convenience function."""

    def test_load_flight_data_function(self, sample_flight_data):
        """Test load_flight_data convenience function."""
        loader = load_flight_data(sample_flight_data)
        assert len(loader.list_dataframes()) > 0

