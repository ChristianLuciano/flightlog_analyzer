"""
Tests to improve callbacks.py coverage.

Focus on UI callback helper functions.
"""

import pytest
import numpy as np
import pandas as pd

from src.ui.callbacks import (
    _build_signal_options,
    _flatten_flight_data,
    _get_signal_data,
    _find_dataframe,
    _create_time_series_plot,
)


class TestBuildSignalOptions:
    """Test _build_signal_options function."""
    
    def test_empty_data(self):
        """Test with empty data."""
        result = _build_signal_options({})
        assert result == []
    
    def test_single_dataframe(self):
        """Test with single DataFrame."""
        data = {
            'df1': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [10, 20, 30]
            })
        }
        result = _build_signal_options(data)
        assert len(result) >= 1
    
    def test_nested_structure(self):
        """Test with nested structure."""
        data = {
            'Sensors': {
                'IMU': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'accel_x': [0.1, 0.2, 0.3]
                })
            }
        }
        result = _build_signal_options(data)
        assert len(result) >= 1


class TestFlattenFlightData:
    """Test _flatten_flight_data function."""
    
    def test_empty_data(self):
        """Test with empty data."""
        result = _flatten_flight_data({})
        assert result == {}
    
    def test_flat_data(self):
        """Test with flat data."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        data = {'test': df}
        result = _flatten_flight_data(data)
        assert 'test' in result
    
    def test_nested_data(self):
        """Test with nested data."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        data = {
            'level1': {
                'level2': df
            }
        }
        result = _flatten_flight_data(data)
        assert 'level1.level2' in result


class TestGetSignalData:
    """Test _get_signal_data function."""
    
    def test_simple_path(self):
        """Test with simple path."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        data = {'test': df}
        # Function signature: _get_signal_data(data, signal_path, ...)
        result = _get_signal_data(data, 'test.value')
        assert result is not None
    
    def test_nested_path(self):
        """Test with nested path."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        data = {
            'Sensors': {
                'IMU': df
            }
        }
        result = _get_signal_data(data, 'Sensors.IMU.value')
        assert result is not None
    
    def test_invalid_path(self):
        """Test with invalid path."""
        data = {'test': pd.DataFrame({'a': [1]})}
        result = _get_signal_data(data, 'invalid.path.signal')
        assert result is None


class TestFindDataframe:
    """Test _find_dataframe function."""
    
    def test_find_by_key(self):
        """Test finding by key name."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        data = {
            'GPS': df,
            'other': pd.DataFrame({'b': [4, 5]})
        }
        result = _find_dataframe(data, ['GPS'])
        assert result is not None
    
    def test_find_in_nested(self):
        """Test finding in nested structure."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        data = {
            'Sensors': {
                'GPS': df
            }
        }
        result = _find_dataframe(data, ['GPS'])
        assert result is not None
    
    def test_not_found(self):
        """Test when not found."""
        data = {'other': pd.DataFrame({'a': [1]})}
        result = _find_dataframe(data, ['GPS', 'Position'])
        assert result is None


class TestCreateTimeSeriesPlot:
    """Test _create_time_series_plot function."""
    
    def test_create_plot(self):
        """Test creating time series plot."""
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'value1': np.sin(np.linspace(0, 10, 100)),
            'value2': np.cos(np.linspace(0, 10, 100))
        })
        data = {'test': df}
        fig = _create_time_series_plot(
            signals=['test.value1', 'test.value2'],
            flight_data=data,
            time_start=None,
            time_end=None
        )
        assert fig is not None
    
    def test_create_plot_with_time_range(self):
        """Test creating plot with time range."""
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'value': np.random.randn(100)
        })
        data = {'test': df}
        fig = _create_time_series_plot(
            signals=['test.value'],
            flight_data=data,
            time_start=2.0,
            time_end=8.0
        )
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

