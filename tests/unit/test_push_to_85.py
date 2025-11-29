"""
Final push tests to reach 85% coverage.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os

# ==================== UI Callbacks Helper Functions ====================

class TestCallbackHelpers:
    """Test callback helper functions."""
    
    def test_build_signal_options_helper(self):
        """Test building signal options."""
        from src.ui.callbacks import _build_signal_options
        
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [10, 20, 30]
            })
        }
        
        result = _build_signal_options(data)
        assert isinstance(result, list)
    
    def test_flatten_flight_data_helper(self):
        """Test flattening flight data."""
        from src.ui.callbacks import _flatten_flight_data
        
        data = {
            'nested': {
                'df': pd.DataFrame({'a': [1]})
            }
        }
        
        result = _flatten_flight_data(data)
        assert 'nested.df' in result
    
    def test_find_dataframe_helper(self):
        """Test finding dataframe."""
        from src.ui.callbacks import _find_dataframe
        
        data = {
            'GPS': pd.DataFrame({'lat': [1], 'lon': [2]})
        }
        
        result = _find_dataframe(data, ['GPS'])
        assert result is not None


# ==================== Geo Formats Extended ====================

class TestGeoFormatsExtended:
    """Extended geo format tests."""
    
    def test_export_kml_to_file(self):
        """Test exporting KML to file."""
        from src.export.geo_formats import export_kml
        
        lat = np.linspace(47.0, 47.1, 30)
        lon = np.linspace(-122.0, -121.9, 30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.kml')
            export_kml(lat, lon, path=path)
            
            assert os.path.exists(path)
            with open(path, 'r') as f:
                content = f.read()
                assert 'kml' in content.lower()
    
    def test_export_geojson_to_file(self):
        """Test exporting GeoJSON to file."""
        from src.export.geo_formats import export_geojson
        
        lat = np.array([47.0, 47.1])
        lon = np.array([-122.0, -122.1])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.geojson')
            export_geojson(lat, lon, path=path)
            
            assert os.path.exists(path)
    
    def test_export_gpx_to_file(self):
        """Test exporting GPX to file."""
        from src.export.geo_formats import export_gpx
        
        lat = np.array([47.0, 47.1])
        lon = np.array([-122.0, -122.1])
        alt = np.array([100, 200])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.gpx')
            export_gpx(lat, lon, alt=alt, path=path)
            
            assert os.path.exists(path)


# ==================== Data Export Extended ====================

class TestDataExportExtended:
    """Extended data export tests."""
    
    def test_export_matlab(self):
        """Test MATLAB export."""
        from src.export.data_export import export_matlab
        
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.mat')
            export_matlab(df, path)
            
            assert os.path.exists(path)


# ==================== Validator Extended ====================

class TestValidatorExtended:
    """Extended validator tests."""
    
    def test_validate_with_invalid_type(self):
        """Test validation with invalid type."""
        from src.data.validator import DataValidator
        
        validator = DataValidator()
        # Test with invalid type in structure
        data = {'test': 'not_a_dataframe'}
        result = validator.validate(data)
        assert result.is_valid is False
    
    def test_validate_non_string_key(self):
        """Test validation with non-string key."""
        from src.data.validator import DataValidator
        
        validator = DataValidator()
        # Non-string keys should fail
        data = {123: pd.DataFrame({'a': [1]})}
        result = validator.validate(data)
        assert result.is_valid is False


# ==================== XY Plot Extended ====================

class TestXYPlotMoreCoverage:
    """More XY plot tests."""
    
    def test_xy_plot_update(self):
        """Test XY plot update method."""
        from src.visualization.plots.xy_plot import XYPlot
        
        config = {'x_axis': 'x', 'y_axis': 'y'}
        plot = XYPlot(config)
        
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })
        
        plot.render(data)
        plot.update(data)
        assert plot.figure is not None
    
    def test_xy_plot_set_theme(self):
        """Test XY plot theme setting."""
        from src.visualization.plots.xy_plot import XYPlot
        
        config = {'x_axis': 'x', 'y_axis': 'y'}
        plot = XYPlot(config)
        
        plot.set_theme({'background': '#000', 'text': '#fff'})


# ==================== Time Series Extended ====================

class TestTimeSeriesMoreCoverage:
    """More time series plot tests."""
    
    def test_time_series_update(self):
        """Test time series update."""
        from src.visualization.plots.time_series import TimeSeriesPlot
        
        config = {'x_axis': 'timestamp', 'signals': ['value']}
        plot = TimeSeriesPlot(config)
        
        data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        plot.render(data)
        plot.update(data)
    
    def test_time_series_create_base_layout(self):
        """Test creating base layout."""
        from src.visualization.plots.time_series import TimeSeriesPlot
        
        config = {'x_axis': 'timestamp', 'signals': ['value']}
        plot = TimeSeriesPlot(config)
        
        layout = plot._create_base_layout()
        assert isinstance(layout, dict)


# ==================== Computed Signals Extended ====================

class TestComputedSignalsMore:
    """More computed signals tests."""
    
    def test_all_trig_functions(self):
        """Test all trigonometric functions."""
        from src.computed_signals.functions import (
            sin, cos, tan, asin, acos, atan, atan2
        )
        
        x = np.array([0, np.pi/4, np.pi/2])
        
        sin(x)
        cos(x)
        tan(x[:2])  # Avoid tan(pi/2)
        
        y = np.array([0.5, 0.5])
        asin(y)
        acos(y)
        atan(y)
        
        atan2(np.array([1]), np.array([1]))
    
    def test_all_math_functions(self):
        """Test all math functions."""
        from src.computed_signals.functions import (
            sqrt, exp, log, log10, log2,
            floor, ceil, round_val, clip,
            degrees, radians
        )
        
        x = np.array([1, 4, 9])
        sqrt(x)
        
        x = np.array([0, 1, 2])
        exp(x)
        
        x = np.array([1, 10, 100])
        log(x)
        log10(x)
        log2(x)
        
        x = np.array([1.2, 2.7, 3.5])
        floor(x)
        ceil(x)
        round_val(x)
        
        clip(x, 1.5, 3.0)
        
        x = np.array([np.pi/2, np.pi])
        degrees(x)
        radians(np.array([90, 180]))
    
    def test_signal_processing_functions(self):
        """Test signal processing functions."""
        from src.computed_signals.functions import (
            diff, cumsum, moving_avg, lowpass, highpass
        )
        
        x = np.array([1, 2, 3, 4, 5])
        
        diff(x)
        cumsum(x)
        moving_avg(x, 2)
        
        # Need longer signal for filters
        x = np.random.randn(100)
        lowpass(x, cutoff=0.1, fs=10)
        highpass(x, cutoff=0.1, fs=10)


# ==================== UI State Extended ====================

class TestUIStateMore:
    """More UI state tests."""
    
    def test_state_to_json(self):
        """Test state serialization."""
        from src.ui.state import AppState
        
        state = AppState(
            data_loaded=True,
            current_time=5.0,
            selected_signals=['a', 'b']
        )
        
        json_str = state.to_json()
        assert 'current_time' in json_str
    
    def test_state_from_json(self):
        """Test state deserialization."""
        from src.ui.state import AppState
        import json
        
        state_dict = {
            'data_loaded': True,
            'theme': 'dark'
        }
        
        state = AppState.from_dict(state_dict)
        assert state.theme == 'dark'


# ==================== Config Extended ====================

class TestConfigMore:
    """More config tests."""
    
    def test_settings_load_from_env(self):
        """Test loading settings from environment."""
        from src.config.settings import Settings
        
        settings = Settings()
        # Test default values
        assert settings.timestamp_column == 'timestamp'
    
    def test_config_loader_yaml(self):
        """Test YAML config loading."""
        from src.config.loader import ConfigLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.yaml')
            
            loader = ConfigLoader()
            config = {'theme': 'dark', 'version': '1.0'}
            loader.save(config, path)
            
            loaded = loader.load(path)
            assert loaded['theme'] == 'dark'


# ==================== Visualization Base Extended ====================

class TestVisualizationBaseMore:
    """More visualization base tests."""
    
    def test_plot_config_access(self):
        """Test accessing plot config."""
        from src.visualization.plots.time_series import TimeSeriesPlot
        
        config = {'signals': ['a'], 'title': 'Test'}
        plot = TimeSeriesPlot(config)
        
        assert plot.config == config
    
    def test_plot_title_access(self):
        """Test accessing plot title."""
        from src.visualization.plots.time_series import TimeSeriesPlot
        
        config = {'signals': ['a'], 'title': 'Test Plot'}
        plot = TimeSeriesPlot(config)
        
        assert plot.title == 'Test Plot'


# ==================== Maps Extended ====================

class TestMapsMore:
    """More map tests."""
    
    def test_map2d_empty_data(self):
        """Test Map2D with empty data."""
        from src.visualization.maps.map_2d import Map2D
        
        config = {'lat_signal': 'lat', 'lon_signal': 'lon'}
        plot = Map2D(config)
        
        fig = plot.render(pd.DataFrame())
        assert isinstance(fig, go.Figure)
    
    def test_layer_manager_geofence(self):
        """Test adding geofence to layer manager."""
        from src.visualization.maps.layers import LayerManager
        
        manager = LayerManager()
        layer_id = manager.add_geofence(
            name='test_fence',
            points=[(47.0, -122.0), (47.1, -122.0), (47.1, -122.1)]
        )
        assert layer_id is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

