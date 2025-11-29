"""
Tests for Map3D and XYPlot visualization modules.

Covers src/visualization/maps/map_3d.py and src/visualization/plots/xy_plot.py
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.maps.map_3d import Map3D
from src.visualization.plots.xy_plot import XYPlot
from src.visualization.plots.time_series import TimeSeriesPlot
from src.visualization.base import BasePlot


@pytest.fixture
def gps_3d_data():
    """Create 3D GPS data."""
    n = 200
    t = np.linspace(0, 100, n)
    return pd.DataFrame({
        'timestamp': t,
        'lat': 47.0 + 0.001 * np.sin(t * 0.1),
        'lon': -122.0 + 0.001 * np.cos(t * 0.1),
        'altitude': 100 + 50 * np.sin(t * 0.05) + t * 0.5,
        'speed': 10 + 5 * np.random.randn(n),
    })


@pytest.fixture
def xy_data():
    """Create X-Y relationship data."""
    n = 500
    x = np.linspace(0, 10, n)
    return pd.DataFrame({
        'timestamp': x,
        'signal_x': np.sin(x) + np.random.randn(n) * 0.1,
        'signal_y': np.cos(x) + np.random.randn(n) * 0.1,
        'signal_z': x + np.random.randn(n) * 0.5,
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


class TestMap3D:
    """Test Map3D class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        plot = Map3D(config)
        assert plot.lat_column == 'lat'
        assert plot.lon_column == 'lon'
        assert plot.alt_column == 'altitude'
    
    def test_init_custom_columns(self):
        """Test initialization with custom columns."""
        config = {
            'lat_column': 'latitude',
            'lon_column': 'longitude',
            'alt_column': 'alt'
        }
        plot = Map3D(config)
        assert plot.lat_column == 'latitude'
        assert plot.lon_column == 'longitude'
        assert plot.alt_column == 'alt'
    
    def test_render(self, gps_3d_data):
        """Test render method."""
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude'
        }
        plot = Map3D(config)
        fig = plot.render(gps_3d_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_render_missing_columns(self, gps_3d_data):
        """Test render with missing columns."""
        config = {
            'lat_column': 'nonexistent',
            'lon_column': 'lon',
            'alt_column': 'altitude'
        }
        plot = Map3D(config)
        fig = plot.render(gps_3d_data)
        assert isinstance(fig, go.Figure)
        # Should return empty figure
        assert len(fig.data) == 0
    
    def test_render_with_nan(self):
        """Test render with NaN values."""
        data = pd.DataFrame({
            'lat': [47.0, np.nan, 47.2, 47.3],
            'lon': [-122.0, -122.1, np.nan, -122.3],
            'altitude': [100, 110, 120, np.nan],
        })
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude'
        }
        plot = Map3D(config)
        fig = plot.render(data)
        assert isinstance(fig, go.Figure)
    
    def test_render_empty_after_nan_removal(self):
        """Test render when all data is NaN."""
        data = pd.DataFrame({
            'lat': [np.nan, np.nan],
            'lon': [np.nan, np.nan],
            'altitude': [np.nan, np.nan],
        })
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude'
        }
        plot = Map3D(config)
        fig = plot.render(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_render_color_by_altitude(self, gps_3d_data):
        """Test color by altitude."""
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude',
            'color_by': 'altitude'
        }
        plot = Map3D(config)
        fig = plot.render(gps_3d_data)
        assert len(fig.data) > 0
    
    def test_render_color_by_other_column(self, gps_3d_data):
        """Test color by speed column."""
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude',
            'color_by': 'speed'
        }
        plot = Map3D(config)
        fig = plot.render(gps_3d_data)
        assert len(fig.data) > 0
    
    def test_render_large_dataset(self):
        """Test render with large dataset (triggers downsampling)."""
        n = 5000
        data = pd.DataFrame({
            'lat': np.linspace(47, 48, n),
            'lon': np.linspace(-122, -121, n),
            'altitude': np.linspace(100, 500, n),
        })
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude',
            'max_path_points': 1000
        }
        plot = Map3D(config)
        fig = plot.render(data)
        assert len(fig.data) > 0
    
    def test_update(self, gps_3d_data):
        """Test update method."""
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'altitude'
        }
        plot = Map3D(config)
        plot.render(gps_3d_data)
        plot.update(gps_3d_data)
        assert plot.figure is not None


class TestXYPlot:
    """Test XYPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        plot = XYPlot(config)
        assert plot.max_display_points == 10000
    
    def test_init_custom_max_points(self):
        """Test initialization with custom max points."""
        config = {'max_points': 5000}
        plot = XYPlot(config)
        assert plot.max_display_points == 5000
    
    def test_render_scatter(self, xy_data):
        """Test render scatter plot."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y',
            'plot_type': 'XY_SCATTER'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_render_line(self, xy_data):
        """Test render line plot."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y',
            'plot_type': 'XY_LINE'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_render_missing_x(self, xy_data):
        """Test render with missing X column."""
        config = {
            'x_axis': 'nonexistent',
            'y_axis': 'signal_y'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_render_missing_y(self, xy_data):
        """Test render with missing Y column."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'nonexistent'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_render_with_color(self, xy_data):
        """Test render with color mapping."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y',
            'color_by': 'signal_z'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert len(fig.data) > 0
    
    def test_render_with_invalid_color(self, xy_data):
        """Test render with invalid color column."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y',
            'color_by': 'nonexistent'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert len(fig.data) > 0
    
    def test_render_large_dataset(self):
        """Test render with large dataset (triggers downsampling)."""
        n = 20000
        data = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
        })
        config = {
            'x_axis': 'x',
            'y_axis': 'y',
            'max_points': 5000
        }
        plot = XYPlot(config)
        fig = plot.render(data)
        assert len(fig.data) > 0
    
    def test_render_with_colorscale(self, xy_data):
        """Test render with custom colorscale."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y',
            'color_by': 'signal_z',
            'colorscale': 'Plasma'
        }
        plot = XYPlot(config)
        fig = plot.render(xy_data)
        assert len(fig.data) > 0
    
    def test_update(self, xy_data):
        """Test update method."""
        config = {
            'x_axis': 'signal_x',
            'y_axis': 'signal_y'
        }
        plot = XYPlot(config)
        plot.render(xy_data)
        plot.update(xy_data)
        assert plot.figure is not None


class TestTimeSeriesPlotFull:
    """Additional tests for TimeSeriesPlot."""
    
    @pytest.fixture
    def ts_data(self):
        """Create time series data."""
        n = 300
        t = np.linspace(0, 30, n)
        return pd.DataFrame({
            'timestamp': t,
            'signal1': np.sin(t) + np.random.randn(n) * 0.1,
            'signal2': np.cos(t) + np.random.randn(n) * 0.1,
            'signal3': t * 0.1 + np.random.randn(n) * 0.1,
        })
    
    def test_render_multiple_signals(self, ts_data):
        """Test rendering multiple signals."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal1', 'signal2', 'signal3']
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(ts_data)
        assert len(fig.data) >= 3
    
    def test_render_no_signals_specified(self, ts_data):
        """Test render without signals specified."""
        config = {'x_axis': 'timestamp'}
        plot = TimeSeriesPlot(config)
        fig = plot.render(ts_data)
        # Should auto-detect numeric columns
        assert len(fig.data) > 0
    
    def test_render_with_time_range(self, ts_data):
        """Test render with time range."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal1'],
            'time_range': (5, 25)
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(ts_data)
        assert len(fig.data) > 0


class TestBasePlotMethods:
    """Test BasePlot methods through concrete implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50]
        })
    
    def test_set_theme(self, sample_data):
        """Test set_theme method."""
        config = {'x_axis': 'timestamp', 'signals': ['value']}
        plot = TimeSeriesPlot(config)
        theme = {'background': '#000000', 'text': '#ffffff'}
        plot.set_theme(theme)
        # Theme should be stored
        assert plot._theme == theme
    
    def test_render_and_get_figure(self, sample_data):
        """Test rendering and getting figure."""
        config = {'x_axis': 'timestamp', 'signals': ['value']}
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_data)
        assert fig is not None
        assert plot.figure is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

