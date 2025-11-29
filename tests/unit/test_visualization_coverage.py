"""
Tests to improve visualization module coverage.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.base import BasePlot
from src.visualization.manager import PlotManager
from src.visualization.theme import Theme, ThemeMode
from src.visualization.maps.map_2d import Map2D
from src.visualization.maps.map_3d import Map3D
from src.visualization.maps.layers import Layer, LayerType, LayerManager
from src.visualization.maps.path_renderer import PathRenderer
from src.visualization.plots.fft import FFTPlot
from src.visualization.plots.histogram import HistogramPlot
from src.visualization.plots.statistics import StatisticsPlot


@pytest.fixture
def sample_data():
    """Create sample data."""
    n = 200
    t = np.linspace(0, 20, n)
    return pd.DataFrame({
        'timestamp': t,
        'signal1': np.sin(t) + np.random.randn(n) * 0.1,
        'signal2': np.cos(t) + np.random.randn(n) * 0.1,
    })


@pytest.fixture
def gps_data():
    """Create GPS data."""
    n = 100
    return pd.DataFrame({
        'timestamp': np.linspace(0, 100, n),
        'lat': np.linspace(47.0, 47.1, n),
        'lon': np.linspace(-122.0, -121.9, n),
        'alt': np.linspace(100, 200, n),
    })


class TestPlotManager:
    """Test PlotManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = PlotManager()
        assert manager is not None
    
    def test_create_time_series_plot(self):
        """Test creating time series plot."""
        manager = PlotManager()
        config = {
            'id': 'plot1',
            'plot_type': 'TIME_SERIES',
            'signals': ['signal1']
        }
        plot = manager.create_plot(config)
        assert plot is not None
    
    def test_create_xy_plot(self):
        """Test creating XY plot."""
        manager = PlotManager()
        config = {
            'id': 'plot2',
            'plot_type': 'XY_SCATTER',
            'x_axis': 'x',
            'y_axis': 'y'
        }
        plot = manager.create_plot(config)
        assert plot is not None
    
    def test_get_plot(self):
        """Test getting plot by ID."""
        manager = PlotManager()
        config = {'id': 'test_plot', 'plot_type': 'TIME_SERIES'}
        manager.create_plot(config)
        plot = manager.get_plot('test_plot')
        assert plot is not None
    
    def test_remove_plot(self):
        """Test removing plot."""
        manager = PlotManager()
        config = {'id': 'to_remove', 'plot_type': 'TIME_SERIES'}
        manager.create_plot(config)
        result = manager.remove_plot('to_remove')
        assert result is True
        assert manager.get_plot('to_remove') is None
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent plot."""
        manager = PlotManager()
        result = manager.remove_plot('doesnt_exist')
        assert result is False
    
    def test_set_theme(self):
        """Test setting theme."""
        manager = PlotManager()
        theme = {'background': '#000', 'text': '#fff'}
        manager.set_theme(theme)


class TestTheme:
    """Test Theme class."""
    
    def test_default_theme(self):
        """Test default theme."""
        theme = Theme()
        assert theme.name == 'default'
        assert theme.mode == ThemeMode.LIGHT
    
    def test_dark_mode(self):
        """Test dark mode theme."""
        theme = Theme(mode=ThemeMode.DARK, name='dark')
        assert theme.mode == ThemeMode.DARK
    
    def test_to_plotly_template(self):
        """Test converting to Plotly template."""
        theme = Theme()
        template = theme.to_plotly_template()
        assert isinstance(template, dict)
        assert 'layout' in template
    
    def test_signal_colors(self):
        """Test signal colors."""
        theme = Theme()
        assert len(theme.signal_colors) > 0
    
    def test_event_colors(self):
        """Test event colors by severity."""
        theme = Theme()
        assert 'critical' in theme.event_colors
        assert 'warning' in theme.event_colors


class TestMap2D:
    """Test Map2D class."""
    
    def test_init(self):
        """Test initialization."""
        map_plot = Map2D({})
        assert map_plot is not None
    
    def test_render(self, gps_data):
        """Test rendering."""
        config = {'lat_signal': 'lat', 'lon_signal': 'lon'}
        map_plot = Map2D(config)
        fig = map_plot.render(gps_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_empty(self):
        """Test rendering empty data."""
        config = {'lat_signal': 'lat', 'lon_signal': 'lon'}
        map_plot = Map2D(config)
        fig = map_plot.render(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestMap3D:
    """Test Map3D class."""
    
    def test_init(self):
        """Test initialization."""
        map_plot = Map3D({})
        assert map_plot is not None
    
    def test_render(self, gps_data):
        """Test rendering."""
        config = {
            'lat_column': 'lat',
            'lon_column': 'lon',
            'alt_column': 'alt'
        }
        map_plot = Map3D(config)
        fig = map_plot.render(gps_data)
        assert isinstance(fig, go.Figure)


class TestLayerManager:
    """Test LayerManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = LayerManager()
        assert manager is not None
    
    def test_add_layer(self):
        """Test adding layer."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.GRID)
        manager.add_layer(layer)
        assert manager.get_layer('test') is not None
    
    def test_remove_layer(self):
        """Test removing layer."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.GRID)
        manager.add_layer(layer)
        result = manager.remove_layer('test')
        assert result is True
    
    def test_set_visibility(self):
        """Test setting visibility."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.GRID, visible=True)
        manager.add_layer(layer)
        manager.set_visibility('test', False)
        assert manager.get_layer('test').visible is False
    
    def test_get_visible_layers(self):
        """Test getting visible layers."""
        manager = LayerManager()
        manager.add_layer(Layer(id='visible', layer_type=LayerType.GRID, visible=True))
        manager.add_layer(Layer(id='hidden', layer_type=LayerType.GRID, visible=False))
        visible = manager.get_visible_layers()
        assert len(visible) == 1
    
    def test_clear(self):
        """Test clearing all layers."""
        manager = LayerManager()
        manager.add_layer(Layer(id='test1', layer_type=LayerType.GRID))
        manager.add_layer(Layer(id='test2', layer_type=LayerType.GRID))
        manager.clear()
        assert manager.get_layer('test1') is None


class TestPathRenderer:
    """Test PathRenderer class."""
    
    def test_init(self):
        """Test initialization."""
        renderer = PathRenderer()
        assert renderer is not None
    
    def test_init_custom_colormap(self):
        """Test custom colormap."""
        renderer = PathRenderer(colormap='plasma')
        assert renderer.colormap == 'plasma'
    
    def test_segment_by_value(self, gps_data):
        """Test segmenting by value."""
        renderer = PathRenderer()
        segments = renderer.segment_by_value(
            gps_data['lat'].values,
            gps_data['lon'].values,
            gps_data['alt'].values,
            n_segments=5
        )
        assert len(segments) > 0
    
    def test_create_time_gradient(self, gps_data):
        """Test creating time gradient."""
        renderer = PathRenderer()
        segments = renderer.create_time_gradient(
            gps_data['lat'].values,
            gps_data['lon'].values,
            n_segments=10
        )
        assert len(segments) > 0
    
    def test_empty_path(self):
        """Test empty path."""
        renderer = PathRenderer()
        segments = renderer.segment_by_value(
            np.array([]),
            np.array([]),
            np.array([])
        )
        assert len(segments) == 0


class TestHistogramPlot:
    """Test HistogramPlot class."""
    
    def test_init(self):
        """Test initialization."""
        plot = HistogramPlot({})
        assert plot is not None
    
    def test_render(self, sample_data):
        """Test rendering."""
        config = {'signals': ['signal1']}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_multiple_signals(self, sample_data):
        """Test with multiple signals."""
        config = {'signals': ['signal1', 'signal2']}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        assert len(fig.data) >= 2


class TestStatisticsPlot:
    """Test StatisticsPlot class."""
    
    def test_init(self):
        """Test initialization."""
        plot = StatisticsPlot({})
        assert plot is not None
    
    def test_render_boxplot(self, sample_data):
        """Test boxplot rendering."""
        config = {'signals': ['signal1', 'signal2'], 'stat_type': 'box'}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_violin(self, sample_data):
        """Test violin plot rendering."""
        config = {'signals': ['signal1'], 'stat_type': 'violin'}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)


class TestFFTPlot:
    """Test FFTPlot class."""
    
    def test_init(self):
        """Test initialization."""
        plot = FFTPlot({})
        assert plot is not None
    
    def test_render(self, sample_data):
        """Test rendering."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal1']
        }
        plot = FFTPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

