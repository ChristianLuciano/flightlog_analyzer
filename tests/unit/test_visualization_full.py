"""
Tests for visualization modules.

Covers src/visualization/plots/* - 100% coverage target.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.plots.histogram import HistogramPlot
from src.visualization.plots.spectrogram import SpectrogramPlot
from src.visualization.plots.statistics import StatisticsPlot
from src.visualization.plots.time_series import TimeSeriesPlot
from src.visualization.plots.xy_plot import XYPlot
from src.visualization.plots.fft import FFTPlot
from src.visualization.maps.map_2d import Map2D
from src.visualization.maps.map_3d import Map3D
from src.visualization.maps.markers import Marker, MarkerType, MarkerManager
from src.visualization.maps.path_renderer import PathRenderer
from src.visualization.maps.layers import Layer, LayerType, LayerManager
from src.visualization.manager import PlotManager
from src.visualization.theme import Theme


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 10, n)
    return pd.DataFrame({
        'timestamp': t,
        'signal1': np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, n),
        'signal2': np.cos(2 * np.pi * t) + np.random.normal(0, 0.1, n),
        'signal3': t + np.random.normal(0, 0.5, n),
    })


@pytest.fixture
def gps_data():
    """Create sample GPS data."""
    n = 100
    return pd.DataFrame({
        'timestamp': np.linspace(0, 100, n),
        'lat': np.linspace(47.0, 47.1, n) + np.random.normal(0, 0.001, n),
        'lon': np.linspace(-122.0, -121.9, n) + np.random.normal(0, 0.001, n),
        'alt': np.linspace(100, 200, n) + np.random.normal(0, 5, n),
    })


class TestHistogramPlot:
    """Test HistogramPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {'title': 'Test Histogram'}
        plot = HistogramPlot(config)
        assert plot.n_bins == 50
        assert plot.normalize is True
        assert plot.show_kde is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = {'n_bins': 30, 'normalize': False, 'show_kde': False}
        plot = HistogramPlot(config)
        assert plot.n_bins == 30
        assert plot.normalize is False
        assert plot.show_kde is False
    
    def test_render(self, sample_data):
        """Test render method."""
        config = {'signals': ['signal1']}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    
    def test_render_multiple_signals(self, sample_data):
        """Test render with multiple signals."""
        config = {'signals': ['signal1', 'signal2']}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        assert len(fig.data) >= 2
    
    def test_render_no_signals(self, sample_data):
        """Test render with no signals specified."""
        config = {}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_with_kde(self, sample_data):
        """Test render with KDE overlay."""
        config = {'signals': ['signal1'], 'show_kde': True}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        # Should have histogram + KDE trace
        assert len(fig.data) >= 2
    
    def test_render_without_kde(self, sample_data):
        """Test render without KDE overlay."""
        config = {'signals': ['signal1'], 'show_kde': False}
        plot = HistogramPlot(config)
        fig = plot.render(sample_data)
        # Should only have histogram trace
        assert len(fig.data) == 1
    
    def test_add_normal_fit(self, sample_data):
        """Test adding normal fit."""
        config = {'signals': ['signal1']}
        plot = HistogramPlot(config)
        plot.render(sample_data)
        plot.add_normal_fit('signal1')
        # Should have additional trace for normal fit
    
    def test_add_normal_fit_no_figure(self):
        """Test adding normal fit with no figure."""
        config = {}
        plot = HistogramPlot(config)
        plot.add_normal_fit('signal1')  # Should not raise
    
    def test_add_statistics_annotation(self, sample_data):
        """Test adding statistics annotation."""
        config = {'signals': ['signal1']}
        plot = HistogramPlot(config)
        plot.render(sample_data)
        plot.add_statistics_annotation('signal1')
    
    def test_add_statistics_annotation_no_figure(self):
        """Test adding annotation with no figure."""
        config = {}
        plot = HistogramPlot(config)
        plot.add_statistics_annotation('signal1')  # Should not raise
    
    def test_update(self, sample_data):
        """Test update method."""
        config = {'signals': ['signal1']}
        plot = HistogramPlot(config)
        plot.render(sample_data)
        plot.update(sample_data)
    
    def test_get_signal_name(self):
        """Test _get_signal_name method."""
        config = {}
        plot = HistogramPlot(config)
        assert plot._get_signal_name('path.to.signal') == 'signal'
        assert plot._get_signal_name('signal') == 'signal'


class TestSpectrogramPlot:
    """Test SpectrogramPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        plot = SpectrogramPlot(config)
        assert plot.overlap == 0.75
        assert plot.window_type == 'hann'
        assert plot.show_db is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = {'window_size': 512, 'overlap': 0.5, 'show_db': False}
        plot = SpectrogramPlot(config)
        assert plot.window_size == 512
        assert plot.overlap == 0.5
        assert plot.show_db is False
    
    def test_render(self, sample_data):
        """Test render method."""
        config = {'signals': ['signal1']}
        plot = SpectrogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_no_signals(self, sample_data):
        """Test render with no signals specified."""
        config = {}
        plot = SpectrogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_db_scale(self, sample_data):
        """Test render with dB scale."""
        config = {'signals': ['signal1'], 'show_db': True}
        plot = SpectrogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_linear_scale(self, sample_data):
        """Test render with linear scale."""
        config = {'signals': ['signal1'], 'show_db': False}
        plot = SpectrogramPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_update(self, sample_data):
        """Test update method."""
        config = {'signals': ['signal1']}
        plot = SpectrogramPlot(config)
        plot.render(sample_data)
        plot.update(sample_data)
    
    def test_set_frequency_range(self, sample_data):
        """Test set_frequency_range method."""
        config = {'signals': ['signal1']}
        plot = SpectrogramPlot(config)
        plot.render(sample_data)
        plot.set_frequency_range(0, 50)
    
    def test_set_frequency_range_no_figure(self):
        """Test set_frequency_range with no figure."""
        config = {}
        plot = SpectrogramPlot(config)
        plot.set_frequency_range(0, 50)  # Should not raise


class TestStatisticsPlot:
    """Test StatisticsPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        plot = StatisticsPlot(config)
        assert plot.stat_type == 'boxplot'
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = {'stat_type': 'violin'}
        plot = StatisticsPlot(config)
        assert plot.stat_type == 'violin'
    
    def test_render_boxplot(self, sample_data):
        """Test rendering box plot."""
        config = {'stat_type': 'boxplot', 'signals': ['signal1', 'signal2']}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_violin(self, sample_data):
        """Test rendering violin plot."""
        config = {'stat_type': 'violin', 'signals': ['signal1', 'signal2']}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_correlation(self, sample_data):
        """Test rendering correlation matrix."""
        config = {'stat_type': 'correlation', 'signals': ['signal1', 'signal2', 'signal3']}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_unknown_type(self, sample_data):
        """Test rendering unknown type defaults to boxplot."""
        config = {'stat_type': 'unknown'}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_no_signals(self, sample_data):
        """Test render with no signals specified."""
        config = {'stat_type': 'boxplot'}
        plot = StatisticsPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_get_statistics_summary(self, sample_data):
        """Test getting statistics summary."""
        config = {'signals': ['signal1']}
        plot = StatisticsPlot(config)
        plot.render(sample_data)
        summary = plot.get_statistics_summary()
        assert 'signal1' in summary
        assert 'mean' in summary['signal1']
        assert 'std' in summary['signal1']
    
    def test_get_statistics_summary_no_data(self):
        """Test getting summary with no data."""
        config = {}
        plot = StatisticsPlot(config)
        summary = plot.get_statistics_summary()
        assert summary == {}
    
    def test_update(self, sample_data):
        """Test update method."""
        config = {'stat_type': 'boxplot'}
        plot = StatisticsPlot(config)
        plot.render(sample_data)
        plot.update(sample_data)


class TestTimeSeriesPlot:
    """Test TimeSeriesPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {'x_axis': 'timestamp'}
        plot = TimeSeriesPlot(config)
        assert plot is not None
    
    def test_render(self, sample_data):
        """Test render method."""
        config = {'x_axis': 'timestamp', 'signals': ['signal1']}
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_multiple_signals(self, sample_data):
        """Test render with multiple signals."""
        config = {'x_axis': 'timestamp', 'signals': ['signal1', 'signal2']}
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_data)
        assert len(fig.data) == 2


class TestXYPlot:
    """Test XYPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {'x_axis': 'signal1', 'y_axis': 'signal2'}
        plot = XYPlot(config)
        assert plot is not None
    
    def test_render(self, sample_data):
        """Test render method."""
        config = {'x_axis': 'signal1', 'y_axis': 'signal2'}
        plot = XYPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)


class TestFFTPlot:
    """Test FFTPlot class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        plot = FFTPlot(config)
        assert plot is not None
    
    def test_render(self, sample_data):
        """Test render method."""
        config = {'signals': ['signal1']}
        plot = FFTPlot(config)
        fig = plot.render(sample_data)
        assert isinstance(fig, go.Figure)


class TestMap2D:
    """Test Map2D class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        map_plot = Map2D(config)
        assert map_plot is not None
    
    def test_render(self, gps_data):
        """Test render method."""
        config = {'lat_signal': 'lat', 'lon_signal': 'lon'}
        map_plot = Map2D(config)
        fig = map_plot.render(gps_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_with_path(self, gps_data):
        """Test rendering with flight path in data."""
        config = {'lat_signal': 'lat', 'lon_signal': 'lon'}
        map_plot = Map2D(config)
        fig = map_plot.render(gps_data)
        # Verify figure has traces
        assert len(fig.data) > 0


class TestMap3D:
    """Test Map3D class."""
    
    def test_init(self):
        """Test initialization."""
        config = {}
        map_plot = Map3D(config)
        assert map_plot is not None
    
    def test_render(self, gps_data):
        """Test render method."""
        config = {
            'lat_signal': 'lat',
            'lon_signal': 'lon',
            'alt_signal': 'alt'
        }
        map_plot = Map3D(config)
        fig = map_plot.render(gps_data)
        assert isinstance(fig, go.Figure)


class TestMapMarkers:
    """Test Marker and MarkerManager classes."""
    
    def test_create_marker(self):
        """Test creating a marker."""
        marker = Marker(lat=47.0, lon=-122.0, marker_type=MarkerType.CUSTOM, label='Test')
        assert marker is not None
        assert marker.lat == 47.0
        assert marker.lon == -122.0
    
    def test_marker_type_enum(self):
        """Test MarkerType enum."""
        assert MarkerType.START.value == "start"
        assert MarkerType.END.value == "end"
        assert MarkerType.EVENT.value == "event"
    
    def test_marker_manager_add(self):
        """Test MarkerManager adding markers."""
        manager = MarkerManager()
        marker = Marker(lat=47.0, lon=-122.0, marker_type=MarkerType.START)
        manager.add_marker(marker)
        assert len(manager.get_markers()) == 1
    
    def test_marker_manager_remove(self):
        """Test MarkerManager removing markers."""
        manager = MarkerManager()
        marker = Marker(lat=47.0, lon=-122.0, marker_type=MarkerType.START)
        manager.add_marker(marker)
        manager.remove_marker(0)
        assert len(manager.get_markers()) == 0
    
    def test_marker_manager_clear(self):
        """Test MarkerManager clearing markers."""
        manager = MarkerManager()
        manager.add_marker(Marker(lat=0, lon=0, marker_type=MarkerType.START))
        manager.add_marker(Marker(lat=1, lon=1, marker_type=MarkerType.END))
        manager.clear()
        assert len(manager.get_markers()) == 0
    
    def test_marker_manager_filter(self):
        """Test MarkerManager filtering by type."""
        manager = MarkerManager()
        manager.add_marker(Marker(lat=0, lon=0, marker_type=MarkerType.START))
        manager.add_marker(Marker(lat=1, lon=1, marker_type=MarkerType.EVENT))
        starts = manager.get_markers(MarkerType.START)
        assert len(starts) == 1
    
    def test_marker_default_metadata(self):
        """Test marker default metadata."""
        marker = Marker(lat=0, lon=0, marker_type=MarkerType.CUSTOM)
        assert marker.metadata == {}
    
    def test_marker_to_plotly_traces(self):
        """Test converting markers to Plotly traces."""
        manager = MarkerManager()
        manager.add_marker(Marker(lat=47.0, lon=-122.0, marker_type=MarkerType.START, label='Home'))
        traces = manager.to_plotly_traces()
        assert len(traces) == 1
        assert traces[0]['lat'] == [47.0]


class TestPathRenderer:
    """Test PathRenderer class."""
    
    def test_init(self):
        """Test initialization."""
        renderer = PathRenderer()
        assert renderer is not None
    
    def test_create_time_gradient(self, gps_data):
        """Test creating time gradient path."""
        renderer = PathRenderer()
        segments = renderer.create_time_gradient(
            gps_data['lat'].values,
            gps_data['lon'].values
        )
        assert len(segments) > 0
    
    def test_segment_by_value(self, gps_data):
        """Test segmenting path by value."""
        renderer = PathRenderer()
        segments = renderer.segment_by_value(
            gps_data['lat'].values,
            gps_data['lon'].values,
            gps_data['alt'].values
        )
        assert len(segments) > 0


class TestMapLayer:
    """Test Layer class."""
    
    def test_create_layer(self):
        """Test creating a layer."""
        layer = Layer(id='test_layer', layer_type=LayerType.CUSTOM)
        assert layer.id == 'test_layer'
        assert layer.layer_type == LayerType.CUSTOM
    
    def test_layer_visibility(self):
        """Test layer visibility."""
        layer = Layer(id='test', layer_type=LayerType.WAYPOINTS)
        assert layer.visible is True
        layer.visible = False
        assert layer.visible is False
    
    def test_layer_type_enum(self):
        """Test LayerType enum."""
        assert LayerType.GEOFENCE.value == "geofence"
        assert LayerType.WAYPOINTS.value == "waypoints"
        assert LayerType.HEATMAP.value == "heatmap"


class TestMapLayerManager:
    """Test LayerManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = LayerManager()
        assert manager is not None
    
    def test_add_layer(self):
        """Test adding layer."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.CUSTOM)
        manager.add_layer(layer)
        assert manager.get_layer('test') is not None
    
    def test_remove_layer(self):
        """Test removing layer."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.CUSTOM)
        manager.add_layer(layer)
        result = manager.remove_layer('test')
        assert result is True
        assert manager.get_layer('test') is None
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent layer."""
        manager = LayerManager()
        result = manager.remove_layer('nonexistent')
        assert result is False
    
    def test_set_visibility(self):
        """Test setting visibility."""
        manager = LayerManager()
        layer = Layer(id='test', layer_type=LayerType.GRID)
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


class TestPlotManager:
    """Test PlotManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = PlotManager()
        assert manager is not None
    
    def test_create_plot(self):
        """Test creating plot."""
        manager = PlotManager()
        config = {'plot_type': 'TIME_SERIES', 'id': 'test_plot'}
        plot = manager.create_plot(config)
        assert plot is not None
    
    def test_get_plot(self):
        """Test getting plot."""
        manager = PlotManager()
        config = {'plot_type': 'TIME_SERIES', 'id': 'plot1'}
        manager.create_plot(config)
        plot = manager.get_plot('plot1')
        assert plot is not None
    
    def test_remove_plot(self):
        """Test removing plot."""
        manager = PlotManager()
        config = {'plot_type': 'TIME_SERIES', 'id': 'plot1'}
        manager.create_plot(config)
        result = manager.remove_plot('plot1')
        assert result is True
        assert manager.get_plot('plot1') is None
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent plot."""
        manager = PlotManager()
        result = manager.remove_plot('nonexistent')
        assert result is False


class TestTheme:
    """Test Theme class."""
    
    def test_default_theme(self):
        """Test default theme."""
        theme = Theme()
        assert theme.name == 'default'
    
    def test_custom_theme(self):
        """Test custom theme."""
        theme = Theme(name='custom')
        assert theme.name == 'custom'
    
    def test_signal_colors(self):
        """Test signal color palette."""
        theme = Theme()
        assert isinstance(theme.signal_colors, list)
        assert len(theme.signal_colors) > 0
    
    def test_to_plotly_template(self):
        """Test getting Plotly template."""
        theme = Theme()
        template = theme.to_plotly_template()
        assert isinstance(template, dict)
    
    def test_event_colors(self):
        """Test event colors by severity."""
        theme = Theme()
        assert 'critical' in theme.event_colors
        assert 'warning' in theme.event_colors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

