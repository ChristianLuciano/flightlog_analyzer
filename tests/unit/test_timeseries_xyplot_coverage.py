"""
Tests to improve coverage for time series and XY plot modules.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.plots.time_series import TimeSeriesPlot
from src.visualization.plots.xy_plot import XYPlot
from src.visualization.plots.fft import FFTPlot


@pytest.fixture
def sample_ts_data():
    """Create sample time series data."""
    n = 500
    t = np.linspace(0, 50, n)
    return pd.DataFrame({
        'timestamp': t,
        'signal_a': np.sin(t * 0.2) + np.random.randn(n) * 0.1,
        'signal_b': np.cos(t * 0.2) + np.random.randn(n) * 0.1,
        'signal_c': t * 0.1 + np.random.randn(n) * 0.5,
    })


class TestTimeSeriesPlotInit:
    """Test TimeSeriesPlot initialization."""
    
    def test_default_config(self):
        """Test default configuration."""
        plot = TimeSeriesPlot({})
        assert plot.timestamp_column == 'timestamp'
        assert plot.max_display_points == 5000
    
    def test_custom_config(self):
        """Test custom configuration."""
        plot = TimeSeriesPlot({
            'x_axis': 'time',
            'max_points': 2000
        })
        assert plot.timestamp_column == 'time'
        assert plot.max_display_points == 2000


class TestTimeSeriesPlotRender:
    """Test TimeSeriesPlot render method."""
    
    def test_render_single_signal(self, sample_ts_data):
        """Test rendering single signal."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal_a']
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
    
    def test_render_multiple_signals(self, sample_ts_data):
        """Test rendering multiple signals."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal_a', 'signal_b', 'signal_c']
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        assert len(fig.data) == 3
    
    def test_render_auto_signals(self, sample_ts_data):
        """Test auto-detecting signals."""
        config = {'x_axis': 'timestamp'}
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        # Should detect signal_a, signal_b, signal_c
        assert len(fig.data) >= 3
    
    def test_render_with_downsampling(self):
        """Test rendering with downsampling."""
        n = 10000
        data = pd.DataFrame({
            'timestamp': np.linspace(0, 100, n),
            'value': np.random.randn(n)
        })
        config = {
            'x_axis': 'timestamp',
            'signals': ['value'],
            'max_points': 1000
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(data)
        # Should downsample to ~1000 points
        assert len(fig.data[0].x) <= 1000
    
    def test_render_missing_signal(self, sample_ts_data):
        """Test handling missing signal."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['nonexistent']
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        # Should skip missing signal
        assert len(fig.data) == 0
    
    def test_render_with_line_width(self, sample_ts_data):
        """Test custom line width."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal_a'],
            'line_width': 3
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        assert fig.data[0].line.width == 3
    
    def test_render_with_title(self, sample_ts_data):
        """Test plot with title."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal_a'],
            'title': 'My Plot'
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_ts_data)
        assert 'My Plot' in str(fig.layout.title)


class TestTimeSeriesPlotUpdate:
    """Test TimeSeriesPlot update method."""
    
    def test_update(self, sample_ts_data):
        """Test updating plot."""
        config = {'x_axis': 'timestamp', 'signals': ['signal_a']}
        plot = TimeSeriesPlot(config)
        plot.render(sample_ts_data)
        
        # Update with new data
        new_data = sample_ts_data.copy()
        new_data['signal_a'] = new_data['signal_a'] * 2
        plot.update(new_data)
        assert plot.figure is not None


class TestXYPlotInit:
    """Test XYPlot initialization."""
    
    def test_default_config(self):
        """Test default configuration."""
        plot = XYPlot({})
        assert plot.max_display_points == 10000
    
    def test_custom_config(self):
        """Test custom configuration."""
        plot = XYPlot({'max_points': 5000})
        assert plot.max_display_points == 5000


class TestXYPlotRender:
    """Test XYPlot render method."""
    
    def test_render_scatter(self, sample_ts_data):
        """Test scatter plot."""
        config = {
            'x_axis': 'signal_a',
            'y_axis': 'signal_b',
            'plot_type': 'XY_SCATTER'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_ts_data)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].mode == 'markers'
    
    def test_render_line(self, sample_ts_data):
        """Test line plot."""
        config = {
            'x_axis': 'signal_a',
            'y_axis': 'signal_b',
            'plot_type': 'XY_LINE'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_ts_data)
        assert fig.data[0].mode == 'lines'
    
    def test_render_with_color(self, sample_ts_data):
        """Test with color mapping."""
        config = {
            'x_axis': 'signal_a',
            'y_axis': 'signal_b',
            'color_by': 'signal_c'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_ts_data)
        assert len(fig.data) > 0
    
    def test_render_missing_x(self, sample_ts_data):
        """Test missing X column."""
        config = {
            'x_axis': 'nonexistent',
            'y_axis': 'signal_b'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_ts_data)
        assert len(fig.data) == 0
    
    def test_render_missing_y(self, sample_ts_data):
        """Test missing Y column."""
        config = {
            'x_axis': 'signal_a',
            'y_axis': 'nonexistent'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_ts_data)
        assert len(fig.data) == 0
    
    def test_render_with_downsampling(self):
        """Test XY plot with downsampling."""
        n = 20000
        data = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n)
        })
        config = {
            'x_axis': 'x',
            'y_axis': 'y',
            'max_points': 5000
        }
        plot = XYPlot(config)
        fig = plot.render(data)
        assert len(fig.data[0].x) <= 5000


class TestFFTPlotInit:
    """Test FFTPlot initialization."""
    
    def test_default_config(self):
        """Test default configuration."""
        plot = FFTPlot({})
        assert plot is not None


class TestFFTPlotRender:
    """Test FFTPlot render method."""
    
    def test_render_basic(self, sample_ts_data):
        """Test basic FFT rendering."""
        config = {
            'x_axis': 'timestamp',
            'signals': ['signal_a']
        }
        plot = FFTPlot(config)
        fig = plot.render(sample_ts_data)
        assert isinstance(fig, go.Figure)
    
    def test_render_no_timestamp(self, sample_ts_data):
        """Test FFT with no timestamp specified."""
        config = {'signals': ['signal_a']}
        plot = FFTPlot(config)
        fig = plot.render(sample_ts_data)
        # Should still work


class TestPlotHelpers:
    """Test plot helper methods."""
    
    def test_get_color_sequence(self):
        """Test getting color sequence."""
        plot = TimeSeriesPlot({})
        colors = plot._get_color_sequence()
        assert len(colors) > 0
    
    def test_get_signal_name(self):
        """Test getting display name for signal."""
        plot = TimeSeriesPlot({})
        name = plot._get_signal_name('Sensors.IMU.accel_x')
        # Should return last part or full path
        assert 'accel_x' in name or 'Sensors' in name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

