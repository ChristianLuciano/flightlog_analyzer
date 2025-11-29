"""Integration tests for visualization."""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.visualization.plots.time_series import TimeSeriesPlot
from src.visualization.plots.fft import FFTPlot
from src.visualization.maps.map_2d import Map2D
from src.core.types import PlotConfig


class TestTimeSeriesPlot:
    """Integration tests for time series plots."""

    def test_render_basic(self, sample_dataframe):
        """Test basic rendering."""
        config: PlotConfig = {
            'id': 'test-plot',
            'plot_type': 'TIME_SERIES',
            'signals': ['sin_wave', 'cos_wave'],
            'title': 'Test Plot',
        }

        plot = TimeSeriesPlot(config)
        figure = plot.render(sample_dataframe)

        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 2

    def test_downsampling(self):
        """Test that large datasets are downsampled."""
        # Large dataset
        n = 100000
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 100, n),
            'signal': np.sin(np.linspace(0, 20 * np.pi, n))
        })

        config: PlotConfig = {
            'id': 'test',
            'plot_type': 'TIME_SERIES',
            'signals': ['signal'],
            'max_points': 1000,
        }

        plot = TimeSeriesPlot(config)
        figure = plot.render(df)

        # Should have fewer points than original
        assert len(figure.data[0].x) <= 1000


class TestFFTPlot:
    """Integration tests for FFT plots."""

    def test_render_fft(self, sample_dataframe):
        """Test FFT plot rendering."""
        config: PlotConfig = {
            'id': 'fft-test',
            'plot_type': 'FFT',
            'signals': ['sin_wave'],
        }

        plot = FFTPlot(config)
        figure = plot.render(sample_dataframe)

        assert isinstance(figure, go.Figure)
        assert len(figure.data) >= 1


class TestMap2D:
    """Integration tests for 2D maps."""

    def test_render_map(self, sample_gps_data):
        """Test map rendering."""
        config: PlotConfig = {
            'id': 'map-test',
            'plot_type': 'MAP_2D',
            'lat_column': 'lat',
            'lon_column': 'lon',
            'title': 'Test Map',
        }

        map_plot = Map2D(config)
        figure = map_plot.render(sample_gps_data)

        assert isinstance(figure, go.Figure)
        # Should have path and markers
        assert len(figure.data) >= 3

    def test_color_by_signal(self, sample_gps_data):
        """Test coloring path by signal."""
        config: PlotConfig = {
            'id': 'map-test',
            'plot_type': 'MAP_2D',
            'lat_column': 'lat',
            'lon_column': 'lon',
            'color_by': 'altitude',
        }

        map_plot = Map2D(config)
        figure = map_plot.render(sample_gps_data)

        assert isinstance(figure, go.Figure)

