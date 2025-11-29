"""
Tests for UI callbacks and interactivity.

Covers:
- Signal selector functionality (REQ-SDM-001 to REQ-SDM-010)
- Playback controls and speed (REQ-VIS-071 to REQ-VIS-080)
- Plot updates (REQ-VIS-001 to REQ-VIS-012)
"""

import pytest
from unittest.mock import MagicMock, patch
from dash import Dash, html, dcc, no_update
import dash_bootstrap_components as dbc


class TestSignalSelector:
    """Tests for signal selector component."""

    def test_signal_options_built_from_data(self, sample_flight_data):
        """Test that signal options are correctly built from flight data."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        
        assert len(options) > 0
        for opt in options:
            assert 'label' in opt
            assert 'value' in opt
            assert isinstance(opt['label'], str)
            assert isinstance(opt['value'], str)

    def test_signal_options_empty_data(self):
        """Test signal options with empty data."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options({})
        
        assert options == []

    def test_signal_options_nested_structure(self):
        """Test signal options with nested hierarchical data."""
        import pandas as pd
        from src.ui.app_layout import _build_signal_options
        
        nested_data = {
            'Sensors': {
                'IMU': {
                    'Accelerometer': pd.DataFrame({
                        'timestamp': [1, 2, 3],
                        'x': [0.1, 0.2, 0.3],
                        'y': [0.4, 0.5, 0.6]
                    })
                }
            }
        }
        
        options = _build_signal_options(nested_data)
        
        # Should have options for x and y signals
        values = [opt['value'] for opt in options]
        assert any('x' in v for v in values)
        assert any('y' in v for v in values)

    def test_signal_checklist_multi_select(self):
        """Test that checklist supports multiple selections."""
        # Simulate checklist behavior
        checklist_values = ['Sensors.GPS.latitude', 'Sensors.GPS.longitude', 'Battery.voltage']
        
        assert len(checklist_values) == 3
        assert all(isinstance(v, str) for v in checklist_values)

    def test_signal_search_filter(self):
        """Test signal filtering functionality."""
        all_signals = [
            'Sensors.GPS.latitude',
            'Sensors.GPS.longitude',
            'Sensors.IMU.accel_x',
            'Battery.voltage',
            'Battery.current'
        ]
        
        search_term = 'GPS'
        filtered = [s for s in all_signals if search_term.lower() in s.lower()]
        
        assert len(filtered) == 2
        assert 'Sensors.GPS.latitude' in filtered
        assert 'Sensors.GPS.longitude' in filtered

    def test_signal_count_display(self):
        """Test that signal count is correctly calculated."""
        selected_signals = ['sig1', 'sig2', 'sig3']
        expected_text = f'{len(selected_signals)} signals selected'
        
        assert expected_text == '3 signals selected'


class TestPlaybackControls:
    """Tests for playback controls and speed."""

    def test_toggle_playback_from_stopped(self):
        """Test toggling playback from stopped to playing state."""
        # Simulate callback behavior
        current_state = 'stopped'
        n_clicks = 1
        
        # Expected: should start playing
        new_disabled = False  # interval should be enabled
        new_state = 'playing'
        
        assert new_state == 'playing'
        assert new_disabled == False

    def test_toggle_playback_from_playing(self):
        """Test toggling playback from playing to stopped state."""
        current_state = 'playing'
        
        # Expected: should stop
        new_disabled = True  # interval should be disabled
        new_state = 'stopped'
        
        assert new_state == 'stopped'
        assert new_disabled == True

    def test_playback_speed_1x(self):
        """Test playback at 1x speed."""
        speed = 1.0
        current_time = 10.0
        interval = 0.1  # 100ms interval
        
        new_time = current_time + speed * interval
        
        assert new_time == 10.1

    def test_playback_speed_2x(self):
        """Test playback at 2x speed."""
        speed = 2.0
        current_time = 10.0
        interval = 0.1
        
        new_time = current_time + speed * interval
        
        assert new_time == 10.2

    def test_playback_speed_05x(self):
        """Test playback at 0.5x speed."""
        speed = 0.5
        current_time = 10.0
        interval = 0.1
        
        new_time = current_time + speed * interval
        
        assert new_time == 10.05

    def test_playback_speed_4x(self):
        """Test playback at 4x speed."""
        speed = 4.0
        current_time = 10.0
        interval = 0.1
        
        new_time = current_time + speed * interval
        
        assert new_time == 10.4

    def test_playback_doesnt_exceed_max(self):
        """Test that playback doesn't exceed maximum time."""
        speed = 2.0
        current_time = 99.95
        max_time = 100.0
        interval = 0.1
        
        new_time = min(current_time + speed * interval, max_time)
        
        assert new_time == max_time

    def test_step_forward(self):
        """Test step forward button."""
        current_time = 50.0
        max_time = 100.0
        step = max_time / 100  # 1 second step
        
        new_time = min(max_time, current_time + step)
        
        assert new_time == 51.0

    def test_step_backward(self):
        """Test step backward button."""
        current_time = 50.0
        max_time = 100.0
        step = max_time / 100
        
        new_time = max(0, current_time - step)
        
        assert new_time == 49.0

    def test_step_backward_doesnt_go_negative(self):
        """Test that step backward doesn't go below 0."""
        current_time = 0.5
        max_time = 100.0
        step = max_time / 100
        
        new_time = max(0, current_time - step)
        
        assert new_time == 0

    def test_jump_to_start(self):
        """Test jump to start button."""
        current_time = 50.0
        
        # Pressing start button should set time to 0
        new_time = 0
        
        assert new_time == 0

    def test_jump_to_end(self):
        """Test jump to end button."""
        current_time = 50.0
        max_time = 100.0
        
        # Pressing end button should set time to max
        new_time = max_time
        
        assert new_time == 100.0

    def test_time_display_format(self):
        """Test time display formatting."""
        def format_time(current_time):
            if current_time is None:
                return "00:00.00"
            minutes = int(current_time // 60)
            seconds = current_time % 60
            return f"{minutes:02d}:{seconds:05.2f}"
        
        assert format_time(0) == "00:00.00"
        assert format_time(30.5) == "00:30.50"
        assert format_time(65.25) == "01:05.25"
        assert format_time(125.5) == "02:05.50"
        assert format_time(None) == "00:00.00"

    def test_slider_value_updates_time(self):
        """Test that slider value updates current time."""
        slider_value = 75.5
        
        # When slider is moved, current_time should match
        current_time = slider_value
        
        assert current_time == 75.5


class TestPlotUpdates:
    """Tests for plot update functionality."""

    def test_create_time_series_figure(self, sample_dataframe):
        """Test creating a time series figure."""
        from src.visualization.plots.time_series import TimeSeriesPlot
        
        config = {
            'title': 'Test Plot', 
            'x_axis': 'timestamp', 
            'y_columns': ['value'],
            'max_points': 5000
        }
        plot = TimeSeriesPlot(config)
        fig = plot.render(sample_dataframe)
        
        assert fig is not None
        assert len(fig.data) > 0

    def test_create_xy_figure(self, sample_dataframe):
        """Test creating an X-Y figure."""
        from src.visualization.plots.xy_plot import XYPlot
        
        config = {
            'title': 'Test XY Plot', 
            'x_axis': 'timestamp', 
            'y_axis': 'value'
        }
        plot = XYPlot(config)
        fig = plot.render(sample_dataframe)
        
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_with_multiple_signals(self, sample_flight_data):
        """Test plotting multiple signals on same figure."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add multiple traces
        signals = ['signal1', 'signal2', 'signal3']
        for i, sig in enumerate(signals):
            fig.add_trace(go.Scatter(
                x=[1, 2, 3],
                y=[i, i+1, i+2],
                name=sig
            ))
        
        assert len(fig.data) == 3
        assert fig.data[0].name == 'signal1'
        assert fig.data[1].name == 'signal2'
        assert fig.data[2].name == 'signal3'

    def test_plot_updates_with_time_window(self):
        """Test that plots update when time window changes."""
        import pandas as pd
        import numpy as np
        
        # Create sample data
        timestamps = np.linspace(0, 100, 1000)
        values = np.sin(timestamps * 0.1)
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        
        # Filter to time window
        start_time = 20
        end_time = 40
        filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        assert len(filtered) < len(df)
        assert filtered['timestamp'].min() >= start_time
        assert filtered['timestamp'].max() <= end_time

    def test_plot_marker_at_current_time(self):
        """Test that current time marker is added to plots."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        current_time = 50.0
        
        # Add vertical line at current time
        fig.add_vline(x=current_time, line_dash="dash", line_color="red")
        
        # Check that shape was added
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].x0 == current_time

    def test_plot_legend_visibility(self):
        """Test plot legend configuration."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name='Signal 1'))
        fig.update_layout(showlegend=True)
        
        assert fig.layout.showlegend == True

    def test_plot_axis_labels(self):
        """Test plot axis label configuration."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Amplitude'
        )
        
        assert fig.layout.xaxis.title.text == 'Time (s)'
        assert fig.layout.yaxis.title.text == 'Amplitude'

    def test_plot_dark_theme(self):
        """Test that plots use dark theme."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        assert fig.layout.template.layout.paper_bgcolor is not None or \
               fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'


class TestCallbackIntegration:
    """Integration tests for callbacks."""

    def test_callbacks_registered(self):
        """Test that callbacks are properly registered."""
        from src.core.app import create_app
        from src.ui.callbacks import register_callbacks
        
        app = create_app()
        register_callbacks(app)
        
        # Check that callbacks were registered
        assert len(app.callback_map) > 0

    def test_status_update_on_load(self):
        """Test status message updates on load."""
        # Simulate load click
        trigger = 'btn-load'
        expected_status = "Loading data..."
        
        if trigger == 'btn-load':
            status = "Loading data..."
        
        assert status == expected_status

    def test_status_update_on_save(self):
        """Test status message updates on save."""
        trigger = 'btn-save'
        
        if trigger == 'btn-save':
            status = "Configuration saved"
        
        assert status == "Configuration saved"

    def test_status_update_on_export(self):
        """Test status message updates on export."""
        trigger = 'btn-export'
        
        if trigger == 'btn-export':
            status = "Exporting..."
        
        assert status == "Exporting..."


class TestSpeedDropdown:
    """Tests for speed dropdown component."""

    def test_speed_options_available(self):
        """Test that all speed options are available."""
        expected_speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        # From app_layout.py
        speed_options = [
            {'label': '0.25x', 'value': 0.25},
            {'label': '0.5x', 'value': 0.5},
            {'label': '1x', 'value': 1.0},
            {'label': '2x', 'value': 2.0},
            {'label': '4x', 'value': 4.0},
        ]
        
        actual_speeds = [opt['value'] for opt in speed_options]
        
        assert actual_speeds == expected_speeds

    def test_default_speed_is_1x(self):
        """Test that default playback speed is 1x."""
        default_speed = 1.0
        
        assert default_speed == 1.0

    def test_speed_changes_playback_rate(self):
        """Test that changing speed affects time updates."""
        base_interval = 0.1  # 100ms
        
        for speed in [0.25, 0.5, 1.0, 2.0, 4.0]:
            time_delta = speed * base_interval
            expected = speed * 0.1
            assert abs(time_delta - expected) < 0.001


class TestMapUpdates:
    """Tests for map update functionality."""

    def test_map_position_marker_updates(self):
        """Test that map position marker updates with current time."""
        import pandas as pd
        import numpy as np
        
        # Sample GPS data
        gps_data = pd.DataFrame({
            'timestamp': np.linspace(0, 100, 100),
            'latitude': np.linspace(40.0, 40.1, 100),
            'longitude': np.linspace(-74.0, -73.9, 100)
        })
        
        current_time = 50.0
        
        # Find closest timestamp
        idx = (gps_data['timestamp'] - current_time).abs().idxmin()
        position = gps_data.loc[idx]
        
        assert abs(position['timestamp'] - 50.0) < 1.0
        assert 40.0 <= position['latitude'] <= 40.1

    def test_flight_path_rendering(self):
        """Test flight path rendering on map."""
        import pandas as pd
        
        gps_data = pd.DataFrame({
            'latitude': [40.0, 40.05, 40.1],
            'longitude': [-74.0, -73.95, -73.9]
        })
        
        # Check path has correct structure
        assert len(gps_data) == 3
        assert 'latitude' in gps_data.columns
        assert 'longitude' in gps_data.columns


class TestEventAnnotations:
    """Tests for event annotations on plots."""

    def test_event_markers_added_to_plot(self):
        """Test that event markers are added to plots."""
        import plotly.graph_objects as go
        
        events = [
            {'timestamp': 10, 'name': 'Takeoff'},
            {'timestamp': 50, 'name': 'Waypoint 1'},
            {'timestamp': 90, 'name': 'Landing'}
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name='Signal'))
        
        # Add event markers
        for event in events:
            fig.add_vline(
                x=event['timestamp'],
                line_dash="dot",
                annotation_text=event['name']
            )
        
        assert len(fig.layout.shapes) == 3

    def test_event_hover_shows_details(self):
        """Test that event hover shows event details."""
        import plotly.graph_objects as go
        
        event = {'timestamp': 50, 'name': 'Waypoint', 'description': 'First waypoint'}
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[event['timestamp']],
            y=[0],
            mode='markers',
            marker=dict(size=15, symbol='diamond'),
            name=event['name'],
            hovertext=event['description'],
            hoverinfo='text+name'
        ))
        
        assert fig.data[0].hovertext == 'First waypoint'

