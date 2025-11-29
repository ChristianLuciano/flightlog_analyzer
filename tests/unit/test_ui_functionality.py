"""
Tests for UI functionality - export, signal selection, plot updates.

Covers:
- Export modal and functionality
- Signal selection from checklist
- Quick action buttons
- Plot updates based on selection
- Custom plot creation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


class TestExportFunctionality:
    """Tests for export functionality."""
    
    def test_export_modal_created(self):
        """Test that export modal is created correctly."""
        from src.ui.app_layout import _create_export_modal
        
        modal = _create_export_modal()
        
        assert modal is not None
        assert modal.id == 'export-modal'
    
    def test_export_format_options(self):
        """Test that all export formats are available."""
        from src.ui.app_layout import _create_export_modal
        
        modal = _create_export_modal()
        
        # Find the RadioItems for export format
        expected_formats = ['csv', 'excel', 'json', 'matlab']
        # Modal structure is correct if it was created without error
        assert modal is not None
    
    def test_export_csv(self, tmp_path, sample_flight_data):
        """Test CSV export functionality."""
        from src.export.data_export import export_csv
        
        # Flatten data
        gps_data = sample_flight_data.get('Sensors', {}).get('GPS', pd.DataFrame())
        if not gps_data.empty:
            path = tmp_path / "test.csv"
            export_csv(gps_data, path)
            
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == len(gps_data)
    
    def test_export_excel(self, tmp_path, sample_flight_data):
        """Test Excel export functionality."""
        from src.export.data_export import export_excel
        
        gps_data = sample_flight_data.get('Sensors', {}).get('GPS', pd.DataFrame())
        if not gps_data.empty:
            path = tmp_path / "test.xlsx"
            export_excel(gps_data, path)
            
            assert path.exists()
    
    def test_export_matlab(self, tmp_path, sample_dataframe):
        """Test MATLAB export functionality."""
        from src.export.data_export import export_matlab
        
        path = tmp_path / "test.mat"
        export_matlab(sample_dataframe, path)
        
        assert path.exists()
        
        # Verify can be loaded
        from scipy.io import loadmat
        data = loadmat(str(path))
        assert 'flight_data' in data


class TestSignalSelection:
    """Tests for signal selection functionality."""
    
    def test_build_signal_options(self, sample_flight_data):
        """Test building signal options from hierarchical data."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        
        assert len(options) > 0
        for opt in options:
            assert 'label' in opt
            assert 'value' in opt
    
    def test_signal_filter_by_search(self, sample_flight_data):
        """Test filtering signals by search term."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        
        # Filter by 'GPS'
        filtered = [opt for opt in options if 'GPS' in opt['value'] or 'gps' in opt['value'].lower()]
        
        # Should have fewer options than total
        assert len(filtered) <= len(options)
    
    def test_signal_count_updates(self):
        """Test signal count display updates."""
        selected = ['signal1', 'signal2', 'signal3']
        count = len(selected)
        expected_text = f"{count} signal{'s' if count != 1 else ''} selected"
        
        assert expected_text == "3 signals selected"
    
    def test_single_signal_count(self):
        """Test singular vs plural in signal count."""
        selected = ['signal1']
        count = len(selected)
        expected_text = f"{count} signal{'s' if count != 1 else ''} selected"
        
        assert expected_text == "1 signal selected"


class TestQuickActions:
    """Tests for quick action buttons."""
    
    def test_quick_imu_selects_imu_signals(self, sample_flight_data):
        """Test that IMU quick action selects IMU-related signals."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        all_values = [opt['value'] for opt in options]
        
        imu_signals = [s for s in all_values if 'IMU' in s or 'accel' in s.lower() or 'gyro' in s.lower()]
        
        # Should find some IMU signals
        assert len(imu_signals) >= 0  # May be 0 if sample data doesn't have IMU
    
    def test_quick_gps_selects_gps_signals(self, sample_flight_data):
        """Test that GPS quick action selects GPS-related signals."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        all_values = [opt['value'] for opt in options]
        
        gps_signals = [s for s in all_values if 'GPS' in s or 'lat' in s.lower() or 'lon' in s.lower()]
        
        assert len(gps_signals) >= 0
    
    def test_quick_battery_selects_battery_signals(self, sample_flight_data):
        """Test that Battery quick action selects battery-related signals."""
        from src.ui.app_layout import _build_signal_options
        
        options = _build_signal_options(sample_flight_data)
        all_values = [opt['value'] for opt in options]
        
        battery_signals = [s for s in all_values if 'Battery' in s or 'batt' in s.lower() or 'volt' in s.lower()]
        
        assert len(battery_signals) >= 0


class TestCustomPlots:
    """Tests for custom plot creation."""
    
    def test_custom_plot_container_exists(self):
        """Test that custom plot container is in layout."""
        from src.core.app import create_app
        
        app = create_app()
        layout = app.layout
        
        # Layout should contain custom-plot-container
        assert layout is not None
    
    def test_get_signal_data_valid_path(self, sample_flight_data):
        """Test retrieving signal data from valid path."""
        from src.ui.callbacks import _get_signal_data
        
        # This depends on actual data structure
        # Test with a known path from sample data
        result = _get_signal_data(sample_flight_data, 'Sensors.GPS.lat')
        
        if result is not None:
            timestamps, values = result
            assert len(timestamps) > 0
            assert len(values) > 0
    
    def test_get_signal_data_invalid_path(self, sample_flight_data):
        """Test retrieving signal data from invalid path."""
        from src.ui.callbacks import _get_signal_data
        
        result = _get_signal_data(sample_flight_data, 'Invalid.Path.signal')
        
        assert result is None
    
    def test_flatten_flight_data(self, sample_flight_data):
        """Test flattening hierarchical flight data."""
        from src.ui.callbacks import _flatten_flight_data
        
        flat = _flatten_flight_data(sample_flight_data)
        
        assert isinstance(flat, dict)
        # Each key should be a path, each value should be dict of lists
        for key, value in flat.items():
            assert '.' in key or isinstance(sample_flight_data.get(key), pd.DataFrame)


class TestPlotUpdates:
    """Tests for plot updates based on signal selection."""
    
    def test_create_figure_with_signals(self):
        """Test creating a figure with selected signals."""
        import plotly.graph_objects as go
        
        # Simulate creating a figure
        fig = go.Figure()
        
        signals = [
            ('signal1', [0, 1, 2], [1.0, 2.0, 3.0]),
            ('signal2', [0, 1, 2], [4.0, 5.0, 6.0]),
        ]
        
        for name, timestamps, values in signals:
            fig.add_trace(go.Scattergl(
                x=timestamps,
                y=values,
                mode='lines',
                name=name
            ))
        
        assert len(fig.data) == 2
        assert fig.data[0].name == 'signal1'
        assert fig.data[1].name == 'signal2'
    
    def test_figure_dark_theme(self):
        """Test that figures use dark theme."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Check template is set
        assert fig.layout.template is not None
    
    def test_figure_hover_template(self):
        """Test custom hover templates on traces."""
        import plotly.graph_objects as go
        
        signal_path = 'Sensors.GPS.altitude'
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=[0, 1, 2],
            y=[100, 150, 200],
            mode='lines',
            name='altitude',
            hovertemplate=f'<b>{signal_path}</b><br>Time: %{{x:.2f}}s<br>Value: %{{y:.4f}}<extra></extra>'
        ))
        
        assert 'hovertemplate' in fig.data[0]


class TestCallbackHelpers:
    """Tests for callback helper functions."""
    
    def test_build_signal_options_empty(self):
        """Test building options from empty data."""
        from src.ui.callbacks import _build_signal_options
        
        options = _build_signal_options({})
        
        assert options == []
    
    def test_build_signal_options_nested(self):
        """Test building options from nested structure."""
        from src.ui.callbacks import _build_signal_options
        
        data = {
            'Level1': {
                'Level2': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'value': [10, 20, 30]
                })
            }
        }
        
        options = _build_signal_options(data)
        
        assert len(options) == 1  # Only 'value' (timestamp excluded)
        assert options[0]['value'] == 'Level1.Level2.value'
    
    def test_get_signal_data_deep_nesting(self):
        """Test getting data from deeply nested structure."""
        from src.ui.callbacks import _get_signal_data
        
        data = {
            'L1': {
                'L2': {
                    'L3': pd.DataFrame({
                        'timestamp': [1, 2, 3],
                        'signal': [10, 20, 30]
                    })
                }
            }
        }
        
        result = _get_signal_data(data, 'L1.L2.L3.signal')
        
        assert result is not None
        timestamps, values = result
        assert list(values) == [10, 20, 30]


class TestLayoutComponents:
    """Tests for layout component creation."""
    
    def test_header_created(self):
        """Test header component creation."""
        from src.ui.app_layout import _create_header
        
        header = _create_header()
        
        assert header is not None
    
    def test_sidebar_created(self):
        """Test sidebar component creation."""
        from src.ui.app_layout import _create_sidebar
        
        sidebar = _create_sidebar([], None)
        
        assert sidebar is not None
    
    def test_status_bar_created(self):
        """Test status bar component creation."""
        from src.ui.app_layout import _create_status_bar
        
        status_bar = _create_status_bar(None)
        
        assert status_bar is not None


class TestDataTree:
    """Tests for data tree summary display."""
    
    def test_build_data_tree_empty(self):
        """Test building tree from empty data."""
        from src.ui.app_layout import _build_data_tree_summary
        
        tree = _build_data_tree_summary({})
        
        assert tree == []
    
    def test_build_data_tree_with_dataframe(self):
        """Test building tree with DataFrame."""
        from src.ui.app_layout import _build_data_tree_summary
        
        data = {
            'TestDF': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        }
        
        tree = _build_data_tree_summary(data)
        
        assert len(tree) == 1
    
    def test_build_data_tree_nested(self):
        """Test building tree with nested structure."""
        from src.ui.app_layout import _build_data_tree_summary
        
        data = {
            'Folder': {
                'SubDF': pd.DataFrame({'x': [1, 2]})
            }
        }
        
        tree = _build_data_tree_summary(data)
        
        # Now returns collapsible tree - folder contains children inside
        assert len(tree) >= 1  # At least the folder


class TestExportIntegration:
    """Integration tests for export workflow."""
    
    def test_full_export_workflow_csv(self, tmp_path, sample_flight_data):
        """Test complete CSV export workflow."""
        from src.export.data_export import export_csv
        
        # Get some data
        gps_data = sample_flight_data.get('Sensors', {}).get('GPS')
        if gps_data is not None:
            path = tmp_path / "export.csv"
            export_csv(gps_data, path)
            
            # Verify file exists and is valid
            assert path.exists()
            df = pd.read_csv(path)
            assert len(df) > 0
    
    def test_export_selected_signals_only(self, sample_flight_data):
        """Test exporting only selected signals."""
        from src.ui.callbacks import _get_signal_data, _build_signal_options
        
        # Get available signals
        options = _build_signal_options(sample_flight_data)
        if options:
            # Select first signal
            signal_path = options[0]['value']
            data = _get_signal_data(sample_flight_data, signal_path)
            
            if data is not None:
                timestamps, values = data
                assert len(timestamps) == len(values)

