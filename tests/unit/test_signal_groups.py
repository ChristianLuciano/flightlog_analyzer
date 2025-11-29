"""
Tests for signal grouping and multi-source signal management.
"""

import pytest
import pandas as pd
import numpy as np


class TestSignalSource:
    """Tests for SignalSource enum."""
    
    def test_signal_source_values(self):
        """Test SignalSource enum values."""
        from src.data.signal_groups import SignalSource
        
        assert SignalSource.MEASUREMENT.value == "measurement"
        assert SignalSource.COMMAND.value == "command"
        assert SignalSource.ESTIMATED.value == "estimated"
        assert SignalSource.REFERENCE.value == "reference"
        assert SignalSource.SIMULATED.value == "simulated"
        assert SignalSource.RAW.value == "raw"
        assert SignalSource.UNKNOWN.value == ""


class TestSourceStyles:
    """Tests for source styling configurations."""
    
    def test_source_styles_exist(self):
        """Test that all sources have styles defined."""
        from src.data.signal_groups import SignalSource, SOURCE_STYLES
        
        for source in SignalSource:
            assert source in SOURCE_STYLES
    
    def test_source_style_properties(self):
        """Test source style properties."""
        from src.data.signal_groups import SignalSource, SOURCE_STYLES
        
        for source, style in SOURCE_STYLES.items():
            assert 'color_index' in style
            assert 'dash' in style
            assert 'opacity' in style
            assert 'width' in style
            assert 'label_suffix' in style
    
    def test_measurement_style(self):
        """Test measurement source style."""
        from src.data.signal_groups import SignalSource, SOURCE_STYLES
        
        style = SOURCE_STYLES[SignalSource.MEASUREMENT]
        assert style['dash'] == 'solid'
        assert style['label_suffix'] == ' (meas)'
    
    def test_command_style(self):
        """Test command source style."""
        from src.data.signal_groups import SignalSource, SOURCE_STYLES
        
        style = SOURCE_STYLES[SignalSource.COMMAND]
        assert style['dash'] == 'dash'
        assert style['label_suffix'] == ' (cmd)'
    
    def test_estimated_style(self):
        """Test estimated source style."""
        from src.data.signal_groups import SignalSource, SOURCE_STYLES
        
        style = SOURCE_STYLES[SignalSource.ESTIMATED]
        assert style['dash'] == 'dot'
        assert style['label_suffix'] == ' (est)'


class TestParsedSignal:
    """Tests for ParsedSignal dataclass."""
    
    def test_parsed_signal_creation(self):
        """Test creating a ParsedSignal."""
        from src.data.signal_groups import ParsedSignal, SignalSource
        
        parsed = ParsedSignal(
            full_path='measurement.sensors.gps.latitude',
            source=SignalSource.MEASUREMENT,
            base_signal='sensors.gps.latitude',
            column_name='latitude',
            dataframe_path='measurement.sensors.gps'
        )
        
        assert parsed.full_path == 'measurement.sensors.gps.latitude'
        assert parsed.source == SignalSource.MEASUREMENT
        assert parsed.base_signal == 'sensors.gps.latitude'
        assert parsed.column_name == 'latitude'
    
    def test_group_key(self):
        """Test group_key property."""
        from src.data.signal_groups import ParsedSignal, SignalSource
        
        parsed = ParsedSignal(
            full_path='est.position.latitude',
            source=SignalSource.ESTIMATED,
            base_signal='position.latitude',
            column_name='latitude',
            dataframe_path='est.position'
        )
        
        assert parsed.group_key == 'position.latitude'
    
    def test_get_style(self):
        """Test get_style method."""
        from src.data.signal_groups import ParsedSignal, SignalSource
        
        parsed = ParsedSignal(
            full_path='cmd.attitude.roll',
            source=SignalSource.COMMAND,
            base_signal='attitude.roll',
            column_name='roll',
            dataframe_path='cmd.attitude'
        )
        
        style = parsed.get_style()
        assert style['dash'] == 'dash'
        assert style['label_suffix'] == ' (cmd)'


class TestSignalGroup:
    """Tests for SignalGroup dataclass."""
    
    def test_signal_group_creation(self):
        """Test creating a SignalGroup."""
        from src.data.signal_groups import SignalGroup
        
        group = SignalGroup(base_signal='position.latitude', color_index=0)
        
        assert group.base_signal == 'position.latitude'
        assert group.color_index == 0
        assert len(group.signals) == 0
    
    def test_add_signal(self):
        """Test adding signals to group."""
        from src.data.signal_groups import SignalGroup, ParsedSignal, SignalSource
        
        group = SignalGroup(base_signal='position.latitude', color_index=0)
        
        parsed1 = ParsedSignal(
            full_path='meas.position.latitude',
            source=SignalSource.MEASUREMENT,
            base_signal='position.latitude',
            column_name='latitude',
            dataframe_path='meas.position'
        )
        parsed2 = ParsedSignal(
            full_path='est.position.latitude',
            source=SignalSource.ESTIMATED,
            base_signal='position.latitude',
            column_name='latitude',
            dataframe_path='est.position'
        )
        
        group.add_signal(parsed1)
        group.add_signal(parsed2)
        
        assert len(group.signals) == 2
    
    def test_get_sources(self):
        """Test getting source types from group."""
        from src.data.signal_groups import SignalGroup, ParsedSignal, SignalSource
        
        group = SignalGroup(base_signal='attitude.roll', color_index=1)
        
        group.add_signal(ParsedSignal(
            full_path='meas.attitude.roll',
            source=SignalSource.MEASUREMENT,
            base_signal='attitude.roll',
            column_name='roll',
            dataframe_path='meas.attitude'
        ))
        group.add_signal(ParsedSignal(
            full_path='cmd.attitude.roll',
            source=SignalSource.COMMAND,
            base_signal='attitude.roll',
            column_name='roll',
            dataframe_path='cmd.attitude'
        ))
        
        sources = group.get_sources()
        
        assert SignalSource.MEASUREMENT in sources
        assert SignalSource.COMMAND in sources
    
    def test_base_color(self):
        """Test base_color property."""
        from src.data.signal_groups import SignalGroup, SIGNAL_COLOR_PALETTE
        
        group = SignalGroup(base_signal='test', color_index=0)
        assert group.base_color == SIGNAL_COLOR_PALETTE[0]
        
        group2 = SignalGroup(base_signal='test2', color_index=5)
        assert group2.base_color == SIGNAL_COLOR_PALETTE[5]


class TestSignalGroupManager:
    """Tests for SignalGroupManager class."""
    
    def test_parse_measurement_signal(self):
        """Test parsing measurement signals."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        # Test various measurement patterns
        parsed = manager.parse_signal('measurement.gps.latitude')
        assert parsed.source == SignalSource.MEASUREMENT
        
        parsed = manager.parse_signal('sensor.imu.accel_x')
        assert parsed.source == SignalSource.MEASUREMENT
        
        parsed = manager.parse_signal('meas_position_x')
        assert parsed.source == SignalSource.MEASUREMENT
    
    def test_parse_command_signal(self):
        """Test parsing command signals."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        parsed = manager.parse_signal('cmd.attitude.roll')
        assert parsed.source == SignalSource.COMMAND
        
        parsed = manager.parse_signal('setpoint.velocity.x')
        assert parsed.source == SignalSource.COMMAND
        
        parsed = manager.parse_signal('target.position.altitude')
        assert parsed.source == SignalSource.COMMAND
    
    def test_parse_estimated_signal(self):
        """Test parsing estimated signals."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        parsed = manager.parse_signal('estimated.position.x')
        assert parsed.source == SignalSource.ESTIMATED
        
        parsed = manager.parse_signal('ekf.attitude.yaw')
        assert parsed.source == SignalSource.ESTIMATED
        
        parsed = manager.parse_signal('filtered.velocity.z')
        assert parsed.source == SignalSource.ESTIMATED
    
    def test_parse_unknown_signal(self):
        """Test parsing signals without source prefix."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        parsed = manager.parse_signal('gps.latitude')
        assert parsed.source == SignalSource.UNKNOWN
        
        parsed = manager.parse_signal('attitude.roll')
        assert parsed.source == SignalSource.UNKNOWN
    
    def test_extract_base_signal(self):
        """Test extracting base signal from path."""
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        
        parsed = manager.parse_signal('measurement.position.latitude')
        assert parsed.base_signal == 'position.latitude'
        
        parsed = manager.parse_signal('cmd.attitude.roll')
        assert parsed.base_signal == 'attitude.roll'
    
    def test_group_signals(self):
        """Test grouping multiple signals."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        signals = [
            'measurement.position.latitude',
            'estimated.position.latitude',
            'command.position.latitude',
            'measurement.position.longitude',
            'estimated.position.longitude',
        ]
        
        groups = manager.group_signals(signals)
        
        # Should have 2 groups: position.latitude and position.longitude
        assert len(groups) == 2
        assert 'position.latitude' in groups
        assert 'position.longitude' in groups
        
        # Latitude group should have 3 signals
        assert len(groups['position.latitude'].signals) == 3
        
        # Longitude group should have 2 signals
        assert len(groups['position.longitude'].signals) == 2
    
    def test_get_plot_config(self):
        """Test getting plot configuration."""
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        
        # Parse a signal first
        manager.parse_signal('measurement.position.latitude')
        
        config = manager.get_plot_config('measurement.position.latitude', group_color_index=0)
        
        assert 'name' in config
        assert 'line' in config
        assert 'dash' in config['line']
        assert 'color' in config['line']
        assert 'opacity' in config
    
    def test_modify_color_lighter(self):
        """Test color modification (lighter)."""
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        
        # Lighten blue
        lighter = manager._modify_color('#1f77b4', 1.2)
        
        # Should be lighter (higher RGB values)
        assert lighter != '#1f77b4'
    
    def test_modify_color_darker(self):
        """Test color modification (darker)."""
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        
        # Darken blue
        darker = manager._modify_color('#1f77b4', 0.7)
        
        # Should be darker (lower RGB values)
        assert darker != '#1f77b4'


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_group_signals_for_plotting(self):
        """Test group_signals_for_plotting function."""
        from src.data.signal_groups import group_signals_for_plotting
        
        signals = [
            'meas.roll',
            'cmd.roll',
            'est.roll',
        ]
        
        result = group_signals_for_plotting(signals)
        
        assert len(result) > 0
        # Each value should be a list of (path, config) tuples
        for base_signal, configs in result.items():
            assert isinstance(configs, list)
            for path, config in configs:
                assert isinstance(path, str)
                assert isinstance(config, dict)
    
    def test_suggest_grouped_plots(self):
        """Test suggest_grouped_plots function."""
        from src.data.signal_groups import suggest_grouped_plots
        
        signals = [
            'measurement.position.latitude',
            'estimated.position.latitude',
            'measurement.position.longitude',
            'estimated.position.longitude',
            'attitude.roll',  # Single signal, should not suggest
        ]
        
        suggestions = suggest_grouped_plots(signals)
        
        # Should suggest plots for signals with multiple sources
        assert len(suggestions) >= 1
        
        for suggestion in suggestions:
            assert 'title' in suggestion
            assert 'signals' in suggestion
            assert len(suggestion['signals']) > 1  # Only multi-source suggestions


class TestSignalGroupIntegration:
    """Integration tests for signal grouping."""
    
    def test_full_workflow(self):
        """Test complete workflow of grouping and plotting config."""
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        
        # Simulate a flight log with multiple signal sources
        signals = [
            'sensor.gps.latitude',
            'sensor.gps.longitude', 
            'sensor.gps.altitude',
            'ekf.position.latitude',
            'ekf.position.longitude',
            'ekf.position.altitude',
            'cmd.position.latitude',
            'cmd.position.longitude',
            'cmd.position.altitude',
            'sensor.imu.accel_x',
            'ekf.velocity.x',
        ]
        
        groups = manager.group_signals(signals)
        
        # Should group related signals together
        assert len(groups) > 0
        
        # Each group should have proper styling
        for base_signal, group in groups.items():
            for parsed in group.signals:
                config = manager.get_plot_config(parsed.full_path, group.color_index)
                
                # All configs should have required fields
                assert 'name' in config
                assert 'line' in config
                assert 'color' in config['line']
                assert 'dash' in config['line']
    
    def test_with_real_data_structure(self):
        """Test with realistic flight data structure."""
        from src.data.signal_groups import SignalGroupManager, SignalSource
        
        manager = SignalGroupManager()
        
        # Simulate paths from a real flight log
        signals = [
            'Flight_001.CU.measurement.gps.lat',
            'Flight_001.CU.measurement.gps.lon',
            'Flight_001.CU.estimation.ekf.lat',
            'Flight_001.CU.estimation.ekf.lon',
            'Flight_001.MU.command.setpoints.lat',
            'Flight_001.MU.command.setpoints.lon',
        ]
        
        groups = manager.group_signals(signals)
        
        # Should identify sources even with deep nesting
        for base_signal, group in groups.items():
            for parsed in group.signals:
                # Source should be detected
                if 'measurement' in parsed.full_path:
                    assert parsed.source == SignalSource.MEASUREMENT
                elif 'command' in parsed.full_path or 'setpoint' in parsed.full_path:
                    assert parsed.source == SignalSource.COMMAND

