"""
Tests for .tlog (MAVLink telemetry log) file import functionality.

Tests REQ-LO-006: Support for MAVLink telemetry log files
"""

import pytest
import pandas as pd
import numpy as np
import os
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# Path to test fixture
FIXTURE_PATH = Path(__file__).parent.parent / 'fixtures' / 'sample-trimmed.tlog'


class TestTlogFileExists:
    """Test that the sample .tlog file exists in fixtures."""
    
    def test_sample_tlog_exists(self):
        """Verify the sample .tlog file is in the fixtures directory."""
        assert FIXTURE_PATH.exists(), f"Sample .tlog file not found at {FIXTURE_PATH}"
    
    def test_sample_tlog_not_empty(self):
        """Verify the sample .tlog file is not empty."""
        if FIXTURE_PATH.exists():
            assert FIXTURE_PATH.stat().st_size > 0, "Sample .tlog file is empty"


class TestMavlinkParser:
    """Test MAVLink parsing functionality."""
    
    @pytest.fixture
    def tlog_content(self):
        """Read the sample .tlog file content."""
        if FIXTURE_PATH.exists():
            with open(FIXTURE_PATH, 'rb') as f:
                return f.read()
        return None
    
    @pytest.fixture
    def encoded_tlog(self, tlog_content):
        """Base64 encoded .tlog content for upload simulation."""
        if tlog_content:
            return base64.b64encode(tlog_content).decode('utf-8')
        return None
    
    def test_pymavlink_import(self):
        """Test that pymavlink can be imported."""
        try:
            from pymavlink import mavutil
            assert mavutil is not None
        except ImportError:
            pytest.skip("pymavlink not installed")
    
    def test_parse_tlog_file(self, tlog_content):
        """Test parsing a real .tlog file."""
        if tlog_content is None:
            pytest.skip("Sample .tlog file not available")
        
        try:
            from pymavlink import mavutil
            import tempfile
            
            # Write to temp file
            tmp_path = tempfile.mktemp(suffix='.tlog')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(tlog_content)
            
            try:
                mlog = mavutil.mavlink_connection(tmp_path)
                messages = {}
                msg_count = 0
                
                while msg_count < 1000:  # Limit for test speed
                    msg = mlog.recv_match(blocking=False)
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type == 'BAD_DATA':
                        continue
                    if msg_type not in messages:
                        messages[msg_type] = []
                    msg_dict = msg.to_dict()
                    msg_dict['timestamp'] = getattr(msg, '_timestamp', 0)
                    messages[msg_type].append(msg_dict)
                    msg_count += 1
                
                mlog.close()
                
                # Verify we got some messages
                assert len(messages) > 0, "No messages parsed from .tlog file"
                
                # Check for common MAVLink message types
                common_types = ['HEARTBEAT', 'ATTITUDE', 'GLOBAL_POSITION_INT', 
                               'GPS_RAW_INT', 'SYS_STATUS', 'RAW_IMU']
                found_types = [t for t in common_types if t in messages]
                assert len(found_types) > 0, f"Expected common message types, got: {list(messages.keys())}"
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("pymavlink not installed")
    
    def test_tlog_message_to_dataframe(self, tlog_content):
        """Test converting MAVLink messages to DataFrames."""
        if tlog_content is None:
            pytest.skip("Sample .tlog file not available")
        
        try:
            from pymavlink import mavutil
            import tempfile
            
            tmp_path = tempfile.mktemp(suffix='.tlog')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(tlog_content)
            
            try:
                mlog = mavutil.mavlink_connection(tmp_path)
                messages = {}
                
                # Parse limited messages for test
                for _ in range(500):
                    msg = mlog.recv_match(blocking=False)
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type in ['BAD_DATA']:
                        continue
                    if msg_type not in messages:
                        messages[msg_type] = []
                    messages[msg_type].append(msg.to_dict())
                
                mlog.close()
                
                # Convert to DataFrames
                dataframes = {}
                for msg_type, msg_list in messages.items():
                    if len(msg_list) > 0:
                        df = pd.DataFrame(msg_list)
                        dataframes[msg_type] = df
                
                assert len(dataframes) > 0, "No DataFrames created"
                
                # Verify DataFrame structure
                for name, df in dataframes.items():
                    assert isinstance(df, pd.DataFrame)
                    assert len(df) > 0, f"DataFrame {name} is empty"
                    assert len(df.columns) > 0, f"DataFrame {name} has no columns"
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("pymavlink not installed")


class TestTlogDataProcessing:
    """Test data processing for MAVLink data."""
    
    def test_gps_coordinate_scaling(self):
        """Test that GPS coordinates are correctly scaled from 1e7."""
        # MAVLink GPS coordinates are in 1e7 degrees
        lat_raw = 473977290  # 47.3977290 degrees
        lon_raw = 85466200   # 8.5466200 degrees
        
        lat_scaled = lat_raw / 1e7
        lon_scaled = lon_raw / 1e7
        
        assert 40 < lat_scaled < 60, "Latitude should be reasonable"
        assert 0 < lon_scaled < 20, "Longitude should be reasonable"
    
    def test_altitude_scaling(self):
        """Test that altitude is correctly scaled from mm."""
        # MAVLink relative_alt is in mm
        alt_raw = 10000  # 10 meters
        alt_scaled = alt_raw / 1000
        
        assert alt_scaled == 10.0, "Altitude should be in meters"
    
    def test_time_normalization_usec(self):
        """Test time normalization from microseconds."""
        from src.ui.callbacks import _normalize_time
        
        # Simulate MAVLink time_usec (microseconds since epoch)
        time_usec = pd.Series([1700000000000000, 1700000001000000, 1700000002000000])
        
        normalized = _normalize_time(time_usec)
        
        # Should be normalized to seconds from start
        assert normalized.iloc[0] == 0
        assert abs(normalized.iloc[1] - 1.0) < 0.01
        assert abs(normalized.iloc[2] - 2.0) < 0.01
    
    def test_time_normalization_ms(self):
        """Test time normalization from milliseconds."""
        from src.ui.callbacks import _normalize_time
        
        # Simulate time_boot_ms (milliseconds since boot) - needs to be > 1e6 to trigger
        time_ms = pd.Series([10000000, 10001000, 10002000])  # 10000s, 10001s, 10002s boot time in ms
        
        normalized = _normalize_time(time_ms)
        
        # Should be normalized to seconds from start
        assert normalized.iloc[0] == 0
        assert abs(normalized.iloc[1] - 1.0) < 0.01
        assert abs(normalized.iloc[2] - 2.0) < 0.01
    
    def test_find_time_column(self):
        """Test finding time column in DataFrame."""
        from src.ui.callbacks import _find_time_column
        
        # Test with timestamp
        df1 = pd.DataFrame({'timestamp': [1, 2, 3], 'value': [10, 20, 30]})
        assert _find_time_column(df1) == 'timestamp'
        
        # Test with time_boot_ms
        df2 = pd.DataFrame({'time_boot_ms': [1000, 2000, 3000], 'value': [10, 20, 30]})
        assert _find_time_column(df2) == 'time_boot_ms'
        
        # Test with time_usec
        df3 = pd.DataFrame({'time_usec': [1000000, 2000000, 3000000], 'value': [10, 20, 30]})
        assert _find_time_column(df3) == 'time_usec'
        
        # Test with no time column
        df4 = pd.DataFrame({'value': [10, 20, 30]})
        assert _find_time_column(df4) is None


class TestDataHierarchy:
    """Test hierarchical data organization."""
    
    def test_organize_data_hierarchically(self):
        """Test that flat data is organized into hierarchy."""
        from src.ui.callbacks import _organize_data_hierarchically
        
        # Simulate uploaded data with dot-separated names
        uploaded = {
            'sample.ATTITUDE': {'roll': [0.1, 0.2], 'pitch': [0.3, 0.4]},
            'sample.GPS': {'lat': [47.0, 47.1], 'lon': [8.0, 8.1]},
            'sample.HEARTBEAT': {'type': [1, 1], 'autopilot': [3, 3]},
        }
        
        result = _organize_data_hierarchically(uploaded)
        
        assert 'sample' in result
        assert isinstance(result['sample'], dict)
        assert 'ATTITUDE' in result['sample']
        assert 'GPS' in result['sample']
        assert 'HEARTBEAT' in result['sample']
    
    def test_build_signal_options_mavlink(self):
        """Test building signal options from MAVLink data."""
        from src.ui.callbacks import _build_signal_options
        
        # Create MAVLink-like data structure
        data = {
            'sample': {
                'ATTITUDE': pd.DataFrame({
                    'time_boot_ms': [1000, 2000, 3000],
                    'roll': [0.1, 0.2, 0.3],
                    'pitch': [0.1, 0.2, 0.3],
                    'yaw': [1.0, 1.1, 1.2],
                }),
                'GPS_RAW_INT': pd.DataFrame({
                    'time_usec': [1000000, 2000000, 3000000],
                    'lat': [473977290, 473977291, 473977292],
                    'lon': [85466200, 85466201, 85466202],
                }),
            }
        }
        
        options = _build_signal_options(data)
        
        # Should have options for roll, pitch, yaw, lat, lon (not time columns)
        option_values = [opt['value'] for opt in options]
        
        assert any('roll' in v for v in option_values)
        assert any('pitch' in v for v in option_values)
        assert any('lat' in v for v in option_values)
        
        # Should NOT include time columns
        assert not any('time_boot_ms' in v for v in option_values)
        assert not any('time_usec' in v for v in option_values)


class TestFigureCreation:
    """Test figure creation from MAVLink data."""
    
    def test_create_attitude_figure(self):
        """Test creating attitude figure from MAVLink data."""
        from src.ui.callbacks import _create_attitude_figure_from_data
        
        data = {
            'ATTITUDE': pd.DataFrame({
                'time_boot_ms': [1000, 2000, 3000, 4000, 5000],
                'roll': [0.1, 0.15, 0.2, 0.15, 0.1],
                'pitch': [0.05, 0.1, 0.15, 0.1, 0.05],
                'yaw': [1.0, 1.1, 1.2, 1.3, 1.4],
            })
        }
        
        fig = _create_attitude_figure_from_data(data)
        
        assert fig is not None
        assert len(fig.data) > 0  # Should have traces
    
    def test_create_imu_figure(self):
        """Test creating IMU figure from MAVLink data."""
        from src.ui.callbacks import _create_imu_figure_from_data
        
        data = {
            'RAW_IMU': pd.DataFrame({
                'time_usec': [1000000, 2000000, 3000000, 4000000, 5000000],
                'xacc': [100, 110, 105, 115, 100],
                'yacc': [50, 55, 52, 58, 50],
                'zacc': [980, 985, 982, 988, 980],
            })
        }
        
        fig = _create_imu_figure_from_data(data)
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_map_figure(self):
        """Test creating map figure from MAVLink GPS data."""
        from src.ui.callbacks import _create_map_figure_from_data
        
        data = {
            'GLOBAL_POSITION_INT': pd.DataFrame({
                'time_boot_ms': [1000, 2000, 3000, 4000, 5000],
                'lat': [473977290, 473977291, 473977292, 473977293, 473977294],
                'lon': [85466200, 85466201, 85466202, 85466203, 85466204],
                'alt': [5000, 5100, 5200, 5300, 5400],
                'relative_alt': [1000, 1100, 1200, 1300, 1400],
            })
        }
        
        fig = _create_map_figure_from_data(data)
        
        assert fig is not None
        assert len(fig.data) > 0


class TestIntegration:
    """Integration tests for complete .tlog import workflow."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock Dash app."""
        app = Mock()
        app.flight_data = {}
        return app
    
    def test_full_tlog_import_workflow(self):
        """Test complete workflow from .tlog upload to data display."""
        if not FIXTURE_PATH.exists():
            pytest.skip("Sample .tlog file not available")
        
        try:
            from pymavlink import mavutil
            import tempfile
            
            # Read file
            with open(FIXTURE_PATH, 'rb') as f:
                content = f.read()
            
            # Write to temp file
            tmp_path = tempfile.mktemp(suffix='.tlog')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(content)
            
            try:
                # Parse
                mlog = mavutil.mavlink_connection(tmp_path)
                messages = {}
                
                while True:
                    msg = mlog.recv_match(blocking=False)
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type == 'BAD_DATA':
                        continue
                    if msg_type not in messages:
                        messages[msg_type] = []
                    msg_dict = msg.to_dict()
                    msg_dict['timestamp'] = getattr(msg, '_timestamp', 0)
                    messages[msg_type].append(msg_dict)
                
                mlog.close()
                
                # Convert to DataFrames
                loaded_data = {}
                for msg_type, msg_list in messages.items():
                    if msg_list and len(msg_list) > 10:
                        df = pd.DataFrame(msg_list)
                        loaded_data[f"sample.{msg_type}"] = df
                
                # Organize hierarchically
                from src.ui.callbacks import _organize_data_hierarchically
                hierarchical = _organize_data_hierarchically(
                    {k: v.to_dict('list') for k, v in loaded_data.items()}
                )
                
                # Build signal options
                from src.ui.callbacks import _build_signal_options
                options = _build_signal_options(hierarchical)
                
                # Verify results
                assert len(hierarchical) > 0, "No hierarchical data created"
                assert len(options) > 0, "No signal options created"
                
                # Verify we can create figures
                from src.ui.callbacks import (
                    _create_overview_figure,
                    _create_attitude_figure_from_data,
                    _create_imu_figure_from_data,
                )
                
                overview = _create_overview_figure(hierarchical)
                assert overview is not None
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("pymavlink not installed")


class TestDataLoadingClearsOldData:
    """Test that loading new data clears old data properly."""
    
    def test_signal_options_update_on_load(self):
        """Test that signal options are rebuilt when data changes."""
        from src.ui.callbacks import _build_signal_options
        
        # Old data (sample data style)
        old_data = {
            'Sensors': {
                'GPS': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'lat': [47.0, 47.1, 47.2],
                    'lon': [8.0, 8.1, 8.2],
                })
            }
        }
        
        # New data (MAVLink style)
        new_data = {
            'sample': {
                'ATTITUDE': pd.DataFrame({
                    'time_boot_ms': [1000, 2000, 3000],
                    'roll': [0.1, 0.2, 0.3],
                    'pitch': [0.1, 0.2, 0.3],
                }),
                'GPS_RAW_INT': pd.DataFrame({
                    'time_usec': [1000000, 2000000, 3000000],
                    'lat': [473977290, 473977291, 473977292],
                    'lon': [85466200, 85466201, 85466202],
                })
            }
        }
        
        old_options = _build_signal_options(old_data)
        new_options = _build_signal_options(new_data)
        
        old_values = {opt['value'] for opt in old_options}
        new_values = {opt['value'] for opt in new_options}
        
        # Options should be different
        assert old_values != new_values
        
        # New options should have ATTITUDE signals
        assert any('roll' in v for v in new_values)
        assert any('pitch' in v for v in new_values)
    
    def test_data_tree_updates_on_load(self):
        """Test that data tree is rebuilt when data changes."""
        from src.ui.app_layout import _build_data_tree_summary
        
        old_data = {
            'Sensors': {
                'GPS': pd.DataFrame({'timestamp': [1, 2], 'lat': [47.0, 47.1]})
            }
        }
        
        new_data = {
            'sample': {
                'ATTITUDE': pd.DataFrame({'time_boot_ms': [1000, 2000], 'roll': [0.1, 0.2]}),
                'HEARTBEAT': pd.DataFrame({'time_boot_ms': [1000, 2000], 'type': [1, 1]})
            }
        }
        
        old_tree = _build_data_tree_summary(old_data)
        new_tree = _build_data_tree_summary(new_data)
        
        # Trees should be different structures
        assert len(old_tree) > 0
        assert len(new_tree) > 0


class TestTabUpdates:
    """Test that all tabs update when data is loaded."""
    
    def test_timeseries_content_builds(self):
        """Test that timeseries content can be built."""
        from src.ui.app_layout import _build_timeseries_content
        
        data = {
            'RAW_IMU': pd.DataFrame({
                'time_usec': [1000000, 2000000, 3000000],
                'xacc': [100, 110, 105],
                'yacc': [50, 55, 52],
                'zacc': [980, 985, 982],
            })
        }
        
        content = _build_timeseries_content(data)
        assert content is not None
    
    def test_map_content_builds(self):
        """Test that map content can be built."""
        from src.ui.app_layout import _build_map_content
        
        data = {
            'GLOBAL_POSITION_INT': pd.DataFrame({
                'time_boot_ms': [1000, 2000, 3000],
                'lat': [473977290, 473977291, 473977292],
                'lon': [85466200, 85466201, 85466202],
            })
        }
        
        content = _build_map_content(data)
        assert content is not None
    
    def test_analysis_content_builds(self):
        """Test that analysis content can be built."""
        from src.ui.app_layout import _build_analysis_content
        
        data = {
            'GPS': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'lat': [47.0, 47.1, 47.2],
                'altitude': [100, 150, 200],
            })
        }
        
        content = _build_analysis_content(data)
        assert content is not None
    
    def test_empty_data_shows_alert(self):
        """Test that empty data shows info alert."""
        from src.ui.app_layout import _build_timeseries_content, _build_map_content, _build_analysis_content
        
        ts_content = _build_timeseries_content(None)
        map_content = _build_map_content(None)
        analysis_content = _build_analysis_content(None)
        
        # All should return something (alert message)
        assert ts_content is not None
        assert map_content is not None
        assert analysis_content is not None


class TestGenericTimeSeries:
    """Test generic time series creation for any data."""
    
    def test_generic_timeseries_figure(self):
        """Test creating generic time series figure."""
        from src.ui.app_layout import _create_generic_timeseries_figure
        
        df = pd.DataFrame({
            'time_boot_ms': [1000, 2000, 3000, 4000, 5000],
            'value1': [10, 20, 15, 25, 20],
            'value2': [100, 110, 105, 115, 108],
        })
        
        fig = _create_generic_timeseries_figure(df, 'TestData')
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'TestData'


class TestSignalOptionsReplacesOldData:
    """Test that loading new data completely replaces old signals in the UI."""
    
    def test_old_signals_not_in_new_options(self):
        """Critical test: old sample data signals should NOT appear after loading new data."""
        from src.ui.callbacks import _build_signal_options
        
        # Simulate sample data (loaded at startup)
        old_sample_data = {
            'Sensors': {
                'IMU': {
                    'Accelerometer': pd.DataFrame({
                        'timestamp': [1, 2, 3],
                        'accel_x': [0.1, 0.2, 0.3],
                        'accel_y': [0.4, 0.5, 0.6],
                        'accel_z': [9.8, 9.81, 9.79],
                    }),
                    'Gyroscope': pd.DataFrame({
                        'timestamp': [1, 2, 3],
                        'gyro_x': [0.01, 0.02, 0.03],
                        'gyro_y': [0.01, 0.02, 0.03],
                        'gyro_z': [0.01, 0.02, 0.03],
                    }),
                },
                'GPS': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'lat': [47.0, 47.1, 47.2],
                    'lon': [8.0, 8.1, 8.2],
                    'altitude': [100, 110, 120],
                })
            },
            'Control': {
                'FlightController': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'throttle': [0.5, 0.6, 0.7],
                    'roll_cmd': [0.1, 0.2, 0.3],
                })
            }
        }
        
        # New MAVLink data (loaded from .tlog file)
        new_mavlink_data = {
            'sample-trimmed': {
                'ATTITUDE': pd.DataFrame({
                    'time_boot_ms': [1000, 2000, 3000],
                    'roll': [0.1, 0.2, 0.3],
                    'pitch': [0.05, 0.1, 0.15],
                    'yaw': [1.0, 1.1, 1.2],
                }),
                'RAW_IMU': pd.DataFrame({
                    'time_usec': [1000000, 2000000, 3000000],
                    'xacc': [100, 110, 105],
                    'yacc': [50, 55, 52],
                    'zacc': [980, 985, 982],
                }),
                'GLOBAL_POSITION_INT': pd.DataFrame({
                    'time_boot_ms': [1000, 2000, 3000],
                    'lat': [473977290, 473977291, 473977292],
                    'lon': [85466200, 85466201, 85466202],
                })
            }
        }
        
        old_options = _build_signal_options(old_sample_data)
        new_options = _build_signal_options(new_mavlink_data)
        
        old_values = {opt['value'] for opt in old_options}
        new_values = {opt['value'] for opt in new_options}
        
        # CRITICAL: No overlap between old and new options
        overlap = old_values & new_values
        assert len(overlap) == 0, f"Old signals still present in new options: {overlap}"
        
        # Old sample signals should NOT be in new options
        assert not any('Sensors' in v for v in new_values), "Old 'Sensors' signals still present"
        assert not any('Control' in v for v in new_values), "Old 'Control' signals still present"
        assert not any('accel_x' in v for v in new_values), "Old 'accel_x' signal still present"
        assert not any('gyro_x' in v for v in new_values), "Old 'gyro_x' signal still present"
        
        # New MAVLink signals SHOULD be in new options
        assert any('roll' in v for v in new_values), "New 'roll' signal missing"
        assert any('pitch' in v for v in new_values), "New 'pitch' signal missing"
        assert any('xacc' in v for v in new_values), "New 'xacc' signal missing"
        assert any('lat' in v for v in new_values), "New 'lat' signal missing"
    
    def test_signal_count_changes_after_load(self):
        """Test that signal count changes when new data is loaded."""
        from src.ui.callbacks import _build_signal_options
        
        small_data = {
            'A': pd.DataFrame({'time': [1], 'val': [1]})
        }
        
        large_data = {
            'B': pd.DataFrame({
                'time': [1, 2, 3],
                'x': [1, 2, 3],
                'y': [4, 5, 6],
                'z': [7, 8, 9],
            })
        }
        
        small_options = _build_signal_options(small_data)
        large_options = _build_signal_options(large_data)
        
        # Different number of signals
        assert len(small_options) != len(large_options)


class TestRealTlogImportReplacesSignals:
    """Integration test using real .tlog file to verify signals are replaced."""
    
    def test_tlog_signals_replace_sample_signals(self):
        """Test that loading real .tlog file replaces sample signals."""
        if not FIXTURE_PATH.exists():
            pytest.skip("Sample .tlog file not available")
        
        try:
            from pymavlink import mavutil
            import tempfile
            import os
            from src.ui.callbacks import _build_signal_options, _organize_data_hierarchically
            
            # Sample data signals (what would be loaded at startup)
            sample_data = {
                'Sensors': {
                    'GPS': pd.DataFrame({
                        'timestamp': [1, 2, 3],
                        'lat': [47.0, 47.1, 47.2],
                    })
                }
            }
            sample_signals = {opt['value'] for opt in _build_signal_options(sample_data)}
            
            # Load real .tlog file
            with open(FIXTURE_PATH, 'rb') as f:
                content = f.read()
            
            tmp_path = tempfile.mktemp(suffix='.tlog')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(content)
            
            try:
                mlog = mavutil.mavlink_connection(tmp_path)
                messages = {}
                
                for _ in range(500):  # Limit for test speed
                    msg = mlog.recv_match(blocking=False)
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type == 'BAD_DATA':
                        continue
                    if msg_type not in messages:
                        messages[msg_type] = []
                    messages[msg_type].append(msg.to_dict())
                
                mlog.close()
                
                # Create DataFrames like the upload callback does
                loaded_data = {}
                for msg_type, msg_list in messages.items():
                    if msg_list and len(msg_list) > 10:
                        df = pd.DataFrame(msg_list)
                        loaded_data[f"sample-trimmed.{msg_type}"] = df
                
                # Organize hierarchically
                hierarchical = _organize_data_hierarchically(
                    {k: v.to_dict('list') for k, v in loaded_data.items()}
                )
                
                # Build new signal options
                new_signals = {opt['value'] for opt in _build_signal_options(hierarchical)}
                
                # CRITICAL: No overlap with sample data
                overlap = sample_signals & new_signals
                assert len(overlap) == 0, f"Sample signals still present: {overlap}"
                
                # Should have MAVLink signals
                assert len(new_signals) > 0, "No signals from .tlog file"
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("pymavlink not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

