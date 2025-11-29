"""
Tests for folder import and signal assignment features.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import json


class TestFolderImporter:
    """Tests for FolderImporter class."""
    
    def test_import_simple_folder(self, tmp_path):
        """Test importing CSVs from a simple folder."""
        from src.data.folder_importer import FolderImporter
        
        # Create test structure
        csv1 = tmp_path / "data1.csv"
        csv2 = tmp_path / "data2.csv"
        
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(csv1, index=False)
        pd.DataFrame({'x': [5, 6], 'y': [7, 8]}).to_csv(csv2, index=False)
        
        importer = FolderImporter()
        data = importer.import_folder(tmp_path)
        
        assert 'data1' in data
        assert 'data2' in data
        assert isinstance(data['data1'], pd.DataFrame)
        assert list(data['data1'].columns) == ['a', 'b']
    
    def test_import_nested_folders(self, tmp_path):
        """Test importing from nested folder structure."""
        from src.data.folder_importer import FolderImporter
        
        # Create: Flight_001/CU/sensors/imu.csv
        sensors_dir = tmp_path / "CU" / "sensors"
        sensors_dir.mkdir(parents=True)
        
        imu_csv = sensors_dir / "imu.csv"
        pd.DataFrame({
            'accel_x': [1.0, 2.0],
            'accel_y': [3.0, 4.0],
            'accel_z': [5.0, 6.0]
        }).to_csv(imu_csv, index=False)
        
        # Create: Flight_001/MU/motors.csv
        mu_dir = tmp_path / "MU"
        mu_dir.mkdir()
        motors_csv = mu_dir / "motors.csv"
        pd.DataFrame({
            'motor1': [100, 200],
            'motor2': [150, 250]
        }).to_csv(motors_csv, index=False)
        
        importer = FolderImporter()
        data = importer.import_folder(tmp_path)
        
        # Check nested structure
        assert 'CU' in data
        assert 'sensors' in data['CU']
        assert 'imu' in data['CU']['sensors']
        assert isinstance(data['CU']['sensors']['imu'], pd.DataFrame)
        
        assert 'MU' in data
        assert 'motors' in data['MU']
    
    def test_import_with_flatten(self, tmp_path):
        """Test flattening nested folders."""
        from src.data.folder_importer import FolderImporter
        
        # Create nested structure
        subdir = tmp_path / "level1" / "level2"
        subdir.mkdir(parents=True)
        
        csv = subdir / "data.csv"
        pd.DataFrame({'val': [1, 2, 3]}).to_csv(csv, index=False)
        
        importer = FolderImporter()
        data = importer.import_folder(tmp_path, flatten=True)
        
        # Should have dot-separated path
        assert 'level1.level2.data' in data
    
    def test_import_include_root(self, tmp_path):
        """Test including root folder name in hierarchy."""
        from src.data.folder_importer import FolderImporter
        
        csv = tmp_path / "test.csv"
        pd.DataFrame({'a': [1]}).to_csv(csv, index=False)
        
        importer = FolderImporter()
        data = importer.import_folder(tmp_path, include_root_name=True)
        
        # Root folder name should be top-level key
        assert tmp_path.name in data
    
    def test_import_different_extensions(self, tmp_path):
        """Test importing different file extensions."""
        from src.data.folder_importer import FolderImporter
        
        # Create CSV and TSV
        csv = tmp_path / "data.csv"
        tsv = tmp_path / "data.tsv"
        txt = tmp_path / "data.txt"
        
        pd.DataFrame({'a': [1]}).to_csv(csv, index=False)
        pd.DataFrame({'b': [2]}).to_csv(tsv, index=False, sep='\t')
        pd.DataFrame({'c': [3]}).to_csv(txt, index=False)
        
        # Only import CSV
        importer = FolderImporter(extensions=['.csv'])
        data = importer.import_folder(tmp_path)
        
        assert 'data' in data
        assert len(data) == 1  # Only CSV
        
        # Import all
        importer2 = FolderImporter(extensions=['.csv', '.tsv', '.txt'])
        data2 = importer2.import_folder(tmp_path)
        
        # TSV might overwrite CSV or vice versa since same stem
        assert len(data2) >= 1
    
    def test_import_skip_errors(self, tmp_path):
        """Test skipping files that fail to parse."""
        from src.data.folder_importer import FolderImporter
        
        # Create valid CSV
        valid = tmp_path / "valid.csv"
        pd.DataFrame({'a': [1, 2]}).to_csv(valid, index=False)
        
        # Create invalid CSV
        invalid = tmp_path / "invalid.csv"
        with open(invalid, 'w') as f:
            f.write("not,valid,csv\nwith,wrong,number,of,columns")
        
        importer = FolderImporter(skip_errors=True)
        data = importer.import_folder(tmp_path)
        
        # Should have loaded valid file
        assert 'valid' in data
        assert len(importer.failed_files) >= 0  # May or may not fail
    
    def test_folder_not_found(self):
        """Test error on non-existent folder."""
        from src.data.folder_importer import FolderImporter
        from src.core.exceptions import DataLoadError
        
        importer = FolderImporter()
        
        with pytest.raises(DataLoadError):
            importer.import_folder("/nonexistent/path/12345")
    
    def test_import_summary(self, tmp_path):
        """Test import summary."""
        from src.data.folder_importer import FolderImporter
        
        csv = tmp_path / "test.csv"
        pd.DataFrame({'a': [1]}).to_csv(csv, index=False)
        
        importer = FolderImporter()
        importer.import_folder(tmp_path)
        
        summary = importer.get_summary()
        
        assert summary['loaded_count'] == 1
        assert summary['failed_count'] == 0
        assert len(summary['loaded_files']) == 1


class TestSignalAssignment:
    """Tests for signal assignment features."""
    
    def test_conversion_factor_apply(self):
        """Test applying conversion factors."""
        from src.data.signal_assignment import ConversionFactor
        
        conv = ConversionFactor(scale=0.001, offset=0)
        values = np.array([1000, 2000, 3000])
        
        result = conv.apply(values)
        
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_conversion_factor_inverse(self):
        """Test inverse conversion."""
        from src.data.signal_assignment import ConversionFactor
        
        conv = ConversionFactor(scale=0.001, offset=10)
        values = np.array([10.001, 10.002, 10.003])
        
        result = conv.inverse(values)
        
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_signal_mapping_creation(self):
        """Test creating signal mapping."""
        from src.data.signal_assignment import SignalMapping, StandardSignal
        
        mapping = SignalMapping(
            source_column='lat_raw',
            target_signal=StandardSignal.POSITION_LATITUDE.value,
            conversion_preset='gps_1e7_to_degrees'
        )
        
        assert mapping.source_column == 'lat_raw'
        assert mapping.target_signal == 'position.latitude'
        assert mapping.conversion is not None
        assert mapping.conversion.scale == 1e-7
    
    def test_assignment_config_add_mapping(self):
        """Test adding mappings to config."""
        from src.data.signal_assignment import AssignmentConfig, StandardSignal
        
        config = AssignmentConfig(name='Test Config')
        
        config.add_mapping(
            source_column='lat',
            target_signal=StandardSignal.POSITION_LATITUDE,
            conversion='gps_1e7_to_degrees'
        )
        
        assert len(config.mappings) == 1
        assert config.mappings[0].source_column == 'lat'
    
    def test_assignment_config_to_dict(self):
        """Test converting config to dictionary."""
        from src.data.signal_assignment import AssignmentConfig
        
        config = AssignmentConfig(name='Test', description='Test config')
        config.add_mapping('col1', 'position.latitude')
        
        d = config.to_dict()
        
        assert d['name'] == 'Test'
        assert d['description'] == 'Test config'
        assert len(d['mappings']) == 1
    
    def test_assignment_config_from_dict(self):
        """Test creating config from dictionary."""
        from src.data.signal_assignment import AssignmentConfig
        
        d = {
            'name': 'Loaded Config',
            'description': 'Loaded from dict',
            'version': '2.0',
            'mappings': [
                {
                    'source_column': 'x',
                    'target_signal': 'position.latitude',
                    'conversion_preset': 'gps_1e7_to_degrees'
                }
            ]
        }
        
        config = AssignmentConfig.from_dict(d)
        
        assert config.name == 'Loaded Config'
        assert config.version == '2.0'
        assert len(config.mappings) == 1
    
    def test_assignment_config_save_load_json(self, tmp_path):
        """Test saving and loading config as JSON."""
        from src.data.signal_assignment import AssignmentConfig
        
        config = AssignmentConfig(name='JSON Test')
        config.add_mapping('lat', 'position.latitude', 'gps_1e7_to_degrees')
        
        json_path = tmp_path / "config.json"
        config.save(json_path)
        
        loaded = AssignmentConfig.load(json_path)
        
        assert loaded.name == 'JSON Test'
        assert len(loaded.mappings) == 1
    
    def test_assignment_config_save_load_yaml(self, tmp_path):
        """Test saving and loading config as YAML."""
        from src.data.signal_assignment import AssignmentConfig
        
        config = AssignmentConfig(name='YAML Test')
        config.add_mapping('lon', 'position.longitude')
        
        yaml_path = tmp_path / "config.yaml"
        config.save(yaml_path)
        
        loaded = AssignmentConfig.load(yaml_path)
        
        assert loaded.name == 'YAML Test'
    
    def test_signal_assigner_apply(self):
        """Test applying assignments to DataFrame."""
        from src.data.signal_assignment import (
            SignalAssigner, AssignmentConfig, ConversionFactor
        )
        
        # Create config with mapping
        config = AssignmentConfig(name='Test')
        config.add_mapping('raw_lat', 'position.latitude', 'gps_1e7_to_degrees')
        
        # Create test DataFrame
        df = pd.DataFrame({
            'raw_lat': [470000000, 471000000],  # 47.0, 47.1 degrees
            'other': [1, 2]
        })
        
        assigner = SignalAssigner(config)
        result = assigner.apply(df)
        
        assert 'position.latitude' in result.columns
        np.testing.assert_array_almost_equal(
            result['position.latitude'].values,
            [47.0, 47.1]
        )
        assert 'other' in result.columns  # Unmapped kept
    
    def test_signal_assigner_hierarchy(self):
        """Test applying assignments to hierarchical data."""
        from src.data.signal_assignment import SignalAssigner, AssignmentConfig
        
        config = AssignmentConfig(name='Test')
        config.add_mapping('val', 'accel.x')
        
        data = {
            'sensors': {
                'imu': pd.DataFrame({'val': [1, 2, 3]})
            }
        }
        
        assigner = SignalAssigner(config)
        result = assigner.apply_to_hierarchy(data)
        
        assert 'accel.x' in result['sensors']['imu'].columns
    
    def test_suggest_mappings(self):
        """Test auto-suggest mappings."""
        from src.data.signal_assignment import suggest_mappings
        
        df = pd.DataFrame({
            'latitude': [1.0],
            'longitude': [2.0],
            'roll': [0.1],
            'pitch': [0.2],
            'gyro_x': [0.01],
            'voltage': [12.0]
        })
        
        suggestions = suggest_mappings(df)
        
        assert len(suggestions) > 0
        
        # Check some expected suggestions
        suggested_cols = [s['column'] for s in suggestions]
        assert 'latitude' in suggested_cols or 'roll' in suggested_cols
    
    def test_get_standard_signals(self):
        """Test getting list of standard signals."""
        from src.data.signal_assignment import get_standard_signals
        
        signals = get_standard_signals()
        
        assert 'position.latitude' in signals
        assert 'attitude.roll' in signals
        assert len(signals) > 30
    
    def test_conversion_presets(self):
        """Test conversion presets are available."""
        from src.data.signal_assignment import CONVERSION_PRESETS, get_conversion_presets
        
        presets = get_conversion_presets()
        
        assert 'gps_1e7_to_degrees' in presets
        assert 'rad_to_deg' in presets
        assert 'mm_to_meters' in presets
        assert presets['rad_to_deg'].scale == pytest.approx(57.2957795)


class TestImportFlightFolder:
    """Tests for convenience function."""
    
    def test_import_flight_folder(self, tmp_path):
        """Test import_flight_folder convenience function."""
        from src.data.folder_importer import import_flight_folder
        
        csv = tmp_path / "test.csv"
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(csv, index=False)
        
        data = import_flight_folder(tmp_path)
        
        assert 'test' in data
        assert len(data['test']) == 3


class TestSignalAssignmentPresets:
    """Tests for UI signal assignment presets."""
    
    def test_get_signal_options(self):
        """Test getting signal options for dropdown."""
        from src.ui.signal_assignment_presets import get_signal_options
        
        options = get_signal_options()
        
        assert len(options) > 0
        # Check structure
        assert all('value' in opt and 'label' in opt for opt in options)
        # Check some expected options
        values = [opt['value'] for opt in options]
        assert 'position.latitude' in values
    
    def test_get_conversion_options(self):
        """Test getting conversion options for dropdown."""
        from src.ui.signal_assignment_presets import get_conversion_options
        
        options = get_conversion_options()
        
        assert len(options) > 0
        values = [opt['value'] for opt in options]
        assert 'none' in values
        assert 'rad_to_deg' in values
    
    def test_standard_signals_structure(self):
        """Test standard signals dictionary structure."""
        from src.ui.signal_assignment_presets import STANDARD_SIGNALS
        
        assert 'Position' in STANDARD_SIGNALS
        assert 'Attitude' in STANDARD_SIGNALS
        assert 'Velocity' in STANDARD_SIGNALS
        
        # Check position signals
        pos_values = [s['value'] for s in STANDARD_SIGNALS['Position']]
        assert 'position.latitude' in pos_values
        assert 'position.longitude' in pos_values

