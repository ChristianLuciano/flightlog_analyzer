"""
Tests for configuration management.

Tests REQ-LO-015 through REQ-LO-021: Configuration management.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path

from src.config.settings import Settings
from src.config.loader import ConfigLoader
from src.config.schema import validate_config


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert settings.theme in ['light', 'dark']
        assert settings.timestamp_column == 'timestamp'
        assert settings.path_delimiter == '.'

    def test_custom_settings(self):
        """Test custom settings initialization."""
        settings = Settings(
            theme='dark',
            timestamp_column='time_us',
            path_delimiter='/',
        )
        
        assert settings.theme == 'dark'
        assert settings.timestamp_column == 'time_us'
        assert settings.path_delimiter == '/'

    def test_settings_to_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings(theme='dark')
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['theme'] == 'dark'

    def test_settings_from_dict(self):
        """Test creating settings from dictionary."""
        config_dict = {
            'theme': 'light',
            'timestamp_column': 'ts',
        }
        
        settings = Settings.from_dict(config_dict)
        
        assert settings.theme == 'light'
        assert settings.timestamp_column == 'ts'


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_save_load_json(self):
        """Test saving and loading JSON config (REQ-LO-015)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            
            config = {
                'theme': 'dark',
                'tabs': [{'name': 'Overview', 'plots': []}],
                'computed_signals': {},
            }
            
            loader = ConfigLoader()
            loader.save(config, path)
            
            loaded = loader.load(path)
            
            assert loaded['theme'] == 'dark'
            assert len(loaded['tabs']) == 1

    def test_save_load_yaml(self):
        """Test saving and loading YAML config (REQ-LO-015)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.yaml'
            
            config = {
                'theme': 'light',
                'layout': {'rows': 2, 'cols': 3},
            }
            
            loader = ConfigLoader()
            loader.save(config, path)
            
            loaded = loader.load(path)
            
            assert loaded['theme'] == 'light'
            assert loaded['layout']['rows'] == 2

    def test_config_versioning(self):
        """Test configuration versioning (REQ-LO-018)."""
        config = {
            'version': '1.0.0',
            'theme': 'dark',
        }
        
        loader = ConfigLoader()
        
        # Loader should handle versioning
        assert loader.get_version(config) == '1.0.0'

    def test_partial_config_update(self):
        """Test partial configuration updates (REQ-LO-017)."""
        original = {
            'theme': 'dark',
            'tabs': [{'name': 'Tab1'}],
            'computed_signals': {'sig1': {'formula': 'x + y'}},
        }
        
        update = {
            'theme': 'light',  # Update this
            # Keep tabs and computed_signals as is
        }
        
        loader = ConfigLoader()
        merged = loader.merge_configs(original, update)
        
        assert merged['theme'] == 'light'
        assert len(merged['tabs']) == 1  # Preserved
        assert 'sig1' in merged['computed_signals']  # Preserved

    def test_computed_signal_in_config(self):
        """Test storing computed signal definitions (REQ-LO-020)."""
        config = {
            'computed_signals': {
                'total_accel': {
                    'formula': 'sqrt(accel_x**2 + accel_y**2 + accel_z**2)',
                    'inputs': ['Sensors.IMU.Accelerometer.accel_x',
                              'Sensors.IMU.Accelerometer.accel_y',
                              'Sensors.IMU.Accelerometer.accel_z'],
                    'unit': 'm/s²',
                    'description': 'Total acceleration magnitude',
                },
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            
            loader = ConfigLoader()
            loader.save(config, path)
            loaded = loader.load(path)
            
            assert 'total_accel' in loaded['computed_signals']
            assert 'formula' in loaded['computed_signals']['total_accel']


class TestConfigValidation:
    """Tests for configuration schema validation."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'version': '1.0.0',
            'theme': 'dark',
            'timestamp_column': 'timestamp',
        }
        
        assert validate_config(config) is True

    def test_invalid_theme(self):
        """Test validation rejects invalid theme."""
        config = {
            'theme': 'invalid_theme',
        }
        
        with pytest.raises(ValueError):
            validate_config(config, strict=True)

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        config = {}  # Empty config
        
        # Should not raise for optional fields
        result = validate_config(config, strict=False)
        assert result is True

    def test_computed_signal_validation(self):
        """Test validation of computed signal definitions."""
        config = {
            'computed_signals': {
                'valid_signal': {
                    'formula': 'x + y',
                    'inputs': ['path.to.x', 'path.to.y'],
                },
                'invalid_signal': {
                    # Missing formula
                    'inputs': ['x'],
                },
            },
        }
        
        # Should warn about invalid signal but not fail
        result = validate_config(config, strict=False)
        # In strict mode, should fail
        with pytest.raises(ValueError):
            validate_config(config, strict=True)


class TestConfigTemplates:
    """Tests for configuration templates (REQ-LO-021)."""

    def test_load_template(self):
        """Test loading configuration templates."""
        loader = ConfigLoader()
        
        # Check if default templates exist
        templates = loader.list_templates()
        
        assert isinstance(templates, list)

    def test_apply_template(self):
        """Test applying a template to create configuration."""
        loader = ConfigLoader()
        
        base_config = {'theme': 'dark'}
        template = {
            'tabs': [
                {'name': 'Overview', 'plots': []},
                {'name': 'GPS', 'plots': []},
            ],
        }
        
        result = loader.apply_template(base_config, template)
        
        assert result['theme'] == 'dark'
        assert len(result['tabs']) == 2

