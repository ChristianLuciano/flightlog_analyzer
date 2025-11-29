"""
Tests to boost coverage for various modules.

Covers: state, settings, importers, validator, geo_formats, etc.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from dataclasses import asdict


# ==================== AppState Tests ====================

class TestAppState:
    """Test AppState dataclass."""
    
    def test_default_init(self):
        """Test default initialization."""
        from src.ui.state import AppState
        state = AppState()
        assert state.data_loaded is False
        assert state.current_time == 0.0
        assert state.is_playing is False
        assert state.playback_speed == 1.0
    
    def test_custom_init(self):
        """Test custom initialization."""
        from src.ui.state import AppState
        state = AppState(
            data_loaded=True,
            current_time=5.0,
            is_playing=True
        )
        assert state.data_loaded is True
        assert state.current_time == 5.0
        assert state.is_playing is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.ui.state import AppState
        state = AppState(theme='dark')
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d['theme'] == 'dark'
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        from src.ui.state import AppState
        d = {'data_loaded': True, 'theme': 'dark'}
        state = AppState.from_dict(d)
        assert state.data_loaded is True
        assert state.theme == 'dark'
    
    def test_selected_signals(self):
        """Test selected signals list."""
        from src.ui.state import AppState
        state = AppState(selected_signals=['sig1', 'sig2'])
        assert 'sig1' in state.selected_signals


# ==================== Settings Tests ====================

class TestSettings:
    """Test Settings dataclass."""
    
    def test_default_settings(self):
        """Test default settings values."""
        from src.config.settings import Settings
        settings = Settings()
        assert settings.timestamp_column == 'timestamp'
        assert settings.max_display_points == 10000
        assert settings.theme == 'light'
    
    def test_custom_settings(self):
        """Test custom settings."""
        from src.config.settings import Settings
        settings = Settings(theme='dark', max_display_points=5000)
        assert settings.theme == 'dark'
        assert settings.max_display_points == 5000
    
    def test_to_dict(self):
        """Test to_dict method."""
        from src.config.settings import Settings
        settings = Settings()
        d = settings.to_dict()
        assert isinstance(d, dict)
        assert 'theme' in d
    
    def test_to_json(self):
        """Test to_json method."""
        from src.config.settings import Settings
        settings = Settings()
        json_str = settings.to_json()
        assert isinstance(json_str, str)
        assert 'theme' in json_str
    
    def test_from_dict(self):
        """Test from_dict method."""
        from src.config.settings import Settings
        d = {'theme': 'dark', 'max_display_points': 5000}
        settings = Settings.from_dict(d)
        assert settings.theme == 'dark'
    
    def test_from_json(self):
        """Test from_json method."""
        from src.config.settings import Settings
        import json
        d = {'theme': 'dark'}
        json_str = json.dumps(d)
        settings = Settings.from_json(json_str)
        assert settings.theme == 'dark'
    
    def test_get_settings(self):
        """Test get_settings function."""
        from src.config.settings import get_settings
        settings = get_settings()
        assert settings is not None


# ==================== ConfigSchema Tests ====================

class TestConfigSchema:
    """Test ConfigSchema class."""
    
    def test_schema_exists(self):
        """Test schema can be created."""
        from src.config.schema import ConfigSchema
        schema = ConfigSchema()
        assert schema is not None
    
    def test_validate_config(self):
        """Test validate_config function."""
        from src.config.schema import validate_config
        config = {'version': '1.0'}
        result = validate_config(config)
        # Should not raise


# ==================== ConfigLoader Tests ====================

class TestConfigLoader:
    """Test ConfigLoader class."""
    
    def test_loader_init(self):
        """Test loader initialization."""
        from src.config.loader import ConfigLoader
        loader = ConfigLoader()
        assert loader is not None
    
    def test_save_load_json(self):
        """Test saving and loading JSON."""
        from src.config.loader import ConfigLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.json')
            loader = ConfigLoader()
            config = {'key': 'value'}
            loader.save(config, path)
            loaded = loader.load(path)
            assert loaded['key'] == 'value'


# ==================== DataValidator Tests ====================

class TestDataValidator:
    """Test DataValidator class."""
    
    def test_validator_init(self):
        """Test validator initialization."""
        from src.data.validator import DataValidator
        validator = DataValidator()
        assert validator.timestamp_column == 'timestamp'
    
    def test_validate_dict_data(self):
        """Test validating dictionary data."""
        from src.data.validator import DataValidator
        validator = DataValidator()
        data = {
            'df1': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [10, 20, 30]
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True
    
    def test_validate_empty_fails(self):
        """Test empty data fails validation."""
        from src.data.validator import DataValidator
        validator = DataValidator()
        result = validator.validate({})
        assert result.is_valid is False


# ==================== Geo Export Tests ====================

class TestGeoExport:
    """Test geographic export functions."""
    
    @pytest.fixture
    def gps_coords(self):
        """Create GPS coordinates."""
        return (
            np.linspace(47.0, 47.1, 20),
            np.linspace(-122.0, -121.9, 20)
        )
    
    def test_export_kml(self, gps_coords):
        """Test KML export."""
        from src.export.geo_formats import export_kml
        lat, lon = gps_coords
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.kml')
            # export_kml(lat, lon, alt=None, path=path)
            result = export_kml(lat, lon, path=path)
            assert os.path.exists(path)
    
    def test_export_kml_string(self, gps_coords):
        """Test KML export returns string if no path."""
        from src.export.geo_formats import export_kml
        lat, lon = gps_coords
        result = export_kml(lat, lon)
        assert isinstance(result, str)
        assert 'kml' in result.lower()
    
    def test_export_geojson(self, gps_coords):
        """Test GeoJSON export."""
        from src.export.geo_formats import export_geojson
        lat, lon = gps_coords
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.geojson')
            result = export_geojson(lat, lon, path=path)
            assert os.path.exists(path)
    
    def test_export_gpx(self, gps_coords):
        """Test GPX export."""
        from src.export.geo_formats import export_gpx
        lat, lon = gps_coords
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.gpx')
            result = export_gpx(lat, lon, path=path)
            assert os.path.exists(path)


# ==================== Importers Tests ====================

class TestImporters:
    """Test data importers."""
    
    def test_csv_importer_exists(self):
        """Test CSV importer can be created."""
        from src.data.importers import CSVImporter
        importer = CSVImporter()
        assert importer is not None
    
    def test_auto_importer_exists(self):
        """Test auto importer can be created."""
        from src.data.importers import AutoImporter
        importer = AutoImporter()
        assert importer is not None
    
    def test_mavlink_importer_exists(self):
        """Test MAVLink importer can be created."""
        from src.data.importers import MAVLinkImporter
        importer = MAVLinkImporter()
        assert importer is not None


# ==================== Computed Signals Tests ====================

class TestComputedSignalsFunctions:
    """Test computed signal functions."""
    
    def test_basic_math_functions(self):
        """Test basic math functions."""
        from src.computed_signals.functions import sin, cos, sqrt, exp, log
        x = np.array([0, np.pi/2, np.pi])
        
        assert abs(sin(x)[0]) < 0.01  # sin(0) = 0
        assert abs(cos(x)[0] - 1) < 0.01  # cos(0) = 1
        assert sqrt(np.array([4]))[0] == 2
        assert abs(exp(np.array([0]))[0] - 1) < 0.01
        assert abs(log(np.array([np.e]))[0] - 1) < 0.01
    
    def test_signal_processing(self):
        """Test signal processing functions."""
        from src.computed_signals.functions import moving_avg, diff, cumsum
        x = np.array([1, 2, 3, 4, 5])
        
        ma = moving_avg(x, 3)
        assert len(ma) == len(x)
        
        d = diff(x)
        # diff pads to same length
        assert len(d) == len(x)
        
        cs = cumsum(x)
        assert cs[-1] == 15
    
    def test_stats_functions(self):
        """Test statistical functions."""
        from src.computed_signals.functions import mean, std, min_val, max_val
        x = np.array([1, 2, 3, 4, 5])
        
        assert mean(x) == 3.0
        assert min_val(x) == 1
        assert max_val(x) == 5
        assert std(x) > 0
    
    def test_geo_functions(self):
        """Test geographic functions."""
        from src.computed_signals.functions import haversine, bearing
        
        # Test haversine
        lat1 = np.array([47.0])
        lon1 = np.array([-122.0])
        lat2 = np.array([47.1])
        lon2 = np.array([-122.0])
        dist = haversine(lat1, lon1, lat2, lon2)
        assert 10000 < dist[0] < 12000  # ~11km
        
        # Test bearing
        b = bearing(47.0, -122.0, 48.0, -122.0)
        assert abs(b) < 5 or abs(b - 360) < 5  # Due north


# ==================== Parser Tests ====================

class TestFormulaParser:
    """Test FormulaParser class."""
    
    def test_parser_init(self):
        """Test parser initialization."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        assert parser is not None
    
    def test_validate_valid(self):
        """Test validating valid expression."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        result = parser.validate('x + y')
        assert result is True
    
    def test_evaluate_simple(self):
        """Test evaluating simple expression."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        result = parser.evaluate('2 + 3', {})
        assert result == 5
    
    def test_evaluate_with_variables(self):
        """Test evaluating with variables."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        result = parser.evaluate('x + y', {
            'x': np.array([1, 2]),
            'y': np.array([10, 20])
        })
        assert list(result) == [11, 22]
    
    def test_extract_variables(self):
        """Test extracting variables."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        vars = parser.extract_variables('x + y * z')
        assert 'x' in vars
        assert 'y' in vars
        assert 'z' in vars


# ==================== Time Utils Tests ====================

class TestTimeUtils:
    """Test time utility functions."""
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        from src.utils.time_utils import parse_timestamp
        result = parse_timestamp(1000.0)
        assert result == 1000.0
    
    def test_format_duration(self):
        """Test duration formatting."""
        from src.utils.time_utils import format_duration
        result = format_duration(125.5)
        assert '2' in result  # 2 minutes
    
    def test_time_to_index(self):
        """Test time to index conversion."""
        from src.utils.time_utils import time_to_index
        timestamps = np.array([0, 1, 2, 3, 4])
        idx = time_to_index(timestamps, 2.5)
        assert idx in [2, 3]


# ==================== Geo Utils Tests ====================

class TestGeoUtils:
    """Test geographic utility functions."""
    
    def test_haversine_distance(self):
        """Test distance calculation."""
        from src.utils.geo import haversine_distance
        d = haversine_distance(47.0, -122.0, 47.1, -122.0)
        assert 10000 < d < 12000


# ==================== Visualization Theme Tests ====================

class TestVisualizationTheme:
    """Test visualization theme."""
    
    def test_theme_creation(self):
        """Test theme creation."""
        from src.visualization.theme import Theme
        theme = Theme()
        assert theme.name == 'default'
    
    def test_theme_to_plotly(self):
        """Test converting to Plotly template."""
        from src.visualization.theme import Theme
        theme = Theme()
        template = theme.to_plotly_template()
        assert isinstance(template, dict)
    
    def test_theme_mode(self):
        """Test theme mode enum."""
        from src.visualization.theme import Theme, ThemeMode
        theme = Theme(mode=ThemeMode.DARK)
        assert theme.mode == ThemeMode.DARK


# ==================== Hierarchy Tests ====================

class TestHierarchy:
    """Test data hierarchy functions."""
    
    def test_hierarchy_navigator_init(self):
        """Test HierarchyNavigator initialization."""
        from src.data.hierarchy import HierarchyNavigator
        data = {'test': pd.DataFrame({'a': [1, 2, 3]})}
        nav = HierarchyNavigator(data)
        assert nav is not None
    
    def test_hierarchy_navigator_get_dataframe(self):
        """Test getting dataframe."""
        from src.data.hierarchy import HierarchyNavigator
        df = pd.DataFrame({'a': [1, 2, 3]})
        data = {'test': df}
        nav = HierarchyNavigator(data)
        result = nav.get_dataframe('test')
        assert result is not None
    
    def test_hierarchy_navigator_list_paths(self):
        """Test listing all paths."""
        from src.data.hierarchy import HierarchyNavigator
        data = {
            'group1': pd.DataFrame({'a': [1, 2]}),
            'group2': pd.DataFrame({'b': [3, 4]})
        }
        nav = HierarchyNavigator(data)
        paths = nav.list_all_paths()
        assert len(paths) >= 2
    
    def test_tree_node(self):
        """Test TreeNode dataclass."""
        from src.data.hierarchy import TreeNode
        node = TreeNode(name='test', path='test', is_dataframe=True, children=[])
        assert node.name == 'test'
        assert node.is_dataframe is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

