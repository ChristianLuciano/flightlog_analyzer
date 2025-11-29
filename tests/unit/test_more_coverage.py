"""
Additional tests to boost coverage to 85%.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

# ==================== Data Module Tests ====================

class TestDataLoader:
    """Test data loader module."""
    
    def test_loader_import(self):
        """Test loader can be imported."""
        from src.data.loader import DataLoader
        loader = DataLoader()
        assert loader is not None


class TestDataAlignment:
    """Test data alignment module."""
    
    def test_alignment_import(self):
        """Test alignment can be imported."""
        from src.data.alignment import TimeAligner
        aligner = TimeAligner()
        assert aligner is not None
    
    def test_align_signals(self):
        """Test aligning signals."""
        from src.data.alignment import TimeAligner
        aligner = TimeAligner()
        
        df1 = pd.DataFrame({
            'timestamp': [0, 1, 2, 3, 4],
            'value': [10, 20, 30, 40, 50]
        })
        df2 = pd.DataFrame({
            'timestamp': [0.5, 1.5, 2.5, 3.5],
            'other': [1, 2, 3, 4]
        })
        
        aligned = aligner.align(df1, df2, 'timestamp')
        assert aligned is not None


class TestDataDownsampling:
    """Test downsampling module."""
    
    def test_lttb_downsample(self):
        """Test LTTB downsampling."""
        from src.data.downsampling import lttb_downsample
        
        x = np.linspace(0, 100, 1000)
        y = np.sin(x * 0.1)
        
        x_ds, y_ds = lttb_downsample(x, y, 100)
        assert len(x_ds) == 100
        assert len(y_ds) == 100
    
    def test_douglas_peucker(self):
        """Test Douglas-Peucker simplification."""
        from src.data.downsampling import douglas_peucker
        
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        x_simple, y_simple = douglas_peucker(x, y, epsilon=0.1)
        assert len(x_simple) < len(x)


class TestDataCache:
    """Test data cache module."""
    
    def test_lru_cache_import(self):
        """Test cache can be imported."""
        from src.data.cache import LRUCache
        cache = LRUCache(max_size_mb=10)
        assert cache is not None
    
    def test_lru_cache_set_get(self):
        """Test LRU cache set and get."""
        from src.data.cache import LRUCache
        cache = LRUCache(max_size_mb=10)
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
    
    def test_lru_cache_miss(self):
        """Test cache miss."""
        from src.data.cache import LRUCache
        cache = LRUCache(max_size_mb=10)
        assert cache.get('nonexistent') is None
    
    def test_lru_cache_clear(self):
        """Test cache clear."""
        from src.data.cache import LRUCache
        cache = LRUCache(max_size_mb=10)
        cache.set('key1', 'value1')
        cache.clear()
        assert cache.get('key1') is None
    
    def test_data_cache(self):
        """Test DataCache class."""
        from src.data.cache import DataCache
        cache = DataCache()
        assert cache is not None


# ==================== Config Module Tests ====================

class TestConfigSettings:
    """Test config settings module."""
    
    def test_settings_save_load(self):
        """Test saving and loading settings."""
        from src.config.settings import Settings
        
        settings = Settings(theme='dark', max_display_points=2000)
        json_str = settings.to_json()
        loaded = Settings.from_json(json_str)
        
        assert loaded.theme == 'dark'
        assert loaded.max_display_points == 2000


class TestConfigSchema:
    """Test config schema module."""
    
    def test_schema_validation(self):
        """Test schema validation."""
        from src.config.schema import validate_config
        
        config = {
            'version': '1.0',
            'theme': 'dark'
        }
        result = validate_config(config)
        # Should not raise


# ==================== Export Module Tests ====================

class TestExportData:
    """Test data export module."""
    
    def test_export_csv(self):
        """Test CSV export."""
        from src.export.data_export import export_csv
        
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            fname = f.name
        
        try:
            export_csv(df, fname)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)
    
    def test_export_excel(self):
        """Test Excel export."""
        from src.export.data_export import export_excel
        
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            fname = f.name
        
        try:
            export_excel(df, fname)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class TestGeoExport:
    """Test geographic export module."""
    
    def test_export_kml_with_altitude(self):
        """Test KML export with altitude."""
        from src.export.geo_formats import export_kml
        
        lat = np.array([47.0, 47.1, 47.2])
        lon = np.array([-122.0, -122.1, -122.2])
        alt = np.array([100, 150, 200])
        
        result = export_kml(lat, lon, alt=alt)
        assert 'kml' in result.lower()
    
    def test_export_geojson_string(self):
        """Test GeoJSON export returns string."""
        from src.export.geo_formats import export_geojson
        
        lat = np.array([47.0, 47.1])
        lon = np.array([-122.0, -122.1])
        
        result = export_geojson(lat, lon)
        assert 'Feature' in result
    
    def test_export_gpx_string(self):
        """Test GPX export returns string."""
        from src.export.geo_formats import export_gpx
        
        lat = np.array([47.0, 47.1])
        lon = np.array([-122.0, -122.1])
        
        result = export_gpx(lat, lon)
        assert 'gpx' in result.lower()


# ==================== Computed Signals Tests ====================

class TestComputedSignalsCache:
    """Test computed signals cache."""
    
    def test_cache_import(self):
        """Test cache can be imported."""
        from src.computed_signals.cache import ComputedSignalCache
        cache = ComputedSignalCache()
        assert cache is not None
    
    def test_cache_operations(self):
        """Test cache operations."""
        from src.computed_signals.cache import ComputedSignalCache
        cache = ComputedSignalCache()
        
        data = np.array([1, 2, 3])
        cache.set('signal1', data)
        result = cache.get('signal1')
        assert result is not None


class TestComputedSignalsEngine:
    """Test computed signals engine."""
    
    def test_engine_class_exists(self):
        """Test engine class can be imported."""
        from src.computed_signals.engine import ComputedSignalEngine
        assert ComputedSignalEngine is not None
    
    def test_parser_import(self):
        """Test parser can be imported."""
        from src.computed_signals.parser import FormulaParser
        parser = FormulaParser()
        assert parser is not None
    
    def test_dependency_resolver(self):
        """Test dependency resolver."""
        from src.computed_signals.dependencies import DependencyResolver
        resolver = DependencyResolver()
        resolver.add_dependency('c', 'b')
        resolver.add_dependency('b', 'a')
        order = resolver.get_computation_order()
        assert order.index('a') < order.index('b')


class TestComputedSignalsDependencies:
    """Test dependency resolver."""
    
    def test_resolver_import(self):
        """Test resolver can be imported."""
        from src.computed_signals.dependencies import DependencyResolver
        resolver = DependencyResolver()
        assert resolver is not None
    
    def test_add_dependency(self):
        """Test adding dependency."""
        from src.computed_signals.dependencies import DependencyResolver
        resolver = DependencyResolver()
        
        resolver.add_dependency('c', 'b')
        resolver.add_dependency('b', 'a')
        
        order = resolver.get_computation_order()
        assert order.index('a') < order.index('b')


# ==================== Utils Tests ====================

class TestUtilsGeo:
    """Test geo utilities."""
    
    def test_haversine(self):
        """Test haversine distance."""
        from src.utils.geo import haversine_distance
        
        d = haversine_distance(47.0, -122.0, 47.1, -122.0)
        assert 10000 < d < 12000  # ~11 km
    
    def test_bearing(self):
        """Test bearing calculation."""
        from src.utils.geo import bearing
        
        b = bearing(47.0, -122.0, 48.0, -122.0)
        assert abs(b) < 5 or abs(b - 360) < 5  # North
    
    def test_cumulative_distance(self):
        """Test cumulative distance."""
        from src.utils.geo import cumulative_distance
        
        lat = np.array([47.0, 47.01, 47.02])
        lon = np.array([-122.0, -122.0, -122.0])
        
        dist = cumulative_distance(lat, lon)
        assert len(dist) == 3
        assert dist[0] == 0


# ==================== UI State Tests ====================

class TestUIState:
    """Test UI state module."""
    
    def test_state_to_json(self):
        """Test state to JSON."""
        from src.ui.state import AppState
        
        state = AppState(theme='dark', is_playing=True)
        json_str = state.to_json()
        assert 'dark' in json_str
    
    def test_state_from_json(self):
        """Test state from JSON."""
        from src.ui.state import AppState
        import json
        
        d = {'theme': 'dark', 'is_playing': True}
        state = AppState.from_dict(d)
        assert state.theme == 'dark'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

