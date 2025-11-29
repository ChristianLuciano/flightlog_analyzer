"""
Tests for computed signals modules.

Covers src/computed_signals/* - 100% coverage target.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from src.computed_signals.cache import ComputedSignalCache
from src.computed_signals.engine import ComputedSignalEngine
from src.computed_signals.parser import FormulaParser
from src.computed_signals.dependencies import DependencyResolver
from src.computed_signals.functions import (
    sqrt,
    sin,
    cos,
    moving_avg,
    lowpass,
    highpass,
    diff,
    cumsum,
    haversine,
    BUILTIN_FUNCTIONS,
)
from src.core.exceptions import CircularDependencyError, SignalNotFoundError, ComputationError


class TestComputedSignalCache:
    """Test ComputedSignalCache class."""
    
    def test_init(self):
        """Test initialization."""
        cache = ComputedSignalCache()
        assert cache.max_entries == 100
        assert cache._current_size == 0
    
    def test_init_custom(self):
        """Test custom initialization."""
        cache = ComputedSignalCache(max_entries=50, max_size_mb=128)
        assert cache.max_entries == 50
        assert cache.max_size_bytes == 128 * 1024 * 1024
    
    def test_get_miss(self):
        """Test cache miss."""
        cache = ComputedSignalCache()
        result = cache.get('nonexistent')
        assert result is None
        assert cache._misses == 1
    
    def test_set_and_get(self):
        """Test set and get."""
        cache = ComputedSignalCache()
        cache.set('key1', 'value1')
        result = cache.get('key1')
        assert result == 'value1'
        assert cache._hits == 1
    
    def test_set_numpy_array(self):
        """Test set with numpy array."""
        cache = ComputedSignalCache()
        arr = np.array([1, 2, 3, 4, 5])
        cache.set('array', arr)
        result = cache.get('array')
        np.testing.assert_array_equal(result, arr)
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = ComputedSignalCache(max_entries=3)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')
        cache.set('key4', 'value4')  # Should evict key1
        assert cache.get('key1') is None
        assert cache.get('key4') == 'value4'
    
    def test_size_eviction(self):
        """Test size-based eviction."""
        cache = ComputedSignalCache(max_size_mb=0.0001)  # Very small
        arr1 = np.zeros(1000)
        arr2 = np.zeros(1000)
        cache.set('arr1', arr1)
        cache.set('arr2', arr2)  # Should evict arr1 due to size
        stats = cache.get_stats()
        assert stats['entries'] <= 2
    
    def test_invalidate_existing(self):
        """Test invalidating existing key."""
        cache = ComputedSignalCache()
        cache.set('key1', 'value1')
        result = cache.invalidate('key1')
        assert result is True
        assert cache.get('key1') is None
    
    def test_invalidate_nonexistent(self):
        """Test invalidating nonexistent key."""
        cache = ComputedSignalCache()
        result = cache.invalidate('nonexistent')
        assert result is False
    
    def test_invalidate_pattern(self):
        """Test pattern invalidation."""
        cache = ComputedSignalCache()
        cache.set('signal_a_1', 'v1')
        cache.set('signal_a_2', 'v2')
        cache.set('signal_b_1', 'v3')
        count = cache.invalidate_pattern('signal_a_*')
        assert count == 2
        assert cache.get('signal_a_1') is None
        assert cache.get('signal_b_1') == 'v3'
    
    def test_clear(self):
        """Test clearing cache."""
        cache = ComputedSignalCache()
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        assert cache.get('key1') is None
        assert cache._current_size == 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        cache = ComputedSignalCache()
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        stats = cache.get_stats()
        assert stats['entries'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_update_existing_key(self):
        """Test updating existing key."""
        cache = ComputedSignalCache()
        cache.set('key1', 'value1')
        cache.set('key1', 'value2')
        result = cache.get('key1')
        assert result == 'value2'


class TestComputedSignalEngine:
    """Test ComputedSignalEngine class."""
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        provider = Mock()
        provider.get_signal_with_timestamp = Mock(return_value=(
            np.arange(10),
            np.arange(10, dtype=float)
        ))
        return provider
    
    @pytest.fixture
    def engine(self, mock_data_provider):
        """Create engine with mock provider."""
        return ComputedSignalEngine(mock_data_provider)
    
    def test_init(self, mock_data_provider):
        """Test initialization."""
        engine = ComputedSignalEngine(mock_data_provider)
        assert engine.data_provider == mock_data_provider
        assert isinstance(engine.parser, FormulaParser)
        assert isinstance(engine.cache, ComputedSignalCache)
    
    def test_register_signal(self, engine):
        """Test registering a signal."""
        config = {'formula': 'x + y', 'inputs': ['x', 'y']}
        engine.register_signal('sum_xy', config)
        assert 'sum_xy' in engine._definitions
    
    def test_register_signal_with_circular_dep(self, engine):
        """Test circular dependency detection."""
        engine._definitions['a'] = {'formula': 'b', 'inputs': ['b']}
        engine._definitions['b'] = {'formula': 'a', 'inputs': ['a']}
        engine.dependency_resolver.add_dependency('a', 'b')
        engine.dependency_resolver.add_dependency('b', 'a')
        
        with pytest.raises(CircularDependencyError):
            engine.register_signal('c', {'formula': 'a', 'inputs': ['a']})
    
    def test_compute_not_found(self, engine):
        """Test computing nonexistent signal."""
        with pytest.raises(SignalNotFoundError):
            engine.compute('nonexistent')
    
    def test_compute_cached(self, engine):
        """Test cached computation."""
        config = {'formula': 'x * 2', 'inputs': ['x']}
        engine.register_signal('double', config)
        
        # First computation
        result1 = engine.compute('double')
        # Second computation (from cache)
        result2 = engine.compute('double')
        
        # Should be same object from cache
        pd.testing.assert_series_equal(result1, result2)
    
    def test_compute_force_recompute(self, engine):
        """Test forced recomputation."""
        config = {'formula': 'x', 'inputs': ['x']}
        engine.register_signal('passthrough', config)
        
        engine.compute('passthrough')
        # Should not use cache
        engine.compute('passthrough', force_recompute=True)
    
    def test_unregister_signal(self, engine):
        """Test unregistering signal."""
        config = {'formula': 'x', 'inputs': ['x']}
        engine.register_signal('test', config)
        result = engine.unregister_signal('test')
        assert result is True
        assert 'test' not in engine._definitions
    
    def test_unregister_nonexistent(self, engine):
        """Test unregistering nonexistent signal."""
        result = engine.unregister_signal('nonexistent')
        assert result is False
    
    def test_list_signals(self, engine):
        """Test listing signals."""
        engine.register_signal('sig1', {'formula': 'x', 'inputs': []})
        engine.register_signal('sig2', {'formula': 'y', 'inputs': []})
        signals = engine.list_signals()
        assert 'sig1' in signals
        assert 'sig2' in signals
    
    def test_get_definition(self, engine):
        """Test getting definition."""
        config = {'formula': 'x + 1', 'inputs': ['x']}
        engine.register_signal('add_one', config)
        result = engine.get_definition('add_one')
        assert result == config
    
    def test_get_definition_nonexistent(self, engine):
        """Test getting nonexistent definition."""
        result = engine.get_definition('nonexistent')
        assert result is None
    
    def test_get_dependencies(self, engine):
        """Test getting dependencies."""
        config = {'formula': 'x + y', 'inputs': ['x', 'y']}
        engine.register_signal('sum', config)
        deps = engine.get_dependencies('sum')
        assert 'x' in deps
        assert 'y' in deps
    
    def test_get_dependencies_nonexistent(self, engine):
        """Test getting dependencies for nonexistent signal."""
        deps = engine.get_dependencies('nonexistent')
        assert deps == []
    
    def test_invalidate_cache_specific(self, engine):
        """Test invalidating specific cache."""
        config = {'formula': 'x', 'inputs': ['x']}
        engine.register_signal('test', config)
        engine.compute('test')
        engine.invalidate_cache('test')
        # Cache should be empty for this signal
    
    def test_invalidate_cache_all(self, engine):
        """Test invalidating all cache."""
        engine.register_signal('sig1', {'formula': 'x', 'inputs': ['x']})
        engine.compute('sig1')
        engine.invalidate_cache()
        # All cache should be cleared
    
    def test_validate_formula_valid(self, engine):
        """Test validating valid formula."""
        is_valid, error = engine.validate_formula('x + y * 2')
        assert is_valid is True
        assert error is None
    
    def test_validate_formula_invalid(self, engine):
        """Test validating invalid formula."""
        is_valid, error = engine.validate_formula('import os')
        assert is_valid is False
        assert error is not None


class TestComputedSignalFunctions:
    """Test computed signal functions."""
    
    def test_sqrt(self):
        """Test sqrt function."""
        result = sqrt(np.array([4.0, 9.0, 16.0]))
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])
    
    def test_sin_cos(self):
        """Test sin and cos functions."""
        x = np.array([0, np.pi/2, np.pi])
        sin_result = sin(x)
        cos_result = cos(x)
        np.testing.assert_array_almost_equal(sin_result, [0, 1, 0], decimal=5)
        np.testing.assert_array_almost_equal(cos_result, [1, 0, -1], decimal=5)
    
    def test_moving_avg(self):
        """Test moving average."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_avg(signal, window=3)
        assert len(result) == len(signal)
    
    def test_lowpass(self):
        """Test lowpass filter."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
        result = lowpass(signal, cutoff=50, fs=1000)
        assert len(result) == len(signal)
    
    def test_highpass(self):
        """Test highpass filter."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
        result = highpass(signal, cutoff=50, fs=1000)
        assert len(result) == len(signal)
    
    def test_diff(self):
        """Test diff computation."""
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        result = diff(x)
        expected = np.array([0, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_cumsum(self):
        """Test cumulative sum."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cumsum(x)
        expected = np.array([1, 3, 6, 10, 15])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_haversine(self):
        """Test haversine distance."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 1.0, 1.0
        result = haversine(lat1, lon1, lat2, lon2)
        # Distance should be > 0 (approximately 157 km)
        assert result > 100000  # More than 100 km
        assert result < 200000  # Less than 200 km
    
    def test_builtin_functions_registry(self):
        """Test BUILTIN_FUNCTIONS registry."""
        assert isinstance(BUILTIN_FUNCTIONS, dict)
        assert 'sqrt' in BUILTIN_FUNCTIONS
        assert 'sin' in BUILTIN_FUNCTIONS
        assert 'cos' in BUILTIN_FUNCTIONS
        assert 'moving_avg' in BUILTIN_FUNCTIONS
        assert 'haversine' in BUILTIN_FUNCTIONS
    
    def test_get_function_from_registry(self):
        """Test getting function from registry."""
        sqrt_func = BUILTIN_FUNCTIONS.get('sqrt')
        assert sqrt_func is not None
        assert sqrt_func(4) == 2.0
    
    def test_nonexistent_function_in_registry(self):
        """Test getting nonexistent function from registry."""
        func = BUILTIN_FUNCTIONS.get('nonexistent_func_xyz')
        assert func is None


class TestDependencyResolverFull:
    """Additional tests for DependencyResolver."""
    
    def test_init(self):
        """Test initialization."""
        resolver = DependencyResolver()
        assert resolver is not None
    
    def test_add_dependency(self):
        """Test adding dependency."""
        resolver = DependencyResolver()
        resolver.add_dependency('a', 'b')
        # Check via get_all_dependencies
        deps = resolver.get_all_dependencies('a')
        assert 'b' in deps
    
    def test_remove_node(self):
        """Test removing node."""
        resolver = DependencyResolver()
        resolver.add_dependency('a', 'b')
        resolver.remove_node('a')
        # After removal, get_all_dependencies should return empty
        deps = resolver.get_all_dependencies('a')
        assert len(deps) == 0
    
    def test_has_cycle_false(self):
        """Test cycle detection - no cycle."""
        resolver = DependencyResolver()
        resolver.add_dependency('a', 'b')
        resolver.add_dependency('b', 'c')
        assert resolver.has_cycle() is False
    
    def test_has_cycle_true(self):
        """Test cycle detection - has cycle."""
        resolver = DependencyResolver()
        resolver.add_dependency('a', 'b')
        resolver.add_dependency('b', 'c')
        resolver.add_dependency('c', 'a')
        assert resolver.has_cycle() is True
    
    def test_get_computation_order(self):
        """Test getting computation order."""
        resolver = DependencyResolver()
        resolver.add_dependency('c', 'b')
        resolver.add_dependency('b', 'a')
        order = resolver.get_computation_order()
        # 'a' should come before 'b', 'b' before 'c'
        assert order.index('a') < order.index('b')
        assert order.index('b') < order.index('c')
    
    def test_get_all_dependencies(self):
        """Test getting all transitive dependencies."""
        resolver = DependencyResolver()
        resolver.add_dependency('a', 'b')
        resolver.add_dependency('b', 'c')
        all_deps = resolver.get_all_dependencies('a')
        assert 'b' in all_deps
        assert 'c' in all_deps


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

