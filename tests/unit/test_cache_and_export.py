"""
Tests for cache and export modules.

Covers src/data/cache.py and src/export/images.py - 100% coverage target.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

from src.data.cache import (
    CacheStats,
    LRUCache,
    DataCache,
    cache_key,
    cached,
)
from src.export.images import (
    export_plot_image,
    export_screenshot,
    figures_to_pdf,
)
from src.core.exceptions import ExportError


class TestCacheStats:
    """Test CacheStats dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.current_size_bytes == 0
        assert stats.entry_count == 0
    
    def test_hit_rate_zero(self):
        """Test hit rate with zero total."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculated(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75
    
    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0
    
    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0


class TestLRUCache:
    """Test LRUCache class."""
    
    def test_init(self):
        """Test initialization."""
        cache = LRUCache(max_size_mb=10, name="test")
        assert cache.name == "test"
        assert cache.max_size_bytes == 10 * 1024 * 1024
    
    def test_get_miss(self):
        """Test cache miss."""
        cache = LRUCache()
        result = cache.get('nonexistent')
        assert result is None
    
    def test_set_and_get(self):
        """Test set and get."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        result = cache.get('key1')
        assert result == 'value1'
    
    def test_set_updates_stats(self):
        """Test set updates statistics."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        stats = cache.get_stats()
        assert stats.entry_count == 1
    
    def test_get_updates_stats(self):
        """Test get updates statistics."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
    
    def test_delete_existing(self):
        """Test deleting existing key."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        result = cache.delete('key1')
        assert result is True
        assert cache.get('key1') is None
    
    def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        cache = LRUCache()
        result = cache.delete('nonexistent')
        assert result is False
    
    def test_clear(self):
        """Test clearing cache."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        stats = cache.get_stats()
        assert stats.entry_count == 0
    
    def test_contains_true(self):
        """Test contains returns True."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        assert cache.contains('key1') is True
    
    def test_contains_false(self):
        """Test contains returns False."""
        cache = LRUCache()
        assert cache.contains('nonexistent') is False
    
    def test_keys(self):
        """Test getting keys."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        keys = cache.keys()
        assert 'key1' in keys
        assert 'key2' in keys
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size_mb=0.00001)  # Very small
        cache.set('key1', 'x' * 1000)
        cache.set('key2', 'x' * 1000)
        # key1 should be evicted
        stats = cache.get_stats()
        assert stats.evictions >= 1
    
    def test_update_existing_key(self):
        """Test updating existing key."""
        cache = LRUCache()
        cache.set('key1', 'value1')
        cache.set('key1', 'value2')
        result = cache.get('key1')
        assert result == 'value2'
    
    def test_cache_numpy_array(self):
        """Test caching numpy array."""
        cache = LRUCache()
        arr = np.array([1, 2, 3, 4, 5])
        cache.set('array', arr)
        result = cache.get('array')
        np.testing.assert_array_equal(result, arr)
    
    def test_cache_dataframe(self):
        """Test caching DataFrame."""
        cache = LRUCache()
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cache.set('df', df)
        result = cache.get('df')
        pd.testing.assert_frame_equal(result, df)


class TestDataCache:
    """Test DataCache class."""
    
    def test_init(self):
        """Test initialization."""
        cache = DataCache(total_size_mb=100)
        assert cache.dataframe_cache is not None
        assert cache.signal_cache is not None
    
    def test_dataframe_cache(self):
        """Test DataFrame caching."""
        cache = DataCache()
        df = pd.DataFrame({'a': [1, 2, 3]})
        cache.set_dataframe('test_df', df)
        result = cache.get_dataframe('test_df')
        pd.testing.assert_frame_equal(result, df)
    
    def test_signal_cache(self):
        """Test signal caching."""
        cache = DataCache()
        signal = np.array([1, 2, 3])
        cache.set_signal('test_signal', signal)
        result = cache.get_signal('test_signal')
        np.testing.assert_array_equal(result, signal)
    
    def test_computed_cache(self):
        """Test computed result caching."""
        cache = DataCache()
        result_value = {'data': [1, 2, 3]}
        cache.set_computed('test_computed', result_value)
        result = cache.get_computed('test_computed')
        assert result == result_value
    
    def test_fft_cache(self):
        """Test FFT result caching."""
        cache = DataCache()
        fft_result = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        cache.set_fft('test_fft', fft_result)
        result = cache.get_fft('test_fft')
        assert result == fft_result
    
    def test_clear_all(self):
        """Test clearing all caches."""
        cache = DataCache()
        cache.set_dataframe('df', pd.DataFrame())
        cache.set_signal('sig', np.array([]))
        cache.clear_all()
        assert cache.get_dataframe('df') is None
        assert cache.get_signal('sig') is None
    
    def test_get_total_stats(self):
        """Test getting total statistics."""
        cache = DataCache()
        stats = cache.get_total_stats()
        assert 'dataframes' in stats
        assert 'signals' in stats
        assert 'computed' in stats
        assert 'fft' in stats
        assert 'misc' in stats


class TestCacheKey:
    """Test cache_key function."""
    
    def test_same_args_same_key(self):
        """Test same args produce same key."""
        key1 = cache_key('a', 'b', c=1)
        key2 = cache_key('a', 'b', c=1)
        assert key1 == key2
    
    def test_different_args_different_key(self):
        """Test different args produce different key."""
        key1 = cache_key('a', 'b')
        key2 = cache_key('a', 'c')
        assert key1 != key2
    
    def test_keyword_order_independent(self):
        """Test keyword order doesn't matter."""
        key1 = cache_key(a=1, b=2)
        key2 = cache_key(b=2, a=1)
        assert key1 == key2
    
    def test_returns_hash(self):
        """Test returns hash string."""
        key = cache_key('test')
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length


class TestCachedDecorator:
    """Test cached decorator."""
    
    def test_cached_function(self):
        """Test function is cached."""
        cache = LRUCache()
        call_count = 0
        
        @cached(cache)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once
    
    def test_cached_with_custom_key_func(self):
        """Test cached with custom key function."""
        cache = LRUCache()
        
        def my_key_func(x):
            return f"custom_key_{x}"
        
        @cached(cache, key_func=my_key_func)
        def my_func(x):
            return x * 3
        
        result = my_func(10)
        assert result == 30
        assert cache.contains('custom_key_10')


class TestExportPlotImage:
    """Test export_plot_image function."""
    
    @pytest.fixture
    def sample_figure(self):
        """Create sample figure."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        return fig
    
    @pytest.mark.skip(reason="Requires kaleido for image export")
    def test_export_png(self, sample_figure):
        """Test PNG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.png'
            export_plot_image(sample_figure, path, format='png')
            assert path.exists()
    
    @pytest.mark.skip(reason="Requires kaleido for image export")
    def test_export_svg(self, sample_figure):
        """Test SVG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.svg'
            export_plot_image(sample_figure, path, format='svg')
            assert path.exists()
    
    @pytest.mark.skip(reason="Requires kaleido for image export")
    def test_export_creates_directory(self, sample_figure):
        """Test export creates directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'subdir' / 'test.png'
            export_plot_image(sample_figure, path)
            assert path.parent.exists()
    
    @pytest.mark.skip(reason="Requires kaleido for image export")
    def test_export_screenshot(self, sample_figure):
        """Test export_screenshot wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'screenshot.png'
            export_screenshot(sample_figure, path)
            assert path.exists()


class TestFiguresToPdf:
    """Test figures_to_pdf function."""
    
    @pytest.fixture
    def sample_figures(self):
        """Create sample figures."""
        figs = []
        for i in range(3):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[i, i+1, i+2]))
            figs.append(fig)
        return figs
    
    @pytest.mark.skip(reason="Requires kaleido and Pillow for PDF export")
    def test_export_pdf(self, sample_figures):
        """Test PDF export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pdf'
            figures_to_pdf(sample_figures, path)
            assert path.exists()
    
    @pytest.mark.skip(reason="Requires kaleido and Pillow for PDF export")
    def test_export_single_figure_pdf(self, sample_figures):
        """Test PDF export with single figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'single.pdf'
            figures_to_pdf(sample_figures[:1], path)
            assert path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

