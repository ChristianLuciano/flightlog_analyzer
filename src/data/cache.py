"""
Data caching functionality.

Provides LRU caching for DataFrames, computed signals, and other
frequently accessed data with configurable size limits.
"""

import time
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import logging
import sys

from ..core.types import CacheEntry
from ..core.constants import CACHE_SIZE_MB
from ..core.exceptions import CacheError, CacheMissError, CacheFullError

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with size limits.

    Implements least-recently-used eviction policy with
    configurable memory limits.
    """

    def __init__(
        self,
        max_size_mb: float = CACHE_SIZE_MB,
        name: str = "cache"
    ):
        """
        Initialize LRU cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
            name: Cache name for logging.
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.name = name
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size_bytes=self.max_size_bytes)

    def get(self, key: str) -> Optional[T]:
        """
        Get item from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                self._stats.hits += 1
                return entry.data
            else:
                self._stats.misses += 1
                return None

    def set(self, key: str, value: T) -> None:
        """
        Set item in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            # Calculate size
            size = self._estimate_size(value)

            # Evict if necessary
            self._evict_if_needed(size)

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._stats.current_size_bytes -= old_entry.size_bytes

            # Add new entry
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                size_bytes=size,
                access_count=1,
                last_access=time.time()
            )
            self._cache[key] = entry
            self._stats.current_size_bytes += size
            self._stats.entry_count = len(self._cache)

    def delete(self, key: str) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key.

        Returns:
            True if item was deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.current_size_bytes -= entry.size_bytes
                self._stats.entry_count = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size_bytes = 0
            self._stats.entry_count = 0
            logger.info(f"Cache '{self.name}' cleared")

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache

    def keys(self) -> list:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                current_size_bytes=self._stats.current_size_bytes,
                max_size_bytes=self._stats.max_size_bytes,
                entry_count=len(self._cache)
            )

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if needed to make room."""
        while (
            self._stats.current_size_bytes + new_size > self.max_size_bytes
            and self._cache
        ):
            # Remove least recently used (first item)
            key, entry = self._cache.popitem(last=False)
            self._stats.current_size_bytes -= entry.size_bytes
            self._stats.evictions += 1
            logger.debug(f"Cache '{self.name}': evicted '{key}'")

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return sys.getsizeof(pickle.dumps(value))
        except (TypeError, pickle.PicklingError):
            return sys.getsizeof(value)


class DataCache:
    """
    Centralized data caching manager.

    Manages multiple specialized caches for different data types
    (DataFrames, signals, computed results, etc.).
    """

    def __init__(self, total_size_mb: float = CACHE_SIZE_MB):
        """
        Initialize DataCache.

        Args:
            total_size_mb: Total cache size budget in MB.
        """
        # Allocate cache budget
        self.dataframe_cache = LRUCache[Any](
            max_size_mb=total_size_mb * 0.4,
            name="dataframes"
        )
        self.signal_cache = LRUCache[Any](
            max_size_mb=total_size_mb * 0.2,
            name="signals"
        )
        self.computed_cache = LRUCache[Any](
            max_size_mb=total_size_mb * 0.25,
            name="computed"
        )
        self.fft_cache = LRUCache[Any](
            max_size_mb=total_size_mb * 0.1,
            name="fft"
        )
        self.misc_cache = LRUCache[Any](
            max_size_mb=total_size_mb * 0.05,
            name="misc"
        )

    def get_dataframe(self, key: str) -> Optional[Any]:
        """Get cached DataFrame."""
        return self.dataframe_cache.get(key)

    def set_dataframe(self, key: str, df: Any) -> None:
        """Cache DataFrame."""
        self.dataframe_cache.set(key, df)

    def get_signal(self, key: str) -> Optional[Any]:
        """Get cached signal."""
        return self.signal_cache.get(key)

    def set_signal(self, key: str, signal: Any) -> None:
        """Cache signal."""
        self.signal_cache.set(key, signal)

    def get_computed(self, key: str) -> Optional[Any]:
        """Get cached computed result."""
        return self.computed_cache.get(key)

    def set_computed(self, key: str, result: Any) -> None:
        """Cache computed result."""
        self.computed_cache.set(key, result)

    def get_fft(self, key: str) -> Optional[Any]:
        """Get cached FFT result."""
        return self.fft_cache.get(key)

    def set_fft(self, key: str, result: Any) -> None:
        """Cache FFT result."""
        self.fft_cache.set(key, result)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.dataframe_cache.clear()
        self.signal_cache.clear()
        self.computed_cache.clear()
        self.fft_cache.clear()
        self.misc_cache.clear()

    def get_total_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            "dataframes": self.dataframe_cache.get_stats(),
            "signals": self.signal_cache.get_stats(),
            "computed": self.computed_cache.get_stats(),
            "fft": self.fft_cache.get_stats(),
            "misc": self.misc_cache.get_stats(),
        }


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Hash string suitable for cache key.
    """
    key_data = (args, tuple(sorted(kwargs.items())))
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(cache: LRUCache, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.

    Args:
        cache: LRUCache instance to use.
        key_func: Optional function to generate cache key.

    Returns:
        Decorated function with caching.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache_key(func.__name__, *args, **kwargs)

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
    return decorator

