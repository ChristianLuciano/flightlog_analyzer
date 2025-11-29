"""
Caching for computed signal results.

Provides specialized caching with invalidation support
for computed signal values.
"""

import time
import fnmatch
from typing import Optional, Any, Dict
from collections import OrderedDict
import threading


class ComputedSignalCache:
    """
    Cache for computed signal results.

    Implements LRU eviction with pattern-based invalidation.
    """

    def __init__(self, max_entries: int = 100, max_size_mb: float = 256):
        """
        Initialize cache.

        Args:
            max_entries: Maximum number of cached entries.
            max_size_mb: Maximum cache size in MB.
        """
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._cache: OrderedDict = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._current_size = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        import sys

        with self._lock:
            # Estimate size
            try:
                size = sys.getsizeof(value)
                if hasattr(value, 'nbytes'):
                    size = value.nbytes
            except:
                size = 1000

            # Evict if necessary
            while (self._current_size + size > self.max_size_bytes or
                   len(self._cache) >= self.max_entries) and self._cache:
                old_key, _ = self._cache.popitem(last=False)
                self._current_size -= self._sizes.pop(old_key, 0)

            # Store
            if key in self._cache:
                self._current_size -= self._sizes.get(key, 0)

            self._cache[key] = value
            self._sizes[key] = size
            self._current_size += size

    def invalidate(self, key: str) -> bool:
        """Invalidate specific key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._current_size -= self._sizes.pop(key, 0)
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern."""
        with self._lock:
            to_remove = [
                k for k in self._cache.keys()
                if fnmatch.fnmatch(k, pattern)
            ]
            for key in to_remove:
                del self._cache[key]
                self._current_size -= self._sizes.pop(key, 0)
            return len(to_remove)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "size_bytes": self._current_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }

