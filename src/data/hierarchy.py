"""
Hierarchical data structure navigation.

Provides utilities for navigating and querying the nested dictionary
structure containing DataFrames.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Generator, Union
from dataclasses import dataclass
import fnmatch
import re
import logging

from ..core.types import FlightDataDict, SignalPath
from ..core.constants import DEFAULT_PATH_DELIMITER, MAX_HIERARCHY_DEPTH
from ..core.exceptions import HierarchyError

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Represents a node in the hierarchy tree."""
    name: str
    path: str
    is_dataframe: bool
    children: List["TreeNode"]
    column_count: Optional[int] = None
    row_count: Optional[int] = None


class HierarchyNavigator:
    """
    Navigates hierarchical DataFrame structures.

    Provides methods for path resolution, tree building, searching,
    and traversal of nested dictionary structures.
    """

    def __init__(
        self,
        data: FlightDataDict,
        delimiter: str = DEFAULT_PATH_DELIMITER
    ):
        """
        Initialize HierarchyNavigator.

        Args:
            data: Hierarchical dictionary containing DataFrames.
            delimiter: Path separator character.
        """
        self.data = data
        self.delimiter = delimiter
        self._path_cache: Dict[str, Any] = {}
        self._tree: Optional[TreeNode] = None

        # Build initial cache
        self._build_cache()

    def _build_cache(self) -> None:
        """Build path cache for fast lookups."""
        self._cache_paths(self.data, "")

    def _cache_paths(self, data: Any, path: str) -> None:
        """Recursively cache all paths."""
        self._path_cache[path] = data

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}{self.delimiter}{key}" if path else key
                self._cache_paths(value, new_path)

    def resolve(self, path: SignalPath) -> Any:
        """
        Resolve a path to its value.

        Args:
            path: Dot-separated path string.

        Returns:
            Value at the path (DataFrame, dict, or signal Series).

        Raises:
            HierarchyError: If path cannot be resolved.
        """
        # Check cache first
        if path in self._path_cache:
            return self._path_cache[path]

        # Try to resolve as signal path
        parts = path.split(self.delimiter)
        current = self.data

        for i, part in enumerate(parts):
            if isinstance(current, pd.DataFrame):
                # Last part might be column name
                if part in current.columns:
                    return current[part]
                else:
                    raise HierarchyError(
                        f"Column '{part}' not found in DataFrame"
                    )
            elif isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    raise HierarchyError(
                        f"Key '{part}' not found at "
                        f"'{self.delimiter.join(parts[:i])}'"
                    )
            else:
                raise HierarchyError(
                    f"Cannot navigate into {type(current).__name__}"
                )

        return current

    def get_dataframe(self, path: str) -> pd.DataFrame:
        """
        Get DataFrame at path.

        Args:
            path: Path to DataFrame.

        Returns:
            The DataFrame at the specified path.

        Raises:
            HierarchyError: If path doesn't point to a DataFrame.
        """
        result = self.resolve(path)
        if not isinstance(result, pd.DataFrame):
            raise HierarchyError(f"Path '{path}' is not a DataFrame")
        return result

    def get_signal(self, path: SignalPath) -> pd.Series:
        """
        Get signal (column) at path.

        Args:
            path: Full path including column name.

        Returns:
            The signal as a pandas Series.
        """
        result = self.resolve(path)
        if isinstance(result, pd.Series):
            return result
        raise HierarchyError(f"Path '{path}' is not a signal")

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            self.resolve(path)
            return True
        except HierarchyError:
            return False

    def list_children(self, path: str = "") -> List[str]:
        """
        List immediate children of a path.

        Args:
            path: Parent path (empty for root).

        Returns:
            List of child names.
        """
        target = self.resolve(path) if path else self.data

        if isinstance(target, dict):
            return list(target.keys())
        elif isinstance(target, pd.DataFrame):
            return list(target.columns)
        else:
            return []

    def list_all_paths(
        self,
        include_signals: bool = True
    ) -> List[str]:
        """
        List all paths in the hierarchy.

        Args:
            include_signals: Whether to include DataFrame column paths.

        Returns:
            List of all paths.
        """
        paths = []
        self._collect_paths(self.data, "", paths, include_signals)
        return paths

    def _collect_paths(
        self,
        data: Any,
        path: str,
        paths: List[str],
        include_signals: bool
    ) -> None:
        """Recursively collect all paths."""
        if path:
            paths.append(path)

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}{self.delimiter}{key}" if path else key
                self._collect_paths(value, new_path, paths, include_signals)
        elif isinstance(data, pd.DataFrame) and include_signals:
            for column in data.columns:
                paths.append(f"{path}{self.delimiter}{column}")

    def search(
        self,
        pattern: str,
        search_type: str = "fuzzy"
    ) -> List[str]:
        """
        Search for paths matching pattern.

        Args:
            pattern: Search pattern.
            search_type: One of 'fuzzy', 'glob', 'regex', 'exact'.

        Returns:
            List of matching paths.
        """
        all_paths = self.list_all_paths(include_signals=True)

        if search_type == "exact":
            return [p for p in all_paths if pattern in p]
        elif search_type == "glob":
            return fnmatch.filter(all_paths, pattern)
        elif search_type == "regex":
            regex = re.compile(pattern, re.IGNORECASE)
            return [p for p in all_paths if regex.search(p)]
        else:  # fuzzy
            pattern_lower = pattern.lower()
            return [
                p for p in all_paths
                if pattern_lower in p.lower()
            ]

    def build_tree(self) -> TreeNode:
        """
        Build tree representation of hierarchy.

        Returns:
            Root TreeNode of the hierarchy.
        """
        if self._tree is None:
            self._tree = self._build_tree_node(self.data, "root", "")
        return self._tree

    def _build_tree_node(
        self,
        data: Any,
        name: str,
        path: str
    ) -> TreeNode:
        """Recursively build tree nodes."""
        if isinstance(data, pd.DataFrame):
            return TreeNode(
                name=name,
                path=path,
                is_dataframe=True,
                children=[],
                column_count=len(data.columns),
                row_count=len(data)
            )
        elif isinstance(data, dict):
            children = []
            for key, value in data.items():
                child_path = f"{path}{self.delimiter}{key}" if path else key
                children.append(
                    self._build_tree_node(value, key, child_path)
                )
            return TreeNode(
                name=name,
                path=path,
                is_dataframe=False,
                children=children
            )
        else:
            raise HierarchyError(f"Unexpected type: {type(data)}")

    def walk(
        self,
        path: str = ""
    ) -> Generator[Tuple[str, List[str], List[str]], None, None]:
        """
        Walk the hierarchy tree.

        Similar to os.walk(), yields (path, subdirs, dataframes) tuples.

        Args:
            path: Starting path.

        Yields:
            Tuples of (current_path, subdict_names, dataframe_names).
        """
        target = self.resolve(path) if path else self.data

        if isinstance(target, dict):
            subdirs = []
            dataframes = []

            for key, value in target.items():
                if isinstance(value, dict):
                    subdirs.append(key)
                elif isinstance(value, pd.DataFrame):
                    dataframes.append(key)

            yield path, subdirs, dataframes

            for subdir in subdirs:
                new_path = f"{path}{self.delimiter}{subdir}" if path else subdir
                yield from self.walk(new_path)

    def get_dataframe_paths(self) -> List[str]:
        """Get all paths that point to DataFrames."""
        paths = []
        for path, value in self._path_cache.items():
            if isinstance(value, pd.DataFrame):
                paths.append(path)
        return paths

    def get_parent_path(self, path: str) -> str:
        """Get parent path of given path."""
        parts = path.split(self.delimiter)
        if len(parts) <= 1:
            return ""
        return self.delimiter.join(parts[:-1])


def resolve_path(
    data: FlightDataDict,
    path: SignalPath,
    delimiter: str = DEFAULT_PATH_DELIMITER
) -> Any:
    """
    Convenience function to resolve a path.

    Args:
        data: Hierarchical data dictionary.
        path: Path to resolve.
        delimiter: Path delimiter.

    Returns:
        Value at the path.
    """
    navigator = HierarchyNavigator(data, delimiter)
    return navigator.resolve(path)

