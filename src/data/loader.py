"""
Data loading functionality.

Handles loading and initial processing of flight log data from
hierarchical dictionary structures containing pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

from ..core.types import FlightDataDict, SignalInfo, SignalMetadata
from ..core.exceptions import DataLoadError, InvalidDataStructure, TimestampError
from ..core.constants import DEFAULT_TIMESTAMP_COLUMN, MAX_HIERARCHY_DEPTH

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and processes flight log data.

    Handles validation, metadata extraction, and initial processing
    of hierarchical DataFrame structures.
    """

    def __init__(
        self,
        timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
        validate_timestamps: bool = True
    ):
        """
        Initialize DataLoader.

        Args:
            timestamp_column: Name of the timestamp column in DataFrames.
            validate_timestamps: Whether to validate timestamp monotonicity.
        """
        self.timestamp_column = timestamp_column
        self.validate_timestamps = validate_timestamps
        self._dataframe_cache: Dict[str, pd.DataFrame] = {}
        self._signal_cache: Dict[str, SignalInfo] = {}
        self._metadata: Dict[str, Any] = {}

    def load(self, data: FlightDataDict) -> "DataLoader":
        """
        Load flight data from hierarchical dictionary.

        Args:
            data: Nested dictionary containing pandas DataFrames.

        Returns:
            Self for method chaining.

        Raises:
            InvalidDataStructure: If data structure is invalid.
            TimestampError: If timestamp validation fails.
        """
        logger.info("Loading flight data...")

        # Validate structure
        self._validate_structure(data)

        # Index all DataFrames
        self._index_dataframes(data)

        # Extract metadata
        self._extract_metadata()

        logger.info(
            f"Loaded {len(self._dataframe_cache)} DataFrames "
            f"with {len(self._signal_cache)} signals"
        )

        return self

    def _validate_structure(
        self,
        data: Any,
        path: str = "",
        depth: int = 0
    ) -> None:
        """Recursively validate data structure."""
        if depth > MAX_HIERARCHY_DEPTH:
            raise InvalidDataStructure(
                f"Maximum hierarchy depth ({MAX_HIERARCHY_DEPTH}) exceeded at {path}"
            )

        if isinstance(data, pd.DataFrame):
            self._validate_dataframe(data, path)
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._validate_structure(value, new_path, depth + 1)
        else:
            raise InvalidDataStructure(
                f"Invalid type at {path}: expected DataFrame or dict, "
                f"got {type(data).__name__}"
            )

    def _validate_dataframe(self, df: pd.DataFrame, path: str) -> None:
        """Validate a single DataFrame."""
        if df.empty:
            logger.warning(f"Empty DataFrame at {path}")
            return

        # Check for timestamp column
        if self.timestamp_column not in df.columns:
            raise TimestampError(
                f"Missing timestamp column '{self.timestamp_column}' "
                f"in DataFrame at {path}"
            )

        # Validate timestamp monotonicity
        if self.validate_timestamps:
            timestamps = df[self.timestamp_column]
            if not timestamps.is_monotonic_increasing:
                # Check for non-monotonic but allow equal values
                if not (timestamps.diff().dropna() >= 0).all():
                    logger.warning(
                        f"Non-monotonic timestamps in DataFrame at {path}"
                    )

    def _index_dataframes(
        self,
        data: Any,
        path: str = ""
    ) -> None:
        """Recursively index all DataFrames."""
        if isinstance(data, pd.DataFrame):
            self._dataframe_cache[path] = data
            self._index_signals(data, path)
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._index_dataframes(value, new_path)

    def _index_signals(self, df: pd.DataFrame, df_path: str) -> None:
        """Index all signals in a DataFrame."""
        for column in df.columns:
            if column == self.timestamp_column:
                continue

            signal_path = f"{df_path}.{column}"
            signal_info = SignalInfo(
                name=column,
                path=signal_path,
                dataframe_path=df_path,
                dtype=df[column].dtype,
                shape=(len(df),),
                is_computed=False
            )
            self._signal_cache[signal_path] = signal_info

    def _extract_metadata(self) -> None:
        """Extract metadata from all DataFrames."""
        for path, df in self._dataframe_cache.items():
            self._metadata[path] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
            }

            # Extract time range
            if self.timestamp_column in df.columns:
                timestamps = df[self.timestamp_column]
                self._metadata[path]["time_range"] = {
                    "start": timestamps.iloc[0],
                    "end": timestamps.iloc[-1],
                    "duration": timestamps.iloc[-1] - timestamps.iloc[0],
                }

                # Estimate sampling rate
                if len(timestamps) > 1:
                    dt = np.diff(timestamps.values)
                    self._metadata[path]["sampling_rate"] = 1.0 / np.median(dt)

    def get_dataframe(self, path: str) -> pd.DataFrame:
        """Get DataFrame by path."""
        if path not in self._dataframe_cache:
            raise KeyError(f"DataFrame not found: {path}")
        return self._dataframe_cache[path]

    def get_signal(self, path: str) -> pd.Series:
        """Get signal data by path."""
        if path not in self._signal_cache:
            raise KeyError(f"Signal not found: {path}")

        info = self._signal_cache[path]
        df = self._dataframe_cache[info.dataframe_path]
        return df[info.name]

    def get_signal_with_timestamp(
        self,
        path: str
    ) -> Tuple[pd.Series, pd.Series]:
        """Get signal data with corresponding timestamps."""
        if path not in self._signal_cache:
            raise KeyError(f"Signal not found: {path}")

        info = self._signal_cache[path]
        df = self._dataframe_cache[info.dataframe_path]
        return df[self.timestamp_column], df[info.name]

    def list_dataframes(self) -> List[str]:
        """List all DataFrame paths."""
        return list(self._dataframe_cache.keys())

    def list_signals(self) -> List[SignalInfo]:
        """List all signal information."""
        return list(self._signal_cache.values())

    def get_metadata(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for specific path or all."""
        if path:
            return self._metadata.get(path, {})
        return self._metadata

    def get_time_range(self) -> Tuple[float, float]:
        """Get overall time range across all DataFrames."""
        min_time = float("inf")
        max_time = float("-inf")

        for meta in self._metadata.values():
            if "time_range" in meta:
                min_time = min(min_time, meta["time_range"]["start"])
                max_time = max(max_time, meta["time_range"]["end"])

        return min_time, max_time


def load_flight_data(
    data: FlightDataDict,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
) -> DataLoader:
    """
    Convenience function to load flight data.

    Args:
        data: Hierarchical dictionary containing DataFrames.
        timestamp_column: Name of timestamp column.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(timestamp_column=timestamp_column).load(data)

