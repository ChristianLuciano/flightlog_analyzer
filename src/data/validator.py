"""
Data validation functionality.

Provides comprehensive validation of flight log data including
timestamp validation, coordinate validation, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

from ..core.types import FlightDataDict, EventData
from ..core.constants import DEFAULT_TIMESTAMP_COLUMN
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


# Event severity ordering
SEVERITY_ORDER = {
    'debug': 0,
    'info': 1,
    'warning': 2,
    'error': 3,
    'critical': 4,
}

# Default event colors by severity
EVENT_COLORS = {
    'debug': '#888888',
    'info': '#3498db',
    'warning': '#f39c12',
    'error': '#e74c3c',
    'critical': '#9b59b6',
}


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    completeness: float
    gap_count: int
    gaps: List[Tuple[float, float]]
    outlier_count: int
    invalid_count: int
    sampling_rate_stats: Dict[str, float]
    timestamp_issues: List[str]


class DataValidator:
    """
    Validates flight log data.

    Performs structural validation, timestamp validation,
    coordinate validation, and data quality analysis.
    """

    def __init__(self, timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN):
        """
        Initialize DataValidator.

        Args:
            timestamp_column: Name of timestamp column.
        """
        self.timestamp_column = timestamp_column

    def validate(
        self,
        data: FlightDataDict,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate entire data structure.

        Args:
            data: Hierarchical data dictionary.
            strict: If True, warnings become errors.

        Returns:
            ValidationResult with errors and warnings.
        """
        result = ValidationResult(is_valid=True)

        # Validate structure
        self._validate_structure(data, "", result)

        # Check for at least one DataFrame
        df_count = self._count_dataframes(data)
        if df_count == 0:
            result.errors.append("No DataFrames found in data structure")
            result.is_valid = False

        result.info["dataframe_count"] = df_count

        # In strict mode, warnings become errors
        if strict and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.is_valid = False

        return result

    def _validate_structure(
        self,
        data: Any,
        path: str,
        result: ValidationResult
    ) -> None:
        """Recursively validate data structure."""
        if isinstance(data, pd.DataFrame):
            self._validate_dataframe(data, path, result)
        elif isinstance(data, dict):
            if not data:
                result.warnings.append(f"Empty dictionary at '{path}'")
            for key, value in data.items():
                if not isinstance(key, str):
                    result.errors.append(
                        f"Non-string key '{key}' at '{path}'"
                    )
                    result.is_valid = False
                new_path = f"{path}.{key}" if path else key
                self._validate_structure(value, new_path, result)
        else:
            result.errors.append(
                f"Invalid type at '{path}': {type(data).__name__}"
            )
            result.is_valid = False

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        result: ValidationResult
    ) -> None:
        """Validate a single DataFrame."""
        # Check for empty DataFrame
        if df.empty:
            result.warnings.append(f"Empty DataFrame at '{path}'")
            return

        # Check for timestamp column
        if self.timestamp_column not in df.columns:
            result.errors.append(
                f"Missing timestamp column '{self.timestamp_column}' "
                f"at '{path}'"
            )
            result.is_valid = False
            return

        # Validate timestamps
        ts_result = self.validate_timestamps(df[self.timestamp_column])
        if not ts_result.is_valid:
            for error in ts_result.errors:
                result.errors.append(f"{path}: {error}")
            result.is_valid = False
        result.warnings.extend(
            f"{path}: {w}" for w in ts_result.warnings
        )

    def validate_timestamps(self, timestamps: pd.Series) -> ValidationResult:
        """
        Validate timestamp series.

        Args:
            timestamps: Series of timestamps.

        Returns:
            ValidationResult for timestamp validation.
        """
        result = ValidationResult(is_valid=True)

        # Check for null values
        null_count = timestamps.isna().sum()
        if null_count > 0:
            result.errors.append(f"Found {null_count} null timestamps")
            result.is_valid = False

        # Check monotonicity
        if not timestamps.dropna().is_monotonic_increasing:
            diffs = timestamps.diff()
            negative_count = (diffs < 0).sum()
            if negative_count > 0:
                result.warnings.append(
                    f"Non-monotonic timestamps: {negative_count} reversals"
                )

        # Check for duplicates
        dup_count = timestamps.duplicated().sum()
        if dup_count > 0:
            result.warnings.append(f"Found {dup_count} duplicate timestamps")

        # Check for large gaps
        if len(timestamps) > 1:
            diffs = timestamps.diff().dropna()
            median_diff = diffs.median()
            large_gaps = diffs[diffs > 10 * median_diff]
            if len(large_gaps) > 0:
                result.warnings.append(
                    f"Found {len(large_gaps)} large timestamp gaps"
                )

        return result

    def validate_coordinates(
        self,
        lat: pd.Series,
        lon: pd.Series
    ) -> ValidationResult:
        """
        Validate geographic coordinates.

        Args:
            lat: Latitude series.
            lon: Longitude series.

        Returns:
            ValidationResult for coordinate validation.
        """
        result = ValidationResult(is_valid=True)

        # Check latitude range
        lat_invalid = (lat < -90) | (lat > 90)
        if lat_invalid.any():
            count = lat_invalid.sum()
            result.errors.append(f"Found {count} invalid latitude values")
            result.is_valid = False

        # Check longitude range
        lon_invalid = (lon < -180) | (lon > 180)
        if lon_invalid.any():
            count = lon_invalid.sum()
            result.errors.append(f"Found {count} invalid longitude values")
            result.is_valid = False

        # Check for null values
        null_count = lat.isna().sum() + lon.isna().sum()
        if null_count > 0:
            result.warnings.append(f"Found {null_count} null coordinate values")

        # Check for zero values (might indicate invalid data)
        zero_count = ((lat == 0) & (lon == 0)).sum()
        if zero_count > 0:
            result.warnings.append(
                f"Found {zero_count} (0,0) coordinates (may be invalid)"
            )

        return result

    def validate_event_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate event log DataFrame.

        Args:
            df: Event DataFrame.

        Returns:
            ValidationResult for event validation.
        """
        result = ValidationResult(is_valid=True)

        required_columns = [self.timestamp_column, "event_type", "description"]

        for col in required_columns:
            if col not in df.columns:
                result.errors.append(f"Missing required column: {col}")
                result.is_valid = False

        return result

    def analyze_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Analyze data quality of a DataFrame.

        Args:
            df: DataFrame to analyze.

        Returns:
            DataQualityReport with quality metrics.
        """
        # Completeness
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0

        # Gap detection
        gaps = []
        gap_count = 0
        if self.timestamp_column in df.columns:
            ts = df[self.timestamp_column]
            if len(ts) > 1:
                diffs = ts.diff().dropna()
                median_diff = diffs.median()
                gap_mask = diffs > 5 * median_diff
                gap_count = gap_mask.sum()

                gap_indices = diffs[gap_mask].index
                for idx in gap_indices:
                    prev_idx = df.index.get_loc(idx) - 1
                    gaps.append((
                        float(ts.iloc[prev_idx]),
                        float(ts.iloc[df.index.get_loc(idx)])
                    ))

        # Outlier detection (simple z-score method)
        outlier_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != self.timestamp_column:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count += (z_scores > 3).sum()

        # Invalid value count
        invalid_count = df.isna().sum().sum()

        # Sampling rate statistics
        sampling_stats = {}
        if self.timestamp_column in df.columns and len(df) > 1:
            ts = df[self.timestamp_column]
            diffs = ts.diff().dropna()
            sampling_stats = {
                "mean_dt": float(diffs.mean()),
                "std_dt": float(diffs.std()),
                "min_dt": float(diffs.min()),
                "max_dt": float(diffs.max()),
                "estimated_rate": 1.0 / float(diffs.median()),
            }

        # Timestamp issues
        ts_issues = []
        if self.timestamp_column in df.columns:
            ts = df[self.timestamp_column]
            if ts.isna().any():
                ts_issues.append("Contains null timestamps")
            if not ts.is_monotonic_increasing:
                ts_issues.append("Non-monotonic timestamps")
            if ts.duplicated().any():
                ts_issues.append("Contains duplicate timestamps")

        return DataQualityReport(
            completeness=completeness,
            gap_count=gap_count,
            gaps=gaps,
            outlier_count=outlier_count,
            invalid_count=invalid_count,
            sampling_rate_stats=sampling_stats,
            timestamp_issues=ts_issues
        )

    def _count_dataframes(self, data: Any) -> int:
        """Count total number of DataFrames."""
        if isinstance(data, pd.DataFrame):
            return 1
        elif isinstance(data, dict):
            return sum(self._count_dataframes(v) for v in data.values())
        return 0

    def is_event_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame appears to be an event log.

        Args:
            df: DataFrame to check.

        Returns:
            True if DataFrame looks like an event log.
        """
        event_indicators = ["event_type", "event", "type", "description", "message"]
        columns_lower = [c.lower() for c in df.columns]

        match_count = sum(
            1 for indicator in event_indicators
            if any(indicator in col for col in columns_lower)
        )

        return match_count >= 2

    def detect_gps_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame contains GPS data.

        Args:
            df: DataFrame to check.

        Returns:
            True if DataFrame contains lat/lon columns.
        """
        columns_lower = [c.lower() for c in df.columns]
        lat_indicators = ["lat", "latitude"]
        lon_indicators = ["lon", "lng", "longitude"]

        has_lat = any(
            any(ind in col for ind in lat_indicators)
            for col in columns_lower
        )
        has_lon = any(
            any(ind in col for ind in lon_indicators)
            for col in columns_lower
        )

        return has_lat and has_lon


class EventDetector:
    """
    Detects and processes event DataFrames.
    
    Provides automatic identification of event logs, filtering,
    searching, and annotation generation.
    """
    
    def __init__(
        self,
        timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
        event_identifiers: Optional[List[str]] = None
    ):
        """
        Initialize EventDetector.
        
        Args:
            timestamp_column: Name of timestamp column.
            event_identifiers: Names that identify event DataFrames.
        """
        self.timestamp_column = timestamp_column
        self.event_identifiers = event_identifiers or [
            'events', 'event', 'logs', 'log', 'alerts', 'messages'
        ]
    
    def is_event_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame is an event log.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            True if has required event columns.
        """
        required = {self.timestamp_column, 'event_type', 'description'}
        columns = set(df.columns)
        
        return required.issubset(columns)
    
    def is_event_by_name(self, name: str) -> bool:
        """
        Check if name matches event identifiers.
        
        Args:
            name: DataFrame name.
            
        Returns:
            True if name matches event patterns.
        """
        name_lower = name.lower()
        return any(eid.lower() in name_lower for eid in self.event_identifiers)
    
    def get_event_fields(self, df: pd.DataFrame) -> Set[str]:
        """
        Get optional event fields present.
        
        Args:
            df: Event DataFrame.
            
        Returns:
            Set of optional field names found.
        """
        optional_fields = {'severity', 'category', 'duration', 'metadata'}
        return optional_fields & set(df.columns)
    
    def find_event_dataframes(
        self,
        data: Dict[str, Any],
        path: str = ""
    ) -> List[str]:
        """
        Find all event DataFrames in hierarchy.
        
        Args:
            data: Hierarchical data dict.
            path: Current path.
            
        Returns:
            List of paths to event DataFrames.
        """
        event_paths = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, pd.DataFrame):
                if self.is_event_dataframe(value) or self.is_event_by_name(key):
                    event_paths.append(current_path)
            elif isinstance(value, dict):
                event_paths.extend(self.find_event_dataframes(value, current_path))
        
        return event_paths
    
    def filter_events(
        self,
        df: pd.DataFrame,
        event_types: Optional[List[str]] = None,
        min_severity: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter events by criteria.
        
        Args:
            df: Event DataFrame.
            event_types: Filter by these event types.
            min_severity: Minimum severity level.
            start_time: Start of time range.
            end_time: End of time range.
            
        Returns:
            Filtered DataFrame.
        """
        result = df.copy()
        
        if event_types:
            result = result[result['event_type'].isin(event_types)]
        
        if min_severity and 'severity' in result.columns:
            min_level = SEVERITY_ORDER.get(min_severity.lower(), 0)
            result = result[
                result['severity'].str.lower().map(
                    lambda s: SEVERITY_ORDER.get(s, 0) >= min_level
                )
            ]
        
        if start_time is not None:
            result = result[result[self.timestamp_column] >= start_time]
        
        if end_time is not None:
            result = result[result[self.timestamp_column] <= end_time]
        
        return result
    
    def search_events(
        self,
        df: pd.DataFrame,
        query: str,
        case_sensitive: bool = False
    ) -> pd.DataFrame:
        """
        Search events by description text.
        
        Args:
            df: Event DataFrame.
            query: Search query.
            case_sensitive: Whether search is case sensitive.
            
        Returns:
            Matching events.
        """
        if not case_sensitive:
            mask = df['description'].str.lower().str.contains(
                query.lower(), na=False
            )
        else:
            mask = df['description'].str.contains(query, na=False)
        
        return df[mask]
    
    def get_event_markers(
        self,
        df: pd.DataFrame,
        color_by: str = 'event_type'
    ) -> List[Dict[str, Any]]:
        """
        Generate markers for plot overlay.
        
        Args:
            df: Event DataFrame.
            color_by: Field to color code by ('event_type' or 'severity').
            
        Returns:
            List of marker dictionaries.
        """
        markers = []
        
        for _, row in df.iterrows():
            color = self._get_event_color(row, color_by)
            
            markers.append({
                'timestamp': row[self.timestamp_column],
                'label': row.get('event_type', 'event'),
                'description': row.get('description', ''),
                'color': color,
                'severity': row.get('severity', 'info'),
            })
        
        return markers
    
    def _get_event_color(
        self,
        row: pd.Series,
        color_by: str
    ) -> str:
        """Get color for an event."""
        if color_by == 'severity' and 'severity' in row.index:
            severity = str(row['severity']).lower()
            return EVENT_COLORS.get(severity, EVENT_COLORS['info'])
        
        # Color by event type - use hash for consistent colors
        event_type = str(row.get('event_type', ''))
        hash_val = hash(event_type)
        
        colors = list(EVENT_COLORS.values())
        return colors[hash_val % len(colors)]
    
    def get_next_event(
        self,
        df: pd.DataFrame,
        current_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get next event after current time.
        
        Args:
            df: Event DataFrame.
            current_time: Current timestamp.
            
        Returns:
            Next event dict or None.
        """
        future = df[df[self.timestamp_column] > current_time]
        
        if future.empty:
            return None
        
        next_row = future.iloc[0]
        return next_row.to_dict()
    
    def get_previous_event(
        self,
        df: pd.DataFrame,
        current_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get previous event before current time.
        
        Args:
            df: Event DataFrame.
            current_time: Current timestamp.
            
        Returns:
            Previous event dict or None.
        """
        past = df[df[self.timestamp_column] < current_time]
        
        if past.empty:
            return None
        
        prev_row = past.iloc[-1]
        return prev_row.to_dict()
    
    def validate_events(self, df: pd.DataFrame) -> List[str]:
        """
        Validate event DataFrame.
        
        Args:
            df: Event DataFrame.
            
        Returns:
            List of issues found.
        """
        issues = []
        
        # Check for duplicates
        if df.duplicated(
            subset=[self.timestamp_column, 'event_type']
        ).any():
            issues.append("Duplicate events found at same timestamp")
        
        # Check timestamp ordering
        if not df[self.timestamp_column].is_monotonic_increasing:
            issues.append("Events not in chronological order")
        
        # Check for null descriptions
        if df['description'].isna().any():
            issues.append("Some events have null descriptions")
        
        return issues

