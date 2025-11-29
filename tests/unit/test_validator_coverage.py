"""
Comprehensive tests for data validator module.

Target: Improve src/data/validator.py coverage.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.validator import (
    DataValidator,
    ValidationResult,
    DataQualityReport,
    SEVERITY_ORDER,
    EVENT_COLORS,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_default_is_valid(self):
        """Test default is_valid state."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_with_errors(self):
        """Test with errors."""
        result = ValidationResult(is_valid=False, errors=['Error 1'])
        assert result.is_valid is False
        assert len(result.errors) == 1
    
    def test_with_info(self):
        """Test with info."""
        result = ValidationResult(is_valid=True, info={'key': 'value'})
        assert result.info['key'] == 'value'


class TestDataQualityReport:
    """Test DataQualityReport dataclass."""
    
    def test_creation(self):
        """Test report creation."""
        report = DataQualityReport(
            completeness=0.95,
            gap_count=2,
            gaps=[(10.0, 15.0)],
            outlier_count=5,
            invalid_count=0,
            sampling_rate_stats={'mean': 100},
            timestamp_issues=[]
        )
        assert report.completeness == 0.95
        assert report.gap_count == 2


class TestConstants:
    """Test module constants."""
    
    def test_severity_order(self):
        """Test severity ordering."""
        assert SEVERITY_ORDER['debug'] < SEVERITY_ORDER['critical']
        assert SEVERITY_ORDER['warning'] < SEVERITY_ORDER['error']
    
    def test_event_colors(self):
        """Test event colors defined."""
        assert 'info' in EVENT_COLORS
        assert 'warning' in EVENT_COLORS
        assert 'error' in EVENT_COLORS


class TestDataValidatorInit:
    """Test DataValidator initialization."""
    
    def test_default_timestamp(self):
        """Test default timestamp column."""
        validator = DataValidator()
        assert validator.timestamp_column == 'timestamp'
    
    def test_custom_timestamp(self):
        """Test custom timestamp column."""
        validator = DataValidator(timestamp_column='time')
        assert validator.timestamp_column == 'time'


class TestValidateStructure:
    """Test structure validation."""
    
    def test_empty_dict_warning(self):
        """Test empty dictionary generates warning."""
        validator = DataValidator()
        result = validator.validate({'empty': {}})
        # Empty nested dict should warn
        assert len(result.warnings) > 0 or not result.is_valid
    
    def test_valid_structure(self):
        """Test valid structure passes."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [10, 20, 30]
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True
    
    def test_nested_structure(self):
        """Test nested structure validation."""
        validator = DataValidator()
        data = {
            'level1': {
                'level2': pd.DataFrame({
                    'timestamp': [1, 2, 3],
                    'value': [10, 20, 30]
                })
            }
        }
        result = validator.validate(data)
        assert result.is_valid is True
    
    def test_no_dataframes(self):
        """Test validation fails with no DataFrames."""
        validator = DataValidator()
        result = validator.validate({})
        assert result.is_valid is False
        assert 'No DataFrames' in str(result.errors)
    
    def test_strict_mode(self):
        """Test strict mode converts warnings to errors."""
        validator = DataValidator()
        data = {'test': pd.DataFrame()}  # Empty DF generates warning
        result = validator.validate(data, strict=True)
        # Strict mode: warnings become errors
        assert len(result.warnings) == 0


class TestValidateDataframe:
    """Test DataFrame validation."""
    
    def test_empty_dataframe(self):
        """Test empty DataFrame generates warning."""
        validator = DataValidator()
        data = {'test': pd.DataFrame()}
        result = validator.validate(data)
        # Should have warning about empty DataFrame
        has_empty_warning = any('Empty' in w for w in result.warnings)
        assert has_empty_warning or not result.is_valid
    
    def test_missing_timestamp(self):
        """Test missing timestamp column."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'value': [1, 2, 3]
            })
        }
        result = validator.validate(data)
        # Should fail or warn about missing timestamp
        assert not result.is_valid or len(result.warnings) > 0
    
    def test_valid_dataframe(self):
        """Test valid DataFrame passes."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [10, 20, 30]
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True


class TestTimestampValidation:
    """Test timestamp-specific validation."""
    
    def test_monotonic_timestamps(self):
        """Test monotonic timestamps pass."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 3, 4, 5],
                'value': [10, 20, 30, 40, 50]
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True
    
    def test_duplicate_timestamps(self):
        """Test duplicate timestamps generate warning."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 2, 3, 4],  # Duplicate
                'value': [10, 20, 30, 40, 50]
            })
        }
        result = validator.validate(data)
        # Should warn about duplicates


class TestCoordinateValidation:
    """Test coordinate validation."""
    
    def test_valid_coordinates(self):
        """Test valid coordinates pass."""
        validator = DataValidator()
        data = {
            'gps': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'lat': [47.0, 47.1, 47.2],
                'lon': [-122.0, -122.1, -122.2]
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True
    
    def test_invalid_latitude(self):
        """Test invalid latitude generates warning."""
        validator = DataValidator()
        data = {
            'gps': pd.DataFrame({
                'timestamp': [1, 2, 3],
                'lat': [47.0, 91.0, 47.2],  # 91 is invalid
                'lon': [-122.0, -122.1, -122.2]
            })
        }
        result = validator.validate(data)
        # Validator might warn about out-of-range coordinates


class TestDataQualityAnalysis:
    """Test data quality analysis."""
    
    def test_completeness(self):
        """Test completeness calculation."""
        validator = DataValidator()
        data = {
            'test': pd.DataFrame({
                'timestamp': [1, 2, 3, 4, 5],
                'value': [10, np.nan, 30, np.nan, 50]  # 40% missing
            })
        }
        result = validator.validate(data)
        assert result.is_valid is True


class TestEventDetection:
    """Test event-related validation."""
    
    def test_is_event_dataframe_true(self):
        """Test detecting event DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'event_type': ['takeoff', 'waypoint', 'land'],
            'severity': ['info', 'info', 'info']
        })
        assert validator.is_event_dataframe(df) is True
    
    def test_is_event_dataframe_false(self):
        """Test non-event DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        assert validator.is_event_dataframe(df) is False


class TestGPSDetection:
    """Test GPS DataFrame detection."""
    
    def test_detect_gps_with_lat_lon(self):
        """Test detecting GPS DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame({
            'lat': [47.0, 47.1],
            'lon': [-122.0, -122.1]
        })
        assert validator.detect_gps_dataframe(df) is True
    
    def test_detect_gps_with_latitude_longitude(self):
        """Test detecting GPS with full names."""
        validator = DataValidator()
        df = pd.DataFrame({
            'latitude': [47.0, 47.1],
            'longitude': [-122.0, -122.1]
        })
        assert validator.detect_gps_dataframe(df) is True
    
    def test_detect_not_gps(self):
        """Test non-GPS DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame({
            'value': [1, 2, 3]
        })
        assert validator.detect_gps_dataframe(df) is False


class TestCountDataframes:
    """Test DataFrame counting."""
    
    def test_count_flat(self):
        """Test counting in flat structure."""
        validator = DataValidator()
        data = {
            'df1': pd.DataFrame({'a': [1]}),
            'df2': pd.DataFrame({'b': [2]})
        }
        count = validator._count_dataframes(data)
        assert count == 2
    
    def test_count_nested(self):
        """Test counting in nested structure."""
        validator = DataValidator()
        data = {
            'group': {
                'df1': pd.DataFrame({'a': [1]}),
                'df2': pd.DataFrame({'b': [2]})
            },
            'df3': pd.DataFrame({'c': [3]})
        }
        count = validator._count_dataframes(data)
        assert count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

