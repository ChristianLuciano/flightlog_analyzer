"""
Tests for event detection and handling.

Tests REQ-DI-021 through REQ-DI-024: Event log support.
Tests REQ-VIS-047 through REQ-VIS-061: Event-based annotations.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.validator import DataValidator, EventDetector


class TestEventDetector:
    """Tests for automatic event detection (REQ-DI-021)."""

    def test_detect_event_dataframe(self, sample_events_data):
        """Test automatic identification of event DataFrames."""
        detector = EventDetector()
        
        is_event_df = detector.is_event_dataframe(sample_events_data)
        
        assert is_event_df is True

    def test_detect_event_required_columns(self):
        """Test event detection requires timestamp, event_type, description (REQ-DI-022)."""
        detector = EventDetector()
        
        # Valid event DataFrame
        valid_events = pd.DataFrame({
            'timestamp': [0, 1, 2],
            'event_type': ['a', 'b', 'c'],
            'description': ['Event A', 'Event B', 'Event C'],
        })
        assert detector.is_event_dataframe(valid_events) is True
        
        # Missing event_type
        missing_type = pd.DataFrame({
            'timestamp': [0, 1],
            'description': ['A', 'B'],
        })
        assert detector.is_event_dataframe(missing_type) is False
        
        # Missing description
        missing_desc = pd.DataFrame({
            'timestamp': [0, 1],
            'event_type': ['a', 'b'],
        })
        assert detector.is_event_dataframe(missing_desc) is False

    def test_detect_optional_event_fields(self):
        """Test detection of optional event fields (REQ-DI-023)."""
        detector = EventDetector()
        
        events_with_optional = pd.DataFrame({
            'timestamp': [0, 1],
            'event_type': ['warning', 'error'],
            'description': ['Low battery', 'GPS lost'],
            'severity': ['warning', 'error'],
            'category': ['power', 'navigation'],
            'duration': [0, 5.0],
            'metadata': ['{"voltage": 11.2}', '{"satellites": 0}'],
        })
        
        assert detector.is_event_dataframe(events_with_optional) is True
        
        fields = detector.get_event_fields(events_with_optional)
        assert 'severity' in fields
        assert 'category' in fields
        assert 'duration' in fields

    def test_user_specified_event_detection(self):
        """Test user-specified event DataFrame identification (REQ-DI-024)."""
        detector = EventDetector(
            event_identifiers=['Events', 'Logs', 'Alerts']
        )
        
        # By name
        assert detector.is_event_by_name('Events') is True
        assert detector.is_event_by_name('Logs') is True
        assert detector.is_event_by_name('GPS_Data') is False

    def test_find_events_in_hierarchy(self, hierarchical_flight_data):
        """Test finding all event DataFrames in hierarchy."""
        detector = EventDetector()
        
        event_paths = detector.find_event_dataframes(hierarchical_flight_data)
        
        assert 'Events' in event_paths


class TestEventFiltering:
    """Tests for event filtering (REQ-VIS-049, REQ-VIS-050)."""

    def test_filter_by_type(self, sample_events_data):
        """Test filtering events by type (REQ-VIS-049)."""
        detector = EventDetector()
        
        filtered = detector.filter_events(
            sample_events_data,
            event_types=['takeoff', 'landing']
        )
        
        assert len(filtered) == 2
        assert 'takeoff' in filtered['event_type'].values
        assert 'landing' in filtered['event_type'].values

    def test_filter_by_severity(self):
        """Test filtering events by severity (REQ-VIS-049)."""
        events = pd.DataFrame({
            'timestamp': [0, 1, 2, 3],
            'event_type': ['a', 'b', 'c', 'd'],
            'description': ['A', 'B', 'C', 'D'],
            'severity': ['info', 'warning', 'error', 'critical'],
        })
        
        detector = EventDetector()
        
        # Filter for warnings and above
        filtered = detector.filter_events(
            events,
            min_severity='warning'
        )
        
        assert len(filtered) == 3  # warning, error, critical
        assert 'info' not in filtered['severity'].values

    def test_search_by_description(self, sample_events_data):
        """Test searching events by text description (REQ-VIS-050)."""
        detector = EventDetector()
        
        results = detector.search_events(
            sample_events_data,
            query='waypoint'
        )
        
        assert len(results) == 1
        assert 'waypoint' in results['description'].values[0].lower()

    def test_filter_by_time_range(self, sample_events_data):
        """Test filtering events by time range."""
        detector = EventDetector()
        
        filtered = detector.filter_events(
            sample_events_data,
            start_time=25,
            end_time=160
        )
        
        # Should include takeoff (30) and waypoint (150)
        assert len(filtered) == 2


class TestEventAnnotations:
    """Tests for event annotations on plots (REQ-VIS-047, REQ-VIS-048)."""

    def test_get_event_markers(self, sample_events_data):
        """Test getting event markers for plot overlay."""
        detector = EventDetector()
        
        markers = detector.get_event_markers(sample_events_data)
        
        assert len(markers) == len(sample_events_data)
        for marker in markers:
            assert 'timestamp' in marker
            assert 'label' in marker
            assert 'color' in marker

    def test_event_color_coding(self):
        """Test color coding by event type/severity (REQ-VIS-052)."""
        events = pd.DataFrame({
            'timestamp': [0, 1, 2],
            'event_type': ['info', 'warning', 'error'],
            'description': ['Info', 'Warning', 'Error'],
            'severity': ['info', 'warning', 'error'],
        })
        
        detector = EventDetector()
        markers = detector.get_event_markers(events, color_by='severity')
        
        # Different severities should have different colors
        colors = [m['color'] for m in markers]
        assert len(set(colors)) == 3  # 3 unique colors

    def test_event_navigation(self, sample_events_data):
        """Test jumping to next/previous event (REQ-VIS-055)."""
        detector = EventDetector()
        
        current_time = 100
        
        next_event = detector.get_next_event(sample_events_data, current_time)
        prev_event = detector.get_previous_event(sample_events_data, current_time)
        
        assert next_event['timestamp'] == 150  # waypoint
        assert prev_event['timestamp'] == 30   # takeoff


class TestEventValidation:
    """Tests for event data validation."""

    def test_validate_event_timestamps(self, sample_events_data):
        """Test validation of event timestamps."""
        detector = EventDetector()
        
        issues = detector.validate_events(sample_events_data)
        
        # Should have no issues with valid data
        assert len(issues) == 0

    def test_detect_duplicate_events(self):
        """Test detection of duplicate events."""
        events = pd.DataFrame({
            'timestamp': [0, 0, 1],  # Duplicate timestamp
            'event_type': ['a', 'a', 'b'],  # Same type at same time
            'description': ['A', 'A duplicate', 'B'],
        })
        
        detector = EventDetector()
        issues = detector.validate_events(events)
        
        assert any('duplicate' in issue.lower() for issue in issues)

    def test_detect_out_of_order_events(self):
        """Test detection of out-of-order events."""
        events = pd.DataFrame({
            'timestamp': [0, 2, 1, 3],  # Out of order
            'event_type': ['a', 'b', 'c', 'd'],
            'description': ['A', 'B', 'C', 'D'],
        })
        
        detector = EventDetector()
        issues = detector.validate_events(events)
        
        assert any('order' in issue.lower() for issue in issues)

