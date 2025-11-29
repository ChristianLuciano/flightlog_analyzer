"""
Tests for the core application module.

Tests REQ-TECH-001 through REQ-TECH-008: Core application functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.app import create_app, run_app
from src.config.settings import Settings


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_default_settings(self):
        """Test creating app with default settings."""
        app = create_app()
        
        assert app is not None
        assert hasattr(app, 'settings')
        assert hasattr(app, 'flight_data')

    def test_create_app_with_settings(self):
        """Test creating app with custom settings."""
        settings = Settings(theme='dark')
        app = create_app(settings=settings)
        
        assert app.settings.theme == 'dark'

    def test_create_app_with_flight_data(self, hierarchical_flight_data):
        """Test creating app with flight data."""
        app = create_app(flight_data=hierarchical_flight_data)
        
        assert app.flight_data is not None
        assert 'Sensors' in app.flight_data

    def test_app_has_layout(self):
        """Test that app has a layout."""
        app = create_app()
        
        assert app.layout is not None

    def test_app_title(self):
        """Test that app has correct title."""
        app = create_app()
        
        assert app.title is not None


class TestRunApp:
    """Tests for run_app function."""

    def test_run_app_calls_run(self):
        """Test that run_app calls app.run (not deprecated run_server)."""
        mock_app = MagicMock()
        
        run_app(mock_app, host='localhost', port=8080, debug=False)
        
        # Should call app.run, not app.run_server
        mock_app.run.assert_called_once_with(
            host='localhost',
            port=8080,
            debug=False
        )
        mock_app.run_server.assert_not_called()

    def test_run_app_default_params(self):
        """Test run_app with default parameters."""
        mock_app = MagicMock()
        
        run_app(mock_app)
        
        mock_app.run.assert_called_once()
        call_kwargs = mock_app.run.call_args[1]
        
        assert call_kwargs['host'] == '127.0.0.1'
        assert call_kwargs['port'] == 8050


class TestAppIntegration:
    """Integration tests for app module."""

    def test_app_can_be_created_and_configured(self, hierarchical_flight_data):
        """Test full app creation workflow."""
        settings = Settings(
            theme='aviation',
            timestamp_column='timestamp',
        )
        
        app = create_app(
            settings=settings,
            flight_data=hierarchical_flight_data
        )
        
        assert app is not None
        assert app.settings.theme == 'aviation'
        assert 'Sensors' in app.flight_data

    def test_app_suppress_callback_exceptions(self):
        """Test that callback exceptions are suppressed."""
        app = create_app()
        
        assert app.config.suppress_callback_exceptions is True

