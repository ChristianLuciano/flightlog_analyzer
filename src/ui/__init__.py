"""
User interface module.

Provides Dash-based UI components, layouts, and callbacks
for the dashboard application.
"""

from .app_layout import create_layout
from .callbacks import register_callbacks
from .state import AppState

