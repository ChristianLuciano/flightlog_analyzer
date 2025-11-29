"""
Main Dash application factory.

Creates and configures the Dash application instance with all
required components, callbacks, and middleware.
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from typing import Optional, Dict, Any

from .constants import APP_TITLE, DEBUG_MODE
from ..config.settings import Settings


def create_app(
    settings: Optional[Settings] = None,
    flight_data: Optional[Dict[str, Any]] = None
) -> Dash:
    """
    Create and configure the Dash application.

    Args:
        settings: Application settings. If None, uses defaults.
        flight_data: Initial flight data dictionary containing DataFrames.

    Returns:
        Configured Dash application instance.
    """
    if settings is None:
        settings = Settings()

    # External stylesheets
    external_stylesheets = [
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ]

    # Initialize Dash app with Bootstrap
    app = Dash(
        __name__,
        title=APP_TITLE,
        update_title="Loading...",
        suppress_callback_exceptions=True,
        assets_folder="../assets",
        prevent_initial_callbacks="initial_duplicate",
        external_stylesheets=external_stylesheets,
    )

    # Store settings and data in app
    app.settings = settings
    app.flight_data = flight_data

    # Setup layout (imported here to avoid circular imports)
    from ..ui.app_layout import create_layout
    app.layout = create_layout(app)

    # Register callbacks
    from ..ui.callbacks import register_callbacks
    register_callbacks(app)

    return app


def run_app(
    app: Dash,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = DEBUG_MODE
) -> None:
    """
    Run the Dash application server.

    Args:
        app: The Dash application instance.
        host: Host address to bind to.
        port: Port number to listen on.
        debug: Enable debug mode.
    """
    app.run(host=host, port=port, debug=debug)

