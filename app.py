"""
Flight Log Analysis Dashboard - Main Entry Point

Run this file to start the dashboard application.
"""

from src.core.app import create_app, run_app
from src.config.settings import Settings


def main():
    """Main entry point for the application."""
    settings = Settings()
    app = create_app(settings=settings)

    print("=" * 50)
    print("Flight Log Analysis Dashboard")
    print("=" * 50)
    print(f"Starting server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    run_app(app, debug=True)


if __name__ == '__main__':
    main()

