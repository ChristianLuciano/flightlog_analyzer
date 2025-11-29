"""
Application settings.

Provides configuration management with defaults and validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import os
import json


@dataclass
class Settings:
    """Application settings with defaults."""

    # Data settings
    timestamp_column: str = "timestamp"
    path_delimiter: str = "."
    max_hierarchy_depth: int = 20

    # Performance settings
    max_display_points: int = 10000
    cache_size_mb: float = 512
    max_computation_time: float = 30.0

    # Map settings
    default_tile_provider: str = "OpenStreetMap"
    default_map_zoom: int = 12

    # UI settings
    theme: str = "light"
    show_events: bool = True
    default_plot_height: int = 400

    # Playback settings
    default_playback_speed: float = 1.0
    playback_frame_rate: int = 30

    # Export settings
    default_export_format: str = "png"
    export_dpi: int = 150

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "Settings":
        """Create from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_env(cls) -> "Settings":
        """Create from environment variables."""
        settings = cls()
        env_mapping = {
            'FLIGHT_LOG_THEME': 'theme',
            'FLIGHT_LOG_CACHE_SIZE': 'cache_size_mb',
            'FLIGHT_LOG_MAX_POINTS': 'max_display_points',
        }
        for env_var, attr in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                field_type = type(getattr(settings, attr))
                setattr(settings, attr, field_type(value))
        return settings

    def update(self, **kwargs) -> "Settings":
        """Update settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance."""
    global _settings
    _settings = settings

