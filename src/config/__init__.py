"""
Configuration management module.

Provides settings, schema validation, and configuration file handling.
"""

from .settings import Settings
from .schema import ConfigSchema, validate_config
from .loader import ConfigLoader, save_config, load_config

