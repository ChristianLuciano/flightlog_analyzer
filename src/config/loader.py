"""
Configuration file loading and saving.

Supports JSON and YAML formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import copy

from .schema import validate_config
from ..core.exceptions import ConfigError, ConfigValidationError

logger = logging.getLogger(__name__)


# Default templates
DEFAULT_TEMPLATES = {
    'basic': {
        'tabs': [
            {'name': 'Overview', 'plots': []},
        ],
    },
    'flight_analysis': {
        'tabs': [
            {'name': 'Overview', 'plots': []},
            {'name': 'GPS', 'plots': []},
            {'name': 'IMU', 'plots': []},
            {'name': 'Battery', 'plots': []},
        ],
    },
}


class ConfigLoader:
    """
    Loads and saves configuration files.

    Supports JSON and YAML formats with validation.
    """

    def __init__(self, validate: bool = True):
        self._validate = validate

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            path: Path to configuration file.

        Returns:
            Configuration dictionary.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        suffix = path.suffix.lower()

        try:
            if suffix == '.json':
                config = self._load_json(path)
            elif suffix in ['.yaml', '.yml']:
                config = self._load_yaml(path)
            else:
                raise ConfigError(f"Unsupported format: {suffix}")
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")

        if self._validate:
            if not validate_config(config, strict=False):
                raise ConfigValidationError("Invalid config")

        return config

    def save(
        self,
        config: Dict[str, Any],
        path: Union[str, Path],
        format: Optional[str] = None
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary.
            path: Output path.
            format: Output format ('json' or 'yaml'). Auto-detected from path if None.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format from extension
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ['.yaml', '.yml']:
                format = 'yaml'
            else:
                format = 'json'

        try:
            if format == 'json':
                self._save_json(config, path)
            elif format in ['yaml', 'yml']:
                self._save_yaml(config, path)
            else:
                raise ConfigError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigError(f"Failed to save config: {e}")

        logger.info(f"Saved config to {path}")

    def get_version(self, config: Dict[str, Any]) -> str:
        """
        Get version from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Version string or '0.0.0' if not present.
        """
        return config.get('version', '0.0.0')

    def merge_configs(
        self,
        original: Dict[str, Any],
        update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge update into original config.

        Args:
            original: Original configuration.
            update: Updates to apply.

        Returns:
            Merged configuration.
        """
        result = copy.deepcopy(original)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result

    def list_templates(self) -> List[str]:
        """
        List available configuration templates.

        Returns:
            List of template names.
        """
        return list(DEFAULT_TEMPLATES.keys())

    def get_template(self, name: str) -> Dict[str, Any]:
        """
        Get a configuration template by name.

        Args:
            name: Template name.

        Returns:
            Template configuration.
        """
        if name not in DEFAULT_TEMPLATES:
            raise ConfigError(f"Unknown template: {name}")
        
        return copy.deepcopy(DEFAULT_TEMPLATES[name])

    def apply_template(
        self,
        base_config: Dict[str, Any],
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a template to a base configuration.

        Args:
            base_config: Base configuration.
            template: Template to apply.

        Returns:
            Merged configuration.
        """
        return self.merge_configs(base_config, template)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ConfigError("PyYAML not installed. Install with: pip install pyyaml")

    def _save_json(self, config: Dict, path: Path) -> None:
        """Save JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _save_yaml(self, config: Dict, path: Path) -> None:
        """Save YAML file."""
        try:
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            raise ConfigError("PyYAML not installed")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load config."""
    return ConfigLoader().load(path)


def save_config(config: Dict[str, Any], path: Union[str, Path], format: str = 'json') -> None:
    """Convenience function to save config."""
    ConfigLoader(validate=False).save(config, path, format)

