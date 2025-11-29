"""
Configuration schema and validation.

Provides schema definitions and validation for configuration files.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Validation error details."""
    path: str
    message: str
    value: Any = None


# Valid theme values
VALID_THEMES = ['light', 'dark', 'aviation']


class ConfigSchema:
    """
    Schema definition for configuration validation.

    Defines expected structure and types for configuration.
    """

    SCHEMA = {
        'version': {'type': str, 'required': False},
        'theme': {'type': str, 'options': VALID_THEMES},
        'timestamp_column': {'type': str},
        'path_delimiter': {'type': str},
        'tabs': {
            'type': list,
            'items': {
                'id': {'type': str, 'required': False},
                'name': {'type': str, 'required': True},
                'grid': {
                    'type': dict,
                    'properties': {
                        'rows': {'type': int, 'min': 1, 'max': 10},
                        'cols': {'type': int, 'min': 1, 'max': 10},
                    }
                }
            }
        },
        'computed_signals': {
            'type': dict,
            'values': {
                'formula': {'type': str, 'required': True},
                'inputs': {'type': list},
                'unit': {'type': str},
            }
        },
        'map_config': {
            'type': dict,
            'properties': {
                'tile_provider': {'type': str},
                'zoom': {'type': int, 'min': 1, 'max': 20},
            }
        }
    }

    @classmethod
    def validate(cls, config: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[ValidationError]]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate.
            strict: If True, be more strict about validation.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors = []
        cls._validate_dict(config, cls.SCHEMA, '', errors, strict)
        
        # Additional strict validations
        if strict:
            cls._validate_computed_signals(config.get('computed_signals', {}), errors)
        
        return len(errors) == 0, errors

    @classmethod
    def _validate_dict(
        cls,
        data: Dict,
        schema: Dict,
        path: str,
        errors: List[ValidationError],
        strict: bool = False
    ) -> None:
        """Recursively validate dictionary."""
        for key, rules in schema.items():
            full_path = f"{path}.{key}" if path else key
            value = data.get(key)

            # Check required
            if rules.get('required') and value is None:
                errors.append(ValidationError(full_path, "Required field missing"))
                continue

            if value is None:
                continue

            # Check type
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(ValidationError(
                    full_path,
                    f"Expected {expected_type.__name__}, got {type(value).__name__}",
                    value
                ))
                continue

            # Check options
            if 'options' in rules and value not in rules['options']:
                errors.append(ValidationError(
                    full_path,
                    f"Value must be one of {rules['options']}",
                    value
                ))

            # Check min/max
            if 'min' in rules and value < rules['min']:
                errors.append(ValidationError(full_path, f"Value must be >= {rules['min']}", value))
            if 'max' in rules and value > rules['max']:
                errors.append(ValidationError(full_path, f"Value must be <= {rules['max']}", value))

            # Recurse into nested dicts
            if expected_type == dict and 'properties' in rules:
                cls._validate_dict(value, rules['properties'], full_path, errors, strict)

    @classmethod
    def _validate_computed_signals(
        cls,
        computed_signals: Dict[str, Any],
        errors: List[ValidationError]
    ) -> None:
        """Validate computed signals definitions in strict mode."""
        for name, definition in computed_signals.items():
            if not isinstance(definition, dict):
                errors.append(ValidationError(
                    f"computed_signals.{name}",
                    "Computed signal definition must be a dictionary"
                ))
                continue
            
            if 'formula' not in definition:
                errors.append(ValidationError(
                    f"computed_signals.{name}",
                    "Computed signal must have 'formula' field"
                ))


def validate_config(config: Dict[str, Any], strict: bool = False) -> bool:
    """
    Validate configuration.

    Args:
        config: Configuration to validate.
        strict: If True, raise ValueError on validation errors.

    Returns:
        True if valid.

    Raises:
        ValueError: If strict=True and validation fails.
    """
    is_valid, errors = ConfigSchema.validate(config, strict=strict)
    
    if not is_valid and strict:
        messages = [f"{e.path}: {e.message}" for e in errors]
        raise ValueError(f"Configuration validation failed: {messages}")
    
    return True  # Return True even for non-critical issues in non-strict mode


def validate_config_full(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Full validation returning detailed results.

    Returns tuple of (is_valid, error_messages).
    """
    is_valid, errors = ConfigSchema.validate(config)
    messages = [f"{e.path}: {e.message}" for e in errors]
    return is_valid, messages
