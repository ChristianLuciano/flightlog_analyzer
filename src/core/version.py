"""
Version management for Flight Log Analysis Dashboard.

Implements REQ-DEPLOY-007 through REQ-DEPLOY-011 (version tracking and updates).
"""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Version history for migration support
VERSION_HISTORY = [
    "0.1.0",  # Initial release
]

# Configuration schema version
CONFIG_SCHEMA_VERSION = "1.0"

# Update checking URL
UPDATE_CHECK_URL = "https://api.github.com/repos/flight-log/dashboard/releases/latest"


def get_version() -> str:
    """Get current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get version as tuple (major, minor, patch)."""
    return __version_info__


def check_config_compatibility(config_version: str) -> bool:
    """
    Check if a configuration version is compatible with current version.
    
    Args:
        config_version: Version string from saved configuration.
        
    Returns:
        True if compatible, False otherwise.
    """
    if not config_version:
        return True  # Assume old configs without version are compatible
    
    try:
        config_major = int(config_version.split('.')[0])
        current_major = int(CONFIG_SCHEMA_VERSION.split('.')[0])
        return config_major == current_major
    except (ValueError, IndexError):
        return False


def migrate_config(config: dict, from_version: str) -> dict:
    """
    Migrate configuration from older version to current.
    
    Args:
        config: Configuration dictionary.
        from_version: Version the config was saved with.
        
    Returns:
        Migrated configuration dictionary.
    """
    # Add version if missing
    if 'version' not in config:
        config['version'] = CONFIG_SCHEMA_VERSION
    
    # Future migrations would go here
    # if from_version < "1.1":
    #     # Migrate from 1.0 to 1.1
    #     pass
    
    return config


async def check_for_updates() -> dict:
    """
    Check for available updates.
    
    Returns:
        Dictionary with update info or None if up to date.
    """
    # Placeholder - would check GitHub releases or custom update server
    return {
        'current_version': __version__,
        'latest_version': __version__,
        'update_available': False,
        'release_notes': None,
        'download_url': None,
    }

