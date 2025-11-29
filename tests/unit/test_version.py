"""
Tests for version module.

Covers src/core/version.py - 100% coverage target.
"""

import pytest
import asyncio
from src.core.version import (
    __version__,
    __version_info__,
    VERSION_HISTORY,
    CONFIG_SCHEMA_VERSION,
    UPDATE_CHECK_URL,
    get_version,
    get_version_info,
    check_config_compatibility,
    migrate_config,
    check_for_updates,
)


class TestVersionInfo:
    """Test version information."""
    
    def test_version_string(self):
        """Test version is a string."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_version_format(self):
        """Test version follows semver format."""
        parts = __version__.split('.')
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
    
    def test_version_info_tuple(self):
        """Test version info is a tuple."""
        assert isinstance(__version_info__, tuple)
        assert len(__version_info__) == 3
        for part in __version_info__:
            assert isinstance(part, int)
    
    def test_version_consistency(self):
        """Test version string matches version tuple."""
        expected = '.'.join(str(v) for v in __version_info__)
        assert __version__ == expected
    
    def test_version_history(self):
        """Test version history is non-empty list."""
        assert isinstance(VERSION_HISTORY, list)
        assert len(VERSION_HISTORY) > 0
        assert __version__ in VERSION_HISTORY
    
    def test_config_schema_version(self):
        """Test config schema version."""
        assert isinstance(CONFIG_SCHEMA_VERSION, str)
        parts = CONFIG_SCHEMA_VERSION.split('.')
        assert len(parts) >= 1
    
    def test_update_check_url(self):
        """Test UPDATE_CHECK_URL is defined."""
        assert isinstance(UPDATE_CHECK_URL, str)
        assert len(UPDATE_CHECK_URL) > 0


class TestGetVersion:
    """Test get_version function."""
    
    def test_get_version_returns_string(self):
        """Test get_version returns a string."""
        version = get_version()
        assert isinstance(version, str)
    
    def test_get_version_matches_module_version(self):
        """Test get_version returns __version__."""
        assert get_version() == __version__


class TestGetVersionInfo:
    """Test get_version_info function."""
    
    def test_get_version_info_returns_tuple(self):
        """Test get_version_info returns a tuple."""
        info = get_version_info()
        assert isinstance(info, tuple)
    
    def test_get_version_info_matches_module(self):
        """Test get_version_info returns __version_info__."""
        assert get_version_info() == __version_info__


class TestConfigCompatibility:
    """Test check_config_compatibility function."""
    
    def test_none_version_compatible(self):
        """Test None version is considered compatible."""
        assert check_config_compatibility(None) is True
    
    def test_empty_version_compatible(self):
        """Test empty string is considered compatible."""
        assert check_config_compatibility('') is True
    
    def test_same_major_version_compatible(self):
        """Test same major version is compatible."""
        current_major = CONFIG_SCHEMA_VERSION.split('.')[0]
        assert check_config_compatibility(f'{current_major}.0') is True
        assert check_config_compatibility(f'{current_major}.5') is True
    
    def test_different_major_version_incompatible(self):
        """Test different major version is incompatible."""
        current_major = int(CONFIG_SCHEMA_VERSION.split('.')[0])
        different_major = current_major + 1
        assert check_config_compatibility(f'{different_major}.0') is False
    
    def test_invalid_version_format(self):
        """Test invalid version format returns False."""
        assert check_config_compatibility('invalid') is False
        assert check_config_compatibility('abc.def') is False
    
    def test_exact_version_compatible(self):
        """Test exact schema version is compatible."""
        assert check_config_compatibility(CONFIG_SCHEMA_VERSION) is True


class TestMigrateConfig:
    """Test migrate_config function."""
    
    def test_adds_version_if_missing(self):
        """Test migration adds version if missing."""
        config = {'setting': 'value'}
        migrated = migrate_config(config, '0.1.0')
        assert 'version' in migrated
        assert migrated['version'] == CONFIG_SCHEMA_VERSION
    
    def test_preserves_existing_settings(self):
        """Test migration preserves existing settings."""
        config = {
            'setting1': 'value1',
            'setting2': 'value2',
            'nested': {'key': 'val'}
        }
        migrated = migrate_config(config.copy(), '0.1.0')
        assert migrated['setting1'] == 'value1'
        assert migrated['setting2'] == 'value2'
        assert migrated['nested']['key'] == 'val'
    
    def test_preserves_existing_version(self):
        """Test migration preserves existing version."""
        config = {'version': '2.0', 'setting': 'value'}
        migrated = migrate_config(config.copy(), '1.0')
        # Should not overwrite existing version
        assert migrated['version'] == '2.0'
    
    def test_empty_config(self):
        """Test migration of empty config."""
        config = {}
        migrated = migrate_config(config, '0.0.1')
        assert 'version' in migrated


class TestCheckForUpdates:
    """Test check_for_updates async function."""
    
    def test_check_for_updates_returns_dict(self):
        """Test check_for_updates returns a dictionary."""
        result = asyncio.run(check_for_updates())
        assert isinstance(result, dict)
    
    def test_check_for_updates_has_required_keys(self):
        """Test check_for_updates returns required keys."""
        result = asyncio.run(check_for_updates())
        assert 'current_version' in result
        assert 'latest_version' in result
        assert 'update_available' in result
    
    def test_check_for_updates_current_version(self):
        """Test check_for_updates returns correct current version."""
        result = asyncio.run(check_for_updates())
        assert result['current_version'] == __version__
    
    def test_check_for_updates_has_download_url(self):
        """Test check_for_updates has download_url key."""
        result = asyncio.run(check_for_updates())
        assert 'download_url' in result
    
    def test_check_for_updates_has_release_notes(self):
        """Test check_for_updates has release_notes key."""
        result = asyncio.run(check_for_updates())
        assert 'release_notes' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
