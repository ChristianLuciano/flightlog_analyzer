"""
Tests to improve importers coverage.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.data.importers import (
    CSVImporter,
    ExcelImporter,
    MATLABImporter,
    MAVLinkImporter,
    PX4Importer,
    ArduPilotImporter,
    AutoImporter,
    import_flight_data,
)


class TestCSVImporter:
    """Test CSV importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = CSVImporter()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = CSVImporter()
        exts = importer.supported_extensions  # Property, not method
        assert '.csv' in exts
    
    def test_can_import_csv(self):
        """Test can_import for CSV."""
        importer = CSVImporter()
        assert importer.can_import(Path('test.csv')) is True
        assert importer.can_import(Path('test.xlsx')) is False


class TestExcelImporter:
    """Test Excel importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = ExcelImporter()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = ExcelImporter()
        exts = importer.supported_extensions
        assert '.xlsx' in exts or '.xls' in exts
    
    def test_can_import_excel(self):
        """Test can_import for Excel."""
        importer = ExcelImporter()
        assert importer.can_import(Path('test.xlsx')) is True


class TestMATLABImporter:
    """Test MATLAB importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = MATLABImporter()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = MATLABImporter()
        exts = importer.supported_extensions
        assert '.mat' in exts
    
    def test_can_import_mat(self):
        """Test can_import for MATLAB."""
        importer = MATLABImporter()
        assert importer.can_import(Path('test.mat')) is True


class TestMAVLinkImporter:
    """Test MAVLink importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = MAVLinkImporter()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = MAVLinkImporter()
        exts = importer.supported_extensions
        assert '.tlog' in exts or '.bin' in exts
    
    def test_can_import_tlog(self):
        """Test can_import for tlog."""
        importer = MAVLinkImporter()
        assert importer.can_import(Path('test.tlog')) is True


class TestPX4Importer:
    """Test PX4 importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = PX4Importer()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = PX4Importer()
        exts = importer.supported_extensions
        assert '.ulg' in exts
    
    def test_can_import_ulg(self):
        """Test can_import for ulg."""
        importer = PX4Importer()
        assert importer.can_import(Path('test.ulg')) is True


class TestArduPilotImporter:
    """Test ArduPilot importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = ArduPilotImporter()
        assert importer is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        importer = ArduPilotImporter()
        exts = importer.supported_extensions
        assert len(exts) > 0
    
    def test_can_import_log(self):
        """Test can_import for log."""
        importer = ArduPilotImporter()
        # Check if it can handle some extension
        exts = importer.supported_extensions
        if exts:
            assert importer.can_import(Path(f'test{exts[0]}')) is True


class TestAutoImporter:
    """Test auto importer."""
    
    def test_init(self):
        """Test initialization."""
        importer = AutoImporter()
        assert importer is not None
    
    def test_detect_format_csv(self):
        """Test detecting CSV format."""
        importer = AutoImporter()
        detected = importer.detect_format('test.csv')
        assert detected is not None
        assert isinstance(detected, CSVImporter)
    
    def test_detect_format_xlsx(self):
        """Test detecting Excel format."""
        importer = AutoImporter()
        detected = importer.detect_format('test.xlsx')
        assert detected is not None
    
    def test_detect_format_mat(self):
        """Test detecting MATLAB format."""
        importer = AutoImporter()
        detected = importer.detect_format('test.mat')
        assert detected is not None
    
    def test_detect_format_tlog(self):
        """Test detecting tlog format."""
        importer = AutoImporter()
        detected = importer.detect_format('test.tlog')
        assert detected is not None
    
    def test_get_supported_formats(self):
        """Test getting all supported formats."""
        importer = AutoImporter()
        formats = importer.get_supported_formats()
        assert isinstance(formats, dict)
        assert len(formats) > 0


class TestImportFlightData:
    """Test import_flight_data function."""
    
    def test_import_nonexistent_file(self):
        """Test importing nonexistent file raises error."""
        with pytest.raises(Exception):
            import_flight_data('definitely_not_a_file.xyz')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

