"""
Tests for import and export functionality.

Covers requirements:
- REQ-INT-001 to REQ-INT-011: Import from various formats
- REQ-EXP-006 to REQ-EXP-020: Export capabilities
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import os


class TestCSVImport:
    """Tests for CSV import functionality."""
    
    def test_import_csv_basic(self, tmp_path):
        """Test basic CSV import."""
        from src.data.importers import CSVImporter
        
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df.to_csv(csv_path, index=False)
        
        importer = CSVImporter()
        result = importer.import_file(csv_path)
        
        assert 'test' in result
        assert len(result['test']) == 3
        assert 'timestamp' in result['test'].columns
    
    def test_import_csv_column_mapping(self, tmp_path):
        """Test CSV import with column mapping."""
        from src.data.importers import CSVImporter
        
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'time': [1, 2, 3],
            'val': [10, 20, 30]
        })
        df.to_csv(csv_path, index=False)
        
        importer = CSVImporter()
        result = importer.import_file(
            csv_path,
            column_mapping={'time': 'timestamp', 'val': 'value'}
        )
        
        assert 'timestamp' in result['test'].columns
        assert 'value' in result['test'].columns
    
    def test_csv_importer_supported_extensions(self):
        """Test CSV importer supported extensions."""
        from src.data.importers import CSVImporter
        
        importer = CSVImporter()
        assert '.csv' in importer.supported_extensions
        assert '.txt' in importer.supported_extensions
        assert '.tsv' in importer.supported_extensions


class TestExcelImport:
    """Tests for Excel import functionality."""
    
    def test_import_excel_basic(self, tmp_path):
        """Test basic Excel import."""
        from src.data.importers import ExcelImporter
        
        excel_path = tmp_path / "test.xlsx"
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df.to_excel(excel_path, index=False)
        
        importer = ExcelImporter()
        result = importer.import_file(excel_path)
        
        assert len(result) >= 1
    
    def test_excel_importer_supported_extensions(self):
        """Test Excel importer supported extensions."""
        from src.data.importers import ExcelImporter
        
        importer = ExcelImporter()
        assert '.xlsx' in importer.supported_extensions
        assert '.xls' in importer.supported_extensions


class TestMATLABExport:
    """Tests for MATLAB export functionality."""
    
    def test_export_matlab_single_dataframe(self, tmp_path, sample_dataframe):
        """Test MATLAB export with single DataFrame."""
        from src.export.data_export import export_matlab
        
        mat_path = tmp_path / "test.mat"
        export_matlab(sample_dataframe, mat_path)
        
        assert mat_path.exists()
        
        # Verify can be loaded
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        assert 'flight_data' in data
    
    def test_export_matlab_multiple_dataframes(self, tmp_path, sample_flight_data):
        """Test MATLAB export with multiple DataFrames."""
        from src.export.data_export import export_matlab
        
        # Flatten nested structure for export
        flat_data = {}
        for key, value in sample_flight_data.items():
            if isinstance(value, pd.DataFrame):
                flat_data[key] = value
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    if isinstance(v2, pd.DataFrame):
                        flat_data[f"{key}_{k2}"] = v2
        
        mat_path = tmp_path / "multi.mat"
        export_matlab(flat_data, mat_path)
        
        assert mat_path.exists()


class TestJupyterNotebookExport:
    """Tests for Jupyter notebook export."""
    
    def test_export_jupyter_notebook(self, tmp_path, sample_dataframe):
        """Test Jupyter notebook export."""
        from src.export.data_export import export_jupyter_notebook
        
        nb_path = tmp_path / "test.ipynb"
        export_jupyter_notebook(sample_dataframe, nb_path, title="Test Analysis")
        
        assert nb_path.exists()
        
        # Verify valid JSON
        with open(nb_path) as f:
            notebook = json.load(f)
        
        assert 'cells' in notebook
        assert 'nbformat' in notebook
        assert notebook['nbformat'] == 4
    
    def test_export_jupyter_with_plots(self, tmp_path, sample_dataframe):
        """Test Jupyter notebook export with plot code."""
        from src.export.data_export import export_jupyter_notebook
        
        nb_path = tmp_path / "test_plots.ipynb"
        export_jupyter_notebook(
            sample_dataframe,
            nb_path,
            title="Test with Plots",
            include_plots=True
        )
        
        assert nb_path.exists()
        
        with open(nb_path) as f:
            notebook = json.load(f)
        
        # Check for plotly import
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        assert len(code_cells) >= 2  # At least import and data cells


class TestGeoExport:
    """Tests for geographic format exports."""
    
    def test_export_kml(self, tmp_path):
        """Test KML export."""
        from src.export.geo_formats import export_kml
        
        lat = np.array([40.0, 40.1, 40.2])
        lon = np.array([-74.0, -73.9, -73.8])
        alt = np.array([100, 150, 200])
        
        kml_path = tmp_path / "test.kml"
        result = export_kml(lat, lon, alt, kml_path)
        
        assert kml_path.exists()
        assert '<?xml version="1.0"' in result
        assert '<kml' in result
        assert 'coordinates' in result
    
    def test_export_geojson(self, tmp_path):
        """Test GeoJSON export."""
        from src.export.geo_formats import export_geojson
        
        lat = np.array([40.0, 40.1, 40.2])
        lon = np.array([-74.0, -73.9, -73.8])
        
        geojson_path = tmp_path / "test.geojson"
        result = export_geojson(lat, lon, path=geojson_path)
        
        assert geojson_path.exists()
        
        data = json.loads(result)
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 1
        assert data['features'][0]['geometry']['type'] == 'LineString'
    
    def test_export_gpx(self, tmp_path):
        """Test GPX export."""
        from src.export.geo_formats import export_gpx
        
        lat = np.array([40.0, 40.1, 40.2])
        lon = np.array([-74.0, -73.9, -73.8])
        
        gpx_path = tmp_path / "test.gpx"
        result = export_gpx(lat, lon, path=gpx_path)
        
        assert gpx_path.exists()
        assert '<?xml version="1.0"' in result
        assert '<gpx' in result
        assert '<trkpt' in result
    
    def test_export_flight_path_kml(self, tmp_path):
        """Test unified flight path export as KML."""
        from src.export.geo_formats import export_flight_path
        
        lat = np.array([40.0, 40.1, 40.2])
        lon = np.array([-74.0, -73.9, -73.8])
        
        path = tmp_path / "flight.kml"
        export_flight_path(lat, lon, path=path, format='kml')
        
        assert path.exists()
    
    def test_export_flight_path_geojson(self, tmp_path):
        """Test unified flight path export as GeoJSON."""
        from src.export.geo_formats import export_flight_path
        
        lat = np.array([40.0, 40.1, 40.2])
        lon = np.array([-74.0, -73.9, -73.8])
        
        path = tmp_path / "flight.geojson"
        export_flight_path(lat, lon, path=path, format='geojson')
        
        assert path.exists()


class TestAutoImporter:
    """Tests for auto-detection import functionality."""
    
    def test_auto_detect_csv(self, tmp_path):
        """Test auto-detection of CSV format."""
        from src.data.importers import AutoImporter
        
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("timestamp,value\n1,10\n2,20")
        
        importer = AutoImporter()
        detected = importer.detect_format(csv_path)
        
        assert detected is not None
        assert detected.__class__.__name__ == 'CSVImporter'
    
    def test_auto_detect_excel(self, tmp_path):
        """Test auto-detection of Excel format."""
        from src.data.importers import AutoImporter
        
        excel_path = tmp_path / "test.xlsx"
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df.to_excel(excel_path, index=False)
        
        importer = AutoImporter()
        detected = importer.detect_format(excel_path)
        
        assert detected is not None
        assert detected.__class__.__name__ == 'ExcelImporter'
    
    def test_get_supported_formats(self):
        """Test getting supported format list."""
        from src.data.importers import AutoImporter
        
        importer = AutoImporter()
        formats = importer.get_supported_formats()
        
        assert 'CSVImporter' in formats
        assert 'ExcelImporter' in formats
        assert 'MATLABImporter' in formats
    
    def test_import_batch(self, tmp_path):
        """Test batch import of multiple files."""
        from src.data.importers import AutoImporter
        
        # Create multiple CSV files
        for i in range(3):
            path = tmp_path / f"file{i}.csv"
            df = pd.DataFrame({
                'timestamp': [1, 2, 3],
                'value': [i*10, i*20, i*30]
            })
            df.to_csv(path, index=False)
        
        importer = AutoImporter()
        result = importer.import_batch(
            [tmp_path / f"file{i}.csv" for i in range(3)]
        )
        
        assert len(result) == 3


class TestDataExport:
    """Tests for general data export functionality."""
    
    def test_export_csv(self, tmp_path, sample_dataframe):
        """Test CSV export."""
        from src.export.data_export import export_csv
        
        csv_path = tmp_path / "output.csv"
        export_csv(sample_dataframe, csv_path)
        
        assert csv_path.exists()
        
        # Verify contents
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(sample_dataframe)
    
    def test_export_excel(self, tmp_path, sample_dataframe):
        """Test Excel export."""
        from src.export.data_export import export_excel
        
        excel_path = tmp_path / "output.xlsx"
        export_excel(sample_dataframe, excel_path)
        
        assert excel_path.exists()
    
    def test_export_fft_results(self, tmp_path):
        """Test FFT results export."""
        from src.export.data_export import export_fft_results
        
        freqs = np.linspace(0, 100, 50)
        mags = np.random.rand(50)
        phases = np.random.rand(50) * 2 * np.pi
        
        csv_path = tmp_path / "fft.csv"
        export_fft_results(freqs, mags, phases, csv_path, format='csv')
        
        assert csv_path.exists()
        
        loaded = pd.read_csv(csv_path)
        assert 'frequency_hz' in loaded.columns
        assert 'magnitude' in loaded.columns
        assert 'phase_rad' in loaded.columns
    
    def test_export_computed_signals(self, tmp_path):
        """Test computed signal definitions export."""
        from src.export.data_export import export_computed_signals
        
        signals = {
            'total_accel': pd.Series([1, 2, 3])
        }
        definitions = {
            'total_accel': {
                'formula': 'sqrt(accel_x**2 + accel_y**2 + accel_z**2)',
                'inputs': ['accel_x', 'accel_y', 'accel_z'],
                'unit': 'm/s^2',
                'description': 'Total acceleration magnitude'
            }
        }
        
        json_path = tmp_path / "signals.json"
        export_computed_signals(signals, definitions, json_path, format='json')
        
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'computed_signals' in data
        assert 'total_accel' in data['computed_signals']
        assert data['computed_signals']['total_accel']['formula'] == definitions['total_accel']['formula']


class TestMATLABImport:
    """Tests for MATLAB import functionality."""
    
    def test_matlab_import_export_roundtrip(self, tmp_path, sample_dataframe):
        """Test MATLAB import/export roundtrip."""
        from src.export.data_export import export_matlab
        from src.data.importers import MATLABImporter
        
        # Export
        mat_path = tmp_path / "roundtrip.mat"
        export_matlab(sample_dataframe, mat_path, variable_name='data')
        
        # Import
        importer = MATLABImporter()
        result = importer.import_file(mat_path)
        
        assert 'data' in result
        # Note: column names may be modified for MATLAB compatibility


class TestImportConvenienceFunction:
    """Tests for convenience import function."""
    
    def test_import_flight_data_auto(self, tmp_path):
        """Test import_flight_data with auto-detection."""
        from src.data.importers import import_flight_data
        
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'altitude': [100, 150, 200]
        })
        df.to_csv(csv_path, index=False)
        
        result = import_flight_data(csv_path)
        
        assert len(result) == 1
        assert 'test' in result
    
    def test_import_flight_data_forced_format(self, tmp_path):
        """Test import_flight_data with forced format."""
        from src.data.importers import import_flight_data
        
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'altitude': [100, 150, 200]
        })
        df.to_csv(csv_path, index=False)
        
        result = import_flight_data(csv_path, format='csv')
        
        assert len(result) == 1


class TestExportIntegration:
    """Integration tests for export functionality."""
    
    def test_full_export_workflow(self, tmp_path, sample_flight_data):
        """Test a complete export workflow."""
        from src.export.data_export import export_csv, export_excel, export_jupyter_notebook
        from src.export.geo_formats import export_flight_path
        
        # Export CSV
        csv_path = tmp_path / "data.csv"
        if 'Sensors' in sample_flight_data and 'GPS' in sample_flight_data['Sensors']:
            gps_data = sample_flight_data['Sensors']['GPS']
            export_csv(gps_data, csv_path)
            assert csv_path.exists()
        
        # Export Jupyter notebook
        nb_path = tmp_path / "analysis.ipynb"
        export_jupyter_notebook(
            sample_flight_data.get('Sensors', {}).get('GPS', pd.DataFrame({'a': [1]})),
            nb_path
        )
        assert nb_path.exists()

