"""
Geographic format export.

Provides export to KML, GeoJSON, GPX, and Shapefile formats.

Covers requirements:
- REQ-EXP-006: Export flight path as KML/KMZ
- REQ-EXP-007: Export flight path as GeoJSON
- REQ-EXP-008: Export flight path as GPX
- REQ-EXP-009: Export flight path as Shapefile
- REQ-GEO-080: Export map as image
- REQ-GEO-081 to REQ-GEO-084: Geographic exports
"""

import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
import logging

from ..core.exceptions import ExportError

logger = logging.getLogger(__name__)


def export_kml(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    name: str = "Flight Path",
    description: str = ""
) -> str:
    """
    Export flight path to KML format.

    Args:
        lat, lon: Coordinate arrays.
        alt: Optional altitude array.
        path: Output file path (if None, returns string).
        name: Path name.
        description: Path description.

    Returns:
        KML string if path is None.
    """
    coordinates = []
    for i in range(len(lat)):
        if alt is not None:
            coordinates.append(f"{lon[i]},{lat[i]},{alt[i]}")
        else:
            coordinates.append(f"{lon[i]},{lat[i]},0")

    coord_str = " ".join(coordinates)

    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <description>{description}</description>
    <Style id="flightPath">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>{name}</name>
      <styleUrl>#flightPath</styleUrl>
      <LineString>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>{coord_str}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''

    if path:
        Path(path).write_text(kml)
        logger.info(f"Exported KML to {path}")

    return kml


def export_geojson(
    lat: np.ndarray,
    lon: np.ndarray,
    properties: Optional[Dict[str, Any]] = None,
    path: Union[str, Path] = None
) -> str:
    """
    Export flight path to GeoJSON format.

    Args:
        lat, lon: Coordinate arrays.
        properties: Optional properties dict.
        path: Output file path.

    Returns:
        GeoJSON string if path is None.
    """
    coordinates = [[float(lon[i]), float(lat[i])] for i in range(len(lat))]

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": properties or {"name": "Flight Path"}
        }]
    }

    json_str = json.dumps(geojson, indent=2)

    if path:
        Path(path).write_text(json_str)
        logger.info(f"Exported GeoJSON to {path}")

    return json_str


def export_gpx(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    name: str = "Flight Path"
) -> str:
    """
    Export flight path to GPX format.

    Args:
        lat, lon: Coordinate arrays.
        alt: Optional altitude array.
        timestamps: Optional timestamp array.
        path: Output file path.
        name: Track name.

    Returns:
        GPX string if path is None.
    """
    from datetime import datetime

    trkpts = []
    for i in range(len(lat)):
        pt = f'      <trkpt lat="{lat[i]}" lon="{lon[i]}">'
        if alt is not None:
            pt += f'\n        <ele>{alt[i]}</ele>'
        if timestamps is not None:
            dt = datetime.fromtimestamp(timestamps[i]).isoformat()
            pt += f'\n        <time>{dt}</time>'
        pt += '\n      </trkpt>'
        trkpts.append(pt)

    gpx = f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="FlightLogDashboard">
  <trk>
    <name>{name}</name>
    <trkseg>
{chr(10).join(trkpts)}
    </trkseg>
  </trk>
</gpx>'''

    if path:
        Path(path).write_text(gpx)
        logger.info(f"Exported GPX to {path}")

    return gpx


def export_shapefile(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    properties: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export flight path to Shapefile format (for GIS).
    
    REQ-EXP-009: Export path as Shapefile
    
    Args:
        lat, lon: Coordinate arrays.
        alt: Optional altitude array.
        path: Output path (without extension, creates .shp, .shx, .dbf).
        properties: Optional attributes dict.
    """
    try:
        import shapefile
    except ImportError:
        raise ExportError("pyshp not installed. Install with: pip install pyshp")
    
    if path is None:
        raise ExportError("Path is required for Shapefile export")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove extension if present
    if path.suffix.lower() in ['.shp', '.shx', '.dbf']:
        path = path.with_suffix('')
    
    try:
        # Create shapefile writer
        w = shapefile.Writer(str(path))
        
        # Define fields
        w.field('name', 'C', 40)
        w.field('length_m', 'N', 12, 2)
        
        # Add properties as fields
        if properties:
            for key, value in properties.items():
                if isinstance(value, (int, float)):
                    w.field(key, 'N', 12, 2)
                else:
                    w.field(key, 'C', 100)
        
        # Create polyline (with Z if altitude available)
        if alt is not None:
            points = [[float(lon[i]), float(lat[i]), float(alt[i])] for i in range(len(lat))]
            w.linez([points])
        else:
            points = [[float(lon[i]), float(lat[i])] for i in range(len(lat))]
            w.line([points])
        
        # Calculate length
        from ..utils.geo import haversine_distance
        total_length = 0.0
        for i in range(1, len(lat)):
            total_length += haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i])
        
        # Add record
        record = ['Flight Path', total_length]
        if properties:
            for value in properties.values():
                record.append(value)
        w.record(*record)
        
        w.close()
        
        # Create .prj file (WGS84)
        prj_content = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        Path(str(path) + '.prj').write_text(prj_content)
        
        logger.info(f"Exported Shapefile to {path}")
        
    except Exception as e:
        raise ExportError(f"Failed to export Shapefile: {e}")


def export_kmz(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    name: str = "Flight Path",
    description: str = ""
) -> None:
    """
    Export flight path to KMZ format (compressed KML).
    
    REQ-EXP-006: Export flight path as KML/KMZ
    
    Args:
        lat, lon: Coordinate arrays.
        alt: Optional altitude array.
        path: Output file path.
        name: Path name.
        description: Path description.
    """
    import zipfile
    
    if path is None:
        raise ExportError("Path is required for KMZ export")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate KML content
    kml_content = export_kml(lat, lon, alt, path=None, name=name, description=description)
    
    try:
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('doc.kml', kml_content)
        
        logger.info(f"Exported KMZ to {path}")
        
    except Exception as e:
        raise ExportError(f"Failed to export KMZ: {e}")


def export_flight_path(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    format: str = 'kml',
    **kwargs
) -> Optional[str]:
    """
    Export flight path to various formats.
    
    Args:
        lat, lon: Coordinate arrays.
        alt: Optional altitude array.
        timestamps: Optional timestamp array.
        path: Output path.
        format: Output format ('kml', 'kmz', 'geojson', 'gpx', 'shapefile').
        **kwargs: Format-specific options.
    
    Returns:
        String content for formats that support it, None otherwise.
    """
    format = format.lower()
    
    if format == 'kml':
        return export_kml(lat, lon, alt, path, **kwargs)
    elif format == 'kmz':
        export_kmz(lat, lon, alt, path, **kwargs)
        return None
    elif format == 'geojson':
        return export_geojson(lat, lon, kwargs.get('properties'), path)
    elif format == 'gpx':
        return export_gpx(lat, lon, alt, timestamps, path, kwargs.get('name', 'Flight Path'))
    elif format in ['shapefile', 'shp']:
        export_shapefile(lat, lon, alt, path, kwargs.get('properties'))
        return None
    else:
        raise ExportError(f"Unsupported format: {format}")

