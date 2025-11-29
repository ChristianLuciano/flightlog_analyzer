"""
Signal assignment and mapping system.

Maps raw CSV columns to standard flight dynamics variables
with optional conversion factors.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Source type prefix for signals."""
    MEASUREMENT = "measurement"    # Raw sensor data
    COMMAND = "command"            # Commanded/setpoint values  
    ESTIMATED = "estimated"        # Filtered/estimated values
    REFERENCE = "reference"        # Reference/target values
    SIMULATED = "simulated"        # Simulation output
    RAW = "raw"                    # Unprocessed raw data
    NONE = ""                      # No source prefix


class StandardSignal(Enum):
    """Standard flight dynamics signal categories."""
    
    # Position
    POSITION_LATITUDE = "position.latitude"
    POSITION_LONGITUDE = "position.longitude"
    POSITION_ALTITUDE = "position.altitude"
    POSITION_ALTITUDE_MSL = "position.altitude_msl"
    POSITION_ALTITUDE_AGL = "position.altitude_agl"
    
    # Velocity
    VELOCITY_NORTH = "velocity.north"
    VELOCITY_EAST = "velocity.east"
    VELOCITY_DOWN = "velocity.down"
    VELOCITY_GROUND_SPEED = "velocity.ground_speed"
    VELOCITY_AIRSPEED = "velocity.airspeed"
    VELOCITY_VERTICAL = "velocity.vertical"
    
    # Attitude
    ATTITUDE_ROLL = "attitude.roll"
    ATTITUDE_PITCH = "attitude.pitch"
    ATTITUDE_YAW = "attitude.yaw"
    ATTITUDE_HEADING = "attitude.heading"
    
    # Angular rates
    GYRO_X = "gyro.x"
    GYRO_Y = "gyro.y"
    GYRO_Z = "gyro.z"
    GYRO_ROLL_RATE = "gyro.roll_rate"
    GYRO_PITCH_RATE = "gyro.pitch_rate"
    GYRO_YAW_RATE = "gyro.yaw_rate"
    
    # Accelerations
    ACCEL_X = "accel.x"
    ACCEL_Y = "accel.y"
    ACCEL_Z = "accel.z"
    
    # Magnetometer
    MAG_X = "mag.x"
    MAG_Y = "mag.y"
    MAG_Z = "mag.z"
    
    # Barometer
    BARO_PRESSURE = "baro.pressure"
    BARO_TEMPERATURE = "baro.temperature"
    
    # GPS
    GPS_FIX = "gps.fix_type"
    GPS_SATELLITES = "gps.satellites"
    GPS_HDOP = "gps.hdop"
    GPS_VDOP = "gps.vdop"
    
    # Battery
    BATTERY_VOLTAGE = "battery.voltage"
    BATTERY_CURRENT = "battery.current"
    BATTERY_REMAINING = "battery.remaining"
    
    # RC Input
    RC_CHANNEL_1 = "rc.channel_1"
    RC_CHANNEL_2 = "rc.channel_2"
    RC_CHANNEL_3 = "rc.channel_3"
    RC_CHANNEL_4 = "rc.channel_4"
    RC_THROTTLE = "rc.throttle"
    RC_ROLL = "rc.roll"
    RC_PITCH = "rc.pitch"
    RC_YAW = "rc.yaw"
    
    # Actuator outputs
    MOTOR_1 = "motor.1"
    MOTOR_2 = "motor.2"
    MOTOR_3 = "motor.3"
    MOTOR_4 = "motor.4"
    SERVO_1 = "servo.1"
    SERVO_2 = "servo.2"
    
    # Time
    TIMESTAMP = "time.timestamp"
    TIME_BOOT_MS = "time.boot_ms"
    TIME_UTC = "time.utc"
    
    # Custom
    CUSTOM = "custom"
    
    @staticmethod
    def with_source(signal: 'StandardSignal', source: SignalSource) -> str:
        """
        Get signal path with source prefix.
        
        Example: StandardSignal.with_source(POSITION_LATITUDE, MEASUREMENT)
                 returns "measurement.position.latitude"
        """
        if source == SignalSource.NONE or not source.value:
            return signal.value
        return f"{source.value}.{signal.value}"


@dataclass
class ConversionFactor:
    """Defines how to convert raw values to standard units."""
    
    scale: float = 1.0
    offset: float = 0.0
    from_unit: str = ""
    to_unit: str = ""
    
    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply conversion to values."""
        return values * self.scale + self.offset
    
    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse conversion."""
        return (values - self.offset) / self.scale


# Common conversion presets
CONVERSION_PRESETS = {
    # GPS coordinates
    "gps_1e7_to_degrees": ConversionFactor(
        scale=1e-7, offset=0, from_unit="1e-7 deg", to_unit="degrees"
    ),
    "gps_1e5_to_degrees": ConversionFactor(
        scale=1e-5, offset=0, from_unit="1e-5 deg", to_unit="degrees"
    ),
    
    # Altitude
    "mm_to_meters": ConversionFactor(
        scale=0.001, offset=0, from_unit="mm", to_unit="m"
    ),
    "cm_to_meters": ConversionFactor(
        scale=0.01, offset=0, from_unit="cm", to_unit="m"
    ),
    "feet_to_meters": ConversionFactor(
        scale=0.3048, offset=0, from_unit="ft", to_unit="m"
    ),
    
    # Velocity
    "cm_s_to_m_s": ConversionFactor(
        scale=0.01, offset=0, from_unit="cm/s", to_unit="m/s"
    ),
    "knots_to_m_s": ConversionFactor(
        scale=0.514444, offset=0, from_unit="knots", to_unit="m/s"
    ),
    "mph_to_m_s": ConversionFactor(
        scale=0.44704, offset=0, from_unit="mph", to_unit="m/s"
    ),
    "kph_to_m_s": ConversionFactor(
        scale=0.277778, offset=0, from_unit="km/h", to_unit="m/s"
    ),
    
    # Angles
    "rad_to_deg": ConversionFactor(
        scale=57.2957795, offset=0, from_unit="rad", to_unit="deg"
    ),
    "deg_to_rad": ConversionFactor(
        scale=0.0174533, offset=0, from_unit="deg", to_unit="rad"
    ),
    "cdeg_to_deg": ConversionFactor(
        scale=0.01, offset=0, from_unit="cdeg", to_unit="deg"
    ),
    "mrad_to_deg": ConversionFactor(
        scale=0.0572958, offset=0, from_unit="mrad", to_unit="deg"
    ),
    
    # Angular rates
    "rad_s_to_deg_s": ConversionFactor(
        scale=57.2957795, offset=0, from_unit="rad/s", to_unit="deg/s"
    ),
    
    # Acceleration
    "mg_to_m_s2": ConversionFactor(
        scale=0.00981, offset=0, from_unit="mG", to_unit="m/s²"
    ),
    "g_to_m_s2": ConversionFactor(
        scale=9.81, offset=0, from_unit="G", to_unit="m/s²"
    ),
    
    # Pressure
    "pa_to_hpa": ConversionFactor(
        scale=0.01, offset=0, from_unit="Pa", to_unit="hPa"
    ),
    "mbar_to_hpa": ConversionFactor(
        scale=1.0, offset=0, from_unit="mbar", to_unit="hPa"
    ),
    
    # Temperature
    "kelvin_to_celsius": ConversionFactor(
        scale=1.0, offset=-273.15, from_unit="K", to_unit="°C"
    ),
    "cdeg_c_to_celsius": ConversionFactor(
        scale=0.01, offset=0, from_unit="cdeg_C", to_unit="°C"
    ),
    
    # Time
    "us_to_s": ConversionFactor(
        scale=1e-6, offset=0, from_unit="µs", to_unit="s"
    ),
    "ms_to_s": ConversionFactor(
        scale=0.001, offset=0, from_unit="ms", to_unit="s"
    ),
    
    # Voltage/Current
    "mv_to_v": ConversionFactor(
        scale=0.001, offset=0, from_unit="mV", to_unit="V"
    ),
    "ma_to_a": ConversionFactor(
        scale=0.001, offset=0, from_unit="mA", to_unit="A"
    ),
    "ca_to_a": ConversionFactor(
        scale=0.01, offset=0, from_unit="cA", to_unit="A"
    ),
    
    # No conversion
    "none": ConversionFactor(scale=1.0, offset=0),
}


@dataclass
class SignalMapping:
    """Maps a source column to a standard signal."""
    
    source_column: str
    target_signal: str  # StandardSignal value or custom path
    conversion: Optional[ConversionFactor] = None
    conversion_preset: Optional[str] = None
    signal_source: SignalSource = SignalSource.NONE  # measurement, command, estimated, etc.
    description: str = ""
    
    def __post_init__(self):
        """Apply conversion preset if specified."""
        if self.conversion_preset and not self.conversion:
            if self.conversion_preset in CONVERSION_PRESETS:
                self.conversion = CONVERSION_PRESETS[self.conversion_preset]
            else:
                logger.warning(f"Unknown conversion preset: {self.conversion_preset}")
    
    @property
    def full_target_path(self) -> str:
        """Get full target path including source prefix."""
        if self.signal_source == SignalSource.NONE or not self.signal_source.value:
            return self.target_signal
        return f"{self.signal_source.value}.{self.target_signal}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            'source_column': self.source_column,
            'target_signal': self.target_signal,
            'description': self.description
        }
        if self.signal_source != SignalSource.NONE:
            d['signal_source'] = self.signal_source.value
        if self.conversion_preset:
            d['conversion_preset'] = self.conversion_preset
        elif self.conversion:
            d['conversion'] = {
                'scale': self.conversion.scale,
                'offset': self.conversion.offset,
                'from_unit': self.conversion.from_unit,
                'to_unit': self.conversion.to_unit
            }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SignalMapping':
        """Create from dictionary."""
        conversion = None
        if 'conversion' in d:
            conversion = ConversionFactor(**d['conversion'])
        
        # Parse signal source
        signal_source = SignalSource.NONE
        if 'signal_source' in d:
            try:
                signal_source = SignalSource(d['signal_source'])
            except ValueError:
                logger.warning(f"Unknown signal source: {d['signal_source']}")
        
        return cls(
            source_column=d['source_column'],
            target_signal=d['target_signal'],
            conversion=conversion,
            conversion_preset=d.get('conversion_preset'),
            signal_source=signal_source,
            description=d.get('description', '')
        )


@dataclass
class AssignmentConfig:
    """Configuration for signal assignments."""
    
    name: str
    description: str = ""
    version: str = "1.0"
    mappings: List[SignalMapping] = field(default_factory=list)
    source_pattern: str = ""  # Regex pattern for matching source files
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_mapping(
        self,
        source_column: str,
        target_signal: Union[str, StandardSignal],
        conversion: Optional[Union[ConversionFactor, str]] = None,
        signal_source: Union[SignalSource, str] = SignalSource.NONE,
        description: str = ""
    ) -> None:
        """
        Add a signal mapping.
        
        Args:
            source_column: Name of column in source data
            target_signal: Target standard signal or custom path
            conversion: ConversionFactor, preset name, or None
            signal_source: Source type (measurement, command, estimated, etc.)
            description: Optional description
        """
        if isinstance(target_signal, StandardSignal):
            target_signal = target_signal.value
        
        # Handle signal source
        if isinstance(signal_source, str):
            try:
                signal_source = SignalSource(signal_source) if signal_source else SignalSource.NONE
            except ValueError:
                signal_source = SignalSource.NONE
        
        conversion_preset = None
        conversion_factor = None
        
        if isinstance(conversion, str):
            conversion_preset = conversion
        elif isinstance(conversion, ConversionFactor):
            conversion_factor = conversion
        
        mapping = SignalMapping(
            source_column=source_column,
            target_signal=target_signal,
            conversion=conversion_factor,
            conversion_preset=conversion_preset,
            signal_source=signal_source,
            description=description
        )
        
        self.mappings.append(mapping)
    
    def remove_mapping(self, source_column: str) -> bool:
        """Remove a mapping by source column name."""
        for i, m in enumerate(self.mappings):
            if m.source_column == source_column:
                del self.mappings[i]
                return True
        return False
    
    def get_mapping(self, source_column: str) -> Optional[SignalMapping]:
        """Get mapping for a source column."""
        for m in self.mappings:
            if m.source_column == source_column:
                return m
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'source_pattern': self.source_pattern,
            'metadata': self.metadata,
            'mappings': [m.to_dict() for m in self.mappings]
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AssignmentConfig':
        """Create from dictionary."""
        mappings = [
            SignalMapping.from_dict(m) for m in d.get('mappings', [])
        ]
        return cls(
            name=d['name'],
            description=d.get('description', ''),
            version=d.get('version', '1.0'),
            mappings=mappings,
            source_pattern=d.get('source_pattern', ''),
            metadata=d.get('metadata', {})
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Supports JSON and YAML formats based on file extension.
        """
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved assignment config to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AssignmentConfig':
        """
        Load configuration from file.
        
        Supports JSON and YAML formats.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        logger.info(f"Loaded assignment config from {path}")
        return cls.from_dict(data)


class SignalAssigner:
    """
    Applies signal assignments to flight data.
    
    Takes raw DataFrames and creates new DataFrames with
    standardized column names and converted values.
    """
    
    def __init__(self, config: AssignmentConfig):
        """
        Initialize SignalAssigner.
        
        Args:
            config: Assignment configuration to use
        """
        self.config = config
    
    def apply(
        self,
        df: pd.DataFrame,
        keep_unmapped: bool = True,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Apply signal assignments to a DataFrame.
        
        Args:
            df: Source DataFrame
            keep_unmapped: If True, keep columns without mappings
            prefix: Prefix for unmapped columns
            
        Returns:
            DataFrame with mapped and converted columns
        """
        result = pd.DataFrame()
        mapped_columns = set()
        
        for mapping in self.config.mappings:
            if mapping.source_column not in df.columns:
                logger.debug(f"Column not found: {mapping.source_column}")
                continue
            
            # Get source data
            values = df[mapping.source_column].values
            
            # Apply conversion if specified
            if mapping.conversion:
                values = mapping.conversion.apply(values)
            
            # Set target column (use full path including source prefix)
            result[mapping.full_target_path] = values
            mapped_columns.add(mapping.source_column)
        
        # Keep unmapped columns if requested
        if keep_unmapped:
            for col in df.columns:
                if col not in mapped_columns:
                    target_name = f"{prefix}{col}" if prefix else col
                    result[target_name] = df[col].values
        
        return result
    
    def apply_to_hierarchy(
        self,
        data: Dict,
        keep_unmapped: bool = True
    ) -> Dict:
        """
        Apply assignments to a hierarchical data structure.
        
        Args:
            data: Hierarchical dictionary of DataFrames
            keep_unmapped: Keep unmapped columns
            
        Returns:
            New hierarchy with mapped DataFrames
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                result[key] = self.apply(value, keep_unmapped)
            elif isinstance(value, dict):
                result[key] = self.apply_to_hierarchy(value, keep_unmapped)
            else:
                result[key] = value
        
        return result


def get_standard_signals() -> List[str]:
    """Get list of all standard signal names."""
    return [s.value for s in StandardSignal]


def get_conversion_presets() -> Dict[str, ConversionFactor]:
    """Get all available conversion presets."""
    return CONVERSION_PRESETS.copy()


def suggest_mappings(df: pd.DataFrame) -> List[Dict]:
    """
    Suggest mappings based on column names.
    
    Uses common naming patterns to suggest standard signals.
    """
    suggestions = []
    
    # Common name patterns to standard signals
    patterns = {
        # Position
        r'(?i)^lat(itude)?$': StandardSignal.POSITION_LATITUDE,
        r'(?i)^lon(gitude)?$': StandardSignal.POSITION_LONGITUDE,
        r'(?i)^alt(itude)?$': StandardSignal.POSITION_ALTITUDE,
        r'(?i)^(alt_)?msl$': StandardSignal.POSITION_ALTITUDE_MSL,
        
        # Attitude
        r'(?i)^roll$': StandardSignal.ATTITUDE_ROLL,
        r'(?i)^pitch$': StandardSignal.ATTITUDE_PITCH,
        r'(?i)^yaw$': StandardSignal.ATTITUDE_YAW,
        r'(?i)^heading$': StandardSignal.ATTITUDE_HEADING,
        
        # Gyro
        r'(?i)^gyro_?x$': StandardSignal.GYRO_X,
        r'(?i)^gyro_?y$': StandardSignal.GYRO_Y,
        r'(?i)^gyro_?z$': StandardSignal.GYRO_Z,
        
        # Accel
        r'(?i)^(x)?acc(el)?(_?x)?$': StandardSignal.ACCEL_X,
        r'(?i)^(y)?acc(el)?(_?y)?$': StandardSignal.ACCEL_Y,
        r'(?i)^(z)?acc(el)?(_?z)?$': StandardSignal.ACCEL_Z,
        
        # Battery
        r'(?i)^(batt(ery)?_?)?volt(age)?$': StandardSignal.BATTERY_VOLTAGE,
        r'(?i)^(batt(ery)?_?)?curr(ent)?$': StandardSignal.BATTERY_CURRENT,
        
        # Time
        r'(?i)^time(stamp)?$': StandardSignal.TIMESTAMP,
        r'(?i)^time_?boot_?ms$': StandardSignal.TIME_BOOT_MS,
    }
    
    import re
    
    for col in df.columns:
        for pattern, signal in patterns.items():
            if re.match(pattern, col):
                suggestions.append({
                    'column': col,
                    'suggested_signal': signal.value,
                    'confidence': 'high'
                })
                break
    
    return suggestions

