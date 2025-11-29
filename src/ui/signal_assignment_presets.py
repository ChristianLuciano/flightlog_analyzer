"""
Signal assignment presets and standard signals for the UI.

Provides predefined options for signal assignment dropdowns.
"""

from typing import Dict, List


# Signal source types for grouping
SIGNAL_SOURCES = [
    {'value': '', 'label': '(No source prefix)'},
    {'value': 'measurement', 'label': '📊 Measurement (sensor data)'},
    {'value': 'command', 'label': '🎯 Command (setpoints)'},
    {'value': 'estimated', 'label': '📈 Estimated (filtered/Kalman)'},
    {'value': 'reference', 'label': '📌 Reference (targets)'},
    {'value': 'simulated', 'label': '💻 Simulated (model output)'},
    {'value': 'raw', 'label': '📝 Raw (unprocessed)'},
]


# Standard flight dynamics signals organized by category
STANDARD_SIGNALS = {
    'Position': [
        {'value': 'position.latitude', 'label': 'Latitude'},
        {'value': 'position.longitude', 'label': 'Longitude'},
        {'value': 'position.altitude', 'label': 'Altitude'},
        {'value': 'position.altitude_msl', 'label': 'Altitude (MSL)'},
        {'value': 'position.altitude_agl', 'label': 'Altitude (AGL)'},
    ],
    'Velocity': [
        {'value': 'velocity.north', 'label': 'Velocity North'},
        {'value': 'velocity.east', 'label': 'Velocity East'},
        {'value': 'velocity.down', 'label': 'Velocity Down'},
        {'value': 'velocity.ground_speed', 'label': 'Ground Speed'},
        {'value': 'velocity.airspeed', 'label': 'Airspeed'},
        {'value': 'velocity.vertical', 'label': 'Vertical Speed'},
    ],
    'Attitude': [
        {'value': 'attitude.roll', 'label': 'Roll'},
        {'value': 'attitude.pitch', 'label': 'Pitch'},
        {'value': 'attitude.yaw', 'label': 'Yaw'},
        {'value': 'attitude.heading', 'label': 'Heading'},
    ],
    'Angular Rates': [
        {'value': 'gyro.x', 'label': 'Gyro X'},
        {'value': 'gyro.y', 'label': 'Gyro Y'},
        {'value': 'gyro.z', 'label': 'Gyro Z'},
        {'value': 'gyro.roll_rate', 'label': 'Roll Rate'},
        {'value': 'gyro.pitch_rate', 'label': 'Pitch Rate'},
        {'value': 'gyro.yaw_rate', 'label': 'Yaw Rate'},
    ],
    'Acceleration': [
        {'value': 'accel.x', 'label': 'Accel X'},
        {'value': 'accel.y', 'label': 'Accel Y'},
        {'value': 'accel.z', 'label': 'Accel Z'},
    ],
    'Magnetometer': [
        {'value': 'mag.x', 'label': 'Mag X'},
        {'value': 'mag.y', 'label': 'Mag Y'},
        {'value': 'mag.z', 'label': 'Mag Z'},
    ],
    'GPS': [
        {'value': 'gps.fix_type', 'label': 'Fix Type'},
        {'value': 'gps.satellites', 'label': 'Satellites'},
        {'value': 'gps.hdop', 'label': 'HDOP'},
        {'value': 'gps.vdop', 'label': 'VDOP'},
    ],
    'Battery': [
        {'value': 'battery.voltage', 'label': 'Voltage'},
        {'value': 'battery.current', 'label': 'Current'},
        {'value': 'battery.remaining', 'label': 'Remaining %'},
    ],
    'RC Input': [
        {'value': 'rc.throttle', 'label': 'Throttle'},
        {'value': 'rc.roll', 'label': 'Roll'},
        {'value': 'rc.pitch', 'label': 'Pitch'},
        {'value': 'rc.yaw', 'label': 'Yaw'},
        {'value': 'rc.channel_1', 'label': 'Channel 1'},
        {'value': 'rc.channel_2', 'label': 'Channel 2'},
        {'value': 'rc.channel_3', 'label': 'Channel 3'},
        {'value': 'rc.channel_4', 'label': 'Channel 4'},
    ],
    'Motors': [
        {'value': 'motor.1', 'label': 'Motor 1'},
        {'value': 'motor.2', 'label': 'Motor 2'},
        {'value': 'motor.3', 'label': 'Motor 3'},
        {'value': 'motor.4', 'label': 'Motor 4'},
    ],
    'Time': [
        {'value': 'time.timestamp', 'label': 'Timestamp'},
        {'value': 'time.boot_ms', 'label': 'Boot Time (ms)'},
        {'value': 'time.utc', 'label': 'UTC Time'},
    ],
}


# Conversion presets for dropdown
CONVERSION_PRESETS = [
    {'value': 'none', 'label': 'No conversion'},
    # Separator
    {'value': '_gps', 'label': '── GPS ──', 'disabled': True},
    {'value': 'gps_1e7_to_degrees', 'label': '1e-7 deg → degrees'},
    {'value': 'gps_1e5_to_degrees', 'label': '1e-5 deg → degrees'},
    # Separator
    {'value': '_length', 'label': '── Length ──', 'disabled': True},
    {'value': 'mm_to_meters', 'label': 'mm → m'},
    {'value': 'cm_to_meters', 'label': 'cm → m'},
    {'value': 'feet_to_meters', 'label': 'ft → m'},
    # Separator
    {'value': '_velocity', 'label': '── Velocity ──', 'disabled': True},
    {'value': 'cm_s_to_m_s', 'label': 'cm/s → m/s'},
    {'value': 'knots_to_m_s', 'label': 'knots → m/s'},
    {'value': 'mph_to_m_s', 'label': 'mph → m/s'},
    {'value': 'kph_to_m_s', 'label': 'km/h → m/s'},
    # Separator
    {'value': '_angles', 'label': '── Angles ──', 'disabled': True},
    {'value': 'rad_to_deg', 'label': 'rad → deg'},
    {'value': 'deg_to_rad', 'label': 'deg → rad'},
    {'value': 'cdeg_to_deg', 'label': 'cdeg → deg'},
    {'value': 'mrad_to_deg', 'label': 'mrad → deg'},
    # Separator
    {'value': '_rates', 'label': '── Angular Rates ──', 'disabled': True},
    {'value': 'rad_s_to_deg_s', 'label': 'rad/s → deg/s'},
    # Separator
    {'value': '_accel', 'label': '── Acceleration ──', 'disabled': True},
    {'value': 'mg_to_m_s2', 'label': 'mG → m/s²'},
    {'value': 'g_to_m_s2', 'label': 'G → m/s²'},
    # Separator
    {'value': '_pressure', 'label': '── Pressure ──', 'disabled': True},
    {'value': 'pa_to_hpa', 'label': 'Pa → hPa'},
    {'value': 'mbar_to_hpa', 'label': 'mbar → hPa'},
    # Separator
    {'value': '_temp', 'label': '── Temperature ──', 'disabled': True},
    {'value': 'kelvin_to_celsius', 'label': 'K → °C'},
    {'value': 'cdeg_c_to_celsius', 'label': 'cdeg_C → °C'},
    # Separator
    {'value': '_time', 'label': '── Time ──', 'disabled': True},
    {'value': 'us_to_s', 'label': 'µs → s'},
    {'value': 'ms_to_s', 'label': 'ms → s'},
    # Separator
    {'value': '_electrical', 'label': '── Electrical ──', 'disabled': True},
    {'value': 'mv_to_v', 'label': 'mV → V'},
    {'value': 'ma_to_a', 'label': 'mA → A'},
    {'value': 'ca_to_a', 'label': 'cA → A'},
    # Custom
    {'value': '_custom', 'label': '── Custom ──', 'disabled': True},
    {'value': 'custom', 'label': 'Custom (enter values below)'},
]


def get_signal_options() -> List[Dict]:
    """Get flat list of signal options for dropdown."""
    options = []
    for category, signals in STANDARD_SIGNALS.items():
        # Add category separator
        options.append({
            'value': f'_cat_{category}',
            'label': f'── {category} ──',
            'disabled': True
        })
        # Add signals
        options.extend(signals)
    
    # Add custom option
    options.append({
        'value': '_custom',
        'label': '── Custom ──',
        'disabled': True
    })
    options.append({
        'value': 'custom',
        'label': 'Custom path (enter below)'
    })
    
    return options


def get_conversion_options() -> List[Dict]:
    """Get conversion preset options for dropdown."""
    return CONVERSION_PRESETS.copy()


def get_source_options() -> List[Dict]:
    """Get signal source options for dropdown."""
    return SIGNAL_SOURCES.copy()

