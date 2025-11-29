# Troubleshooting Guide

This guide helps resolve common issues with the Flight Log Analysis Dashboard.

## Quick Diagnostics

Run the diagnostic script to check your installation:

```bash
python -c "
from src.core.version import get_version
print(f'Version: {get_version()}')

import dash; print(f'Dash: {dash.__version__}')
import plotly; print(f'Plotly: {plotly.__version__}')
import pandas; print(f'Pandas: {pandas.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')

try:
    from pymavlink import mavutil
    print('pymavlink: OK')
except ImportError:
    print('pymavlink: NOT INSTALLED')
"
```

---

## Installation Issues

### ModuleNotFoundError: No module named 'src'

**Cause**: Package not installed in development mode.

**Solution**:
```bash
cd /path/to/Flight_Log
pip install -e .
```

### pip install fails with "Failed building wheel"

**Cause**: Missing C compiler for some dependencies.

**Solution** (Windows):
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "Desktop development with C++"
3. Retry installation

**Solution** (Linux):
```bash
sudo apt install build-essential python3-dev
```

### pymavlink installation fails

**Windows**:
```bash
# Install pre-built wheel
pip install pymavlink --only-binary=:all:

# Or install from conda-forge
conda install -c conda-forge pymavlink
```

**Linux/macOS**:
```bash
pip install future lxml
pip install pymavlink
```

---

## Startup Issues

### Port 8050 already in use

**Error**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Find and kill the process using the port
# Linux/macOS:
lsof -i :8050
kill -9 <PID>

# Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# Or use a different port:
# In your code, change: app.run(port=8051)
```

### Dashboard starts but browser shows blank page

**Possible causes**:
1. JavaScript errors in browser console
2. Firewall blocking localhost
3. Antivirus interference

**Solutions**:
1. Open browser dev tools (F12) and check Console tab
2. Add exception for localhost:8050 in firewall
3. Temporarily disable antivirus and test

### Browser shows "This site can't be reached"

**Cause**: Dashboard didn't start properly.

**Solution**: Check terminal output for errors:
```bash
# Run with debug output
python examples/basic_usage.py 2>&1 | tee debug.log
```

---

## Data Loading Issues

### "Unsupported format" when loading .tlog file

**Cause**: pymavlink not installed.

**Solution**:
```bash
pip install pymavlink
```

### "Error parsing MAVLink" 

**Possible causes**:
1. Corrupted .tlog file
2. Incompatible pymavlink version
3. File permissions issue

**Solutions**:
1. Try a different .tlog file to verify
2. Update pymavlink: `pip install --upgrade pymavlink`
3. Check file is readable: `ls -la yourfile.tlog`

### Old signals still showing after loading new data

**Cause**: Known issue in versions < 0.1.0 where callbacks used stale data reference.

**Solution**: Update to latest version or restart the dashboard.

### GPS coordinates show as 0 or incorrect

**Cause**: MAVLink coordinates are stored as integers (degrees × 1e7).

**Solution**: The dashboard automatically scales these. If still wrong:
```python
# Check your data format
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df['lat'].describe())
# If max > 1e6, data needs scaling
```

### Large file takes forever to load

**Cause**: File too large for browser memory.

**Solutions**:
1. Use time range filter to load subset
2. Trim the file before loading:
```python
import pandas as pd
df = pd.read_csv('large_file.csv')
df_trimmed = df[df['timestamp'] < 3600]  # First hour
df_trimmed.to_csv('trimmed.csv', index=False)
```

---

## Visualization Issues

### Plots are empty or show "No data"

**Possible causes**:
1. Data columns don't match expected names
2. Data is non-numeric
3. Time column not found

**Solutions**:
1. Check data structure in Data Summary panel
2. Verify numeric data: `df.dtypes`
3. Ensure time column exists: `timestamp`, `time`, `time_boot_ms`, or `time_usec`

### Map doesn't display

**Possible causes**:
1. No GPS data in file
2. Invalid coordinates
3. No internet connection (maps require network)

**Solutions**:
1. Check if GPS DataFrame exists in data tree
2. Verify lat/lon values are valid (not all zeros)
3. For offline use, implement offline tiles (future feature)

### Attitude/Gyroscope plot missing

**Cause**: No recognized attitude columns found.

**Solution**: Ensure your data has one of these column sets:
- `roll`, `pitch`, `yaw`
- `gyro_x`, `gyro_y`, `gyro_z`
- `rollspeed`, `pitchspeed`, `yawspeed`

### Plots are very slow or laggy

**Cause**: Too many data points rendered.

**Solutions**:
1. Use time range filter to limit data
2. Enable downsampling (automatic for > 100k points)
3. Close other browser tabs
4. Use Chrome/Firefox (better WebGL support than Safari)

---

## Export Issues

### Excel export fails

**Cause**: openpyxl not installed.

**Solution**:
```bash
pip install openpyxl
```

### Export produces empty file

**Cause**: No data selected or data filtering returned empty set.

**Solution**: 
1. Select signals in the signal selector
2. Check time range doesn't exclude all data
3. Verify scope is set correctly ("All Data" vs "Selected Signals")

---

## Performance Issues

### High memory usage

**Cause**: Large datasets loaded in memory.

**Solutions**:
1. Close unused browser tabs
2. Load only needed signals
3. Use time range filter
4. Restart dashboard periodically for long sessions

### Dashboard becomes unresponsive

**Possible causes**:
1. Browser memory limit reached
2. Too many plots rendered
3. Large dataset in overview

**Solutions**:
1. Clear custom plots
2. Refresh the page
3. Reduce data size before loading

---

## Configuration Issues

### Settings not saved

**Cause**: No write permission to config directory.

**Solution**:
```bash
# Check config directory permissions
# Linux/macOS:
ls -la ~/.config/FlightLogDashboard/

# Windows:
icacls %APPDATA%\FlightLogDashboard
```

### Configuration file corrupted

**Solution**: Delete and recreate:
```bash
# Linux/macOS:
rm ~/.config/FlightLogDashboard/config.yaml

# Windows:
del %APPDATA%\FlightLogDashboard\config.yaml
```

---

## Getting Help

If your issue isn't covered here:

1. **Check existing issues**: https://github.com/flight-log/dashboard/issues
2. **Create a new issue** with:
   - Operating system and version
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "dash|plotly|pandas"`
   - Error message (full traceback)
   - Steps to reproduce
   - Sample data file (if possible)

3. **Join the discussion**: https://github.com/flight-log/dashboard/discussions

