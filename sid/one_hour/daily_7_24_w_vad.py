import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timezone

# --- ERA5: loop over all 24 hourly files ---
era5_dir = '/scr/isf_apg/models/m2hats/era5/20230724/'
era5_files = sorted(glob.glob(era5_dir + 'era5_pressure_20230724_*_ISS1.nc'))

era5_times = []
era5_ws_profiles = []  # one profile per hour

for f in era5_files:
    ds = nc.Dataset(f)
    u = ds.variables['u'][0, :, 0, 0]
    v = ds.variables['v'][0, :, 0, 0]
    t = ds.variables['valid_time'][0]
    ws = np.sqrt(u**2 + v**2)
    era5_times.append(datetime.fromtimestamp(t, tz=timezone.utc))
    era5_ws_profiles.append(ws)
    ds.close()

era5_ws_profiles = np.array(era5_ws_profiles)  # (24, 37)
era5_ws_profiles = np.ma.masked_invalid(era5_ws_profiles)

# get altitude from last file (same for all)
ds = nc.Dataset(era5_files[0])
z = ds.variables['z'][0, :, 0, 0]
alt_era5 = z / 9.80665
ds.close()

# --- VAD ---
vad = nc.Dataset('/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/30min_winds_20230724.nc')
ws_vad = vad.variables['wind_speed'][:]  # (48, 100)
height_vad = vad.variables['height'][:]  # (100,)
base_time = vad.variables['base_time'][:]
time_offset = vad.variables['time_offset'][:]
vad_times = [datetime.fromtimestamp(base_time + t, tz=timezone.utc) for t in time_offset]
ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
print(f"ERA5 altitude points: {len(alt_era5)}")
print(f"VAD height points: {len(height_vad)}")
print(np.sum(alt_era5 < 5000))
# --- Plot: day-averaged vertical profile comparison ---
era5_ws_mean = np.ma.mean(era5_ws_profiles, axis=0)  # avg over 24 hours
vad_ws_mean = np.ma.mean(ws_vad, axis=0)              # avg over 48 half-hours

plt.figure(figsize=(6, 10))
plt.scatter(era5_ws_mean, alt_era5, label='ERA5 (day avg)', color='blue')
plt.scatter(vad_ws_mean, height_vad, label='VAD (day avg)', color='red', s=10)
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Altitude (m)')
plt.title('ERA5 vs VAD Wind Speed Vertical Profile\n2023-07-24 Day Average')
plt.legend()
plt.grid(True)
plt.ylim(0, 5000)
plt.show()

