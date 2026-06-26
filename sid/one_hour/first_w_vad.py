import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# --- ERA5 ---
era5 = nc.Dataset('/scr/isf_apg/models/m2hats/era5/20230724/era5_pressure_20230724_00_ISS1.nc')
u_era5 = era5.variables['u'][0, :, 0, 0]
v_era5 = era5.variables['v'][0, :, 0, 0]
z_era5 = era5.variables['z'][0, :, 0, 0]
alt_era5 = z_era5 / 9.80665  # geopotential to meters
ws_era5 = np.sqrt(u_era5**2 + v_era5**2)

# --- VAD ---
vad = nc.Dataset('/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/30min_winds_20230724.nc')
ws_vad = vad.variables['wind_speed'][:]  # (48, 100)
height_vad = vad.variables['height'][:]  # (100,)

# mask missing values
ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)

# average VAD over all 48 time steps for fair comparison with single ERA5 hour
ws_vad_mean = np.ma.mean(ws_vad, axis=0)

# --- Plot ---
plt.figure(figsize=(6, 10))
plt.scatter(ws_era5, alt_era5, label='ERA5 00Z', color='blue')
plt.scatter(ws_vad_mean, height_vad, label='VAD (day avg)', color='red', s=10)
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Altitude (m)')
plt.title('ERA5 vs VAD Wind Speed Vertical Profile\n2023-07-24')
plt.legend()
plt.grid(True)
plt.show()


