import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

ds = nc.Dataset('/scr/isf_apg/models/m2hats/era5/20230724/era5_pressure_20230724_00_ISS1.nc')

u = ds.variables['u'][0, :, 0, 0]
v = ds.variables['v'][0, :, 0, 0]
z = ds.variables['z'][0, :, 0, 0]

altitude = z / 9.80665  # convert geopotential to meters

windspeed = np.sqrt(u**2 + v**2)

plt.figure(figsize=(6, 10))
plt.scatter(windspeed, altitude)
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Altitude (m)')
plt.title('ERA5 Wind Speed Vertical Profile\n2023-07-24 00Z')
plt.grid(True)
plt.show()

