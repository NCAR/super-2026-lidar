import netCDF4 as nc
import numpy as np

sf = '/net/isf/radiosonde_archive/2023_m2hats/qc/ncdf_v1/NCAR_M2HATS_ISS1_RS41_v1_20230723_172736_asc.nc'
ds = nc.Dataset(sf)

print("launch_time:", ds.variables['launch_time'][:], ds.variables['launch_time'].units)
print("time[:5]:", ds.variables['time'][:5], ds.variables['time'].units)
print("alt[:5]:", ds.variables['alt'][:5], "  -- is this MSL or AGL?")
print("alt units:", ds.variables['alt'].units)
ds.close()