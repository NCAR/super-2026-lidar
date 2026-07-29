import netCDF4 as nc
import numpy as np
import glob

sf = sorted(glob.glob('/scr/isf_apg/models/m2hats/era5/*/era5_surface_*_ISS1.nc'))[0]
ds = nc.Dataset(sf)
print(list(ds.variables.keys()))
if 'z' in ds.variables:
    z = ds.variables['z'][:]
    print('surface z shape:', z.shape)
    print('surface elevation estimate:', float(np.ravel(z)[0]) / 9.80665, 'm')
ds.close()