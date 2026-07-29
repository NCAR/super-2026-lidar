import netCDF4 as nc, glob

era5_base = '/scr/isf_apg/models/m2hats/era5/'
# adjust this glob to match your actual ERA5 filenames
e_files = sorted(glob.glob(era5_base + '*/era5_pressure_*_ISS1.nc'))
print(f"Found {len(e_files)} ERA5 files")

ds = nc.Dataset(e_files[0])
print('Variables:')
for v in ds.variables.keys():
    var = ds.variables[v]
    print(f"  {v}: shape={var.shape}, units={getattr(var, 'units', 'none')}")
ds.close()