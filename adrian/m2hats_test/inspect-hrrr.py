import netCDF4 as nc, glob

hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'  # adjust to your dataset
h_files = sorted(glob.glob(hrrr_base + '*/hrrr_profile_*_ISS1.nc'))
print(f"Found {len(h_files)} HRRR files")

ds = nc.Dataset(h_files[0])
wspd = ds.variables['wspd']
print('attributes:', {a: wspd.getncattr(a) for a in wspd.ncattrs()})
raw = wspd[:]
print('min/max:', raw.min(), raw.max())
print('any huge values:', (raw > 100).sum() if hasattr(raw, 'count') else 'n/a')
ds.close()