import netCDF4 as nc, glob

vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_cnr_33_witherror/'  # use the same path that gave you 5593 points

vad_files = sorted(glob.glob(vad_base + 'VAD_*.nc'))  # adjust pattern if needed
print(f"Found {len(vad_files)} files")

vad = nc.Dataset(vad_files[0])
ws = vad.variables['wind_speed']
print('dtype:', ws.dtype)
print('attributes:', {a: ws.getncattr(a) for a in ws.ncattrs()})
raw = ws[:]
print('type after read:', type(raw))
print('min/max:', raw.min(), raw.max())
if hasattr(raw, 'mask'):
    print('already masked count:', raw.mask.sum())
vad.close()