import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

hrrr_base = '/scr/isf_apg/models/lotos2025/hrrr/'
vad_base = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'

hrrr_ws_m, vad_ws_m, hrrr_dir_m, vad_dir_m = [], [], [], []

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
  date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
  h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))
  
  if not h_files:
    continue
    
  # surface elevation (MSL) from surface geopotential - constant, take first file
#  ds0 = nc.Dataset(h_files[0])
#  z0 = ds0.variables['z'][0, :, 0, 0]
#  surf_alt = float(z0[-1]) / 9.80665   # lowest pressure level as proxy for surface
#  ds0.close()
  
  # VAD
  vad = nc.Dataset(vad_file)
  ws_vad = vad.variables['wind_speed'][:]
  wd_vad = vad.variables['wind_direction'][:]
  height = vad.variables['height'][:]            # AGL
  base_t = int(vad.variables['base_time'][:])
  time_vad = vad.variables['time'][:]
  vad.close()
  ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
  wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
  vad_epoch = base_t + time_vad
  
  # HRRR pressure levels per hour
  for f in h_files:
    ds = nc.Dataset(f)
    hrrr_ws = ds.variables['wspd'][:]
    hrrr_dir = ds.variables['wdir'][:]
    hrrr_agl = ds.variables['height'][:]
    et = int(ds.variables['time'][0])
    ds.close()
    
#    hrrr_ws = np.sqrt(u**2 + v**2)
#    hrrr_dir = np.degrees(np.arctan2(-u, -v)) % 360
#    hrrr_agl = z / 9.80665 - surf_alt
    
    ti = np.argmin(np.abs(vad_epoch - et))
    if abs(vad_epoch[ti] - et) > 900:
      continue
    valid = ~np.ma.getmaskarray(ws_vad[ti])
    if not valid.any():
      continue
    h_valid = height[valid]
    idx_valid = np.where(valid)[0]

    for k in range(len(hrrr_agl)):
      if not (100 <= hrrr_agl[k] <= 2000):
        continue
      j = np.argmin(np.abs(h_valid - hrrr_agl[k]))
      if np.abs(h_valid[j] - hrrr_agl[k]) > 25:
        continue
      idx = idx_valid[j]
      hrrr_ws_m.append(hrrr_ws[k]); vad_ws_m.append(ws_vad[ti, idx])
      if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0:
        hrrr_dir_m.append(hrrr_dir[k]); vad_dir_m.append(wd_vad[ti, idx])
      
hrrr_ws_m = np.array(hrrr_ws_m); vad_ws_m = np.array(vad_ws_m)
hrrr_dir_m = np.array(hrrr_dir_m); vad_dir_m = np.array(vad_dir_m)
print(f"Matched points: {len(hrrr_ws_m)}")
print(f"Matched points: {len(hrrr_dir_m)}")

if len(hrrr_ws_m) == 0:
  raise SystemExit("Still zero matches - check surf_alt and hrrr_agl values.")
  
# stats
sdiff = vad_ws_m - hrrr_ws_m
s_mad, s_sd = np.mean(np.abs(sdiff)), np.std(sdiff)
s_fit = np.polyfit(hrrr_ws_m, vad_ws_m, 1)

dmask = ~np.isnan(vad_dir_m) & ~np.isnan(hrrr_dir_m)
cdiff = ((vad_dir_m[dmask] - hrrr_dir_m[dmask] + 180) % 360) - 180
d_mad, d_sd = np.mean(np.abs(cdiff)), np.std(cdiff)
d_fit = np.polyfit(hrrr_dir_m[dmask], vad_dir_m[dmask], 1)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('LOTOS2025 - Difference Histogram')

# wind speed difference histogram
ax1.hist(sdiff, bins=50, color='black', edgecolor='none')
ax1.axvline(0, color='red', linewidth=1)
ax1.set_xlabel('VAD - HRRR Wind Speed (m/s)')
ax1.set_ylabel('Count')
ax1.set_title('Wind Speed Difference')
ax1.text(0.05, 0.95, f"{len(hrrr_ws_m)} pts\nmad: {s_mad:.1f}, sd: {s_sd:.1f}",
         transform=ax1.transAxes, va='top')

# wind direction difference histogram
ax2.hist(cdiff, bins=50, color='black', edgecolor='none')
ax2.axvline(0, color='red', linewidth=1)
ax2.set_xlabel('VAD - HRRR Wind Direction (deg)')
ax2.set_ylabel('Count')
ax2.set_title('Wind Direction Difference')
ax2.text(0.05, 0.95, f"{int(dmask.sum())} pts\nmad: {d_mad:.1f}, sd: {d_sd:.1f}",
         transform=ax2.transAxes, va='top')

plt.tight_layout()
plt.show()