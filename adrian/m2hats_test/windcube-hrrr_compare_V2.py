import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

hrrr_ws_m, vad_ws_m, hrrr_dir_m, vad_dir_m = [], [], [], []

for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
  date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
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
      hrrr_dir_m.append(hrrr_dir[k]); vad_dir_m.append(wd_vad[ti, idx])
      
hrrr_ws_m = np.array(hrrr_ws_m); vad_ws_m = np.array(vad_ws_m)
hrrr_dir_m = np.array(hrrr_dir_m); vad_dir_m = np.array(vad_dir_m)
print(f"Matched points: {len(hrrr_ws_m)}")

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
fig.suptitle('M2HATS 23 Jul - 24 Sep 2023')

ax1.scatter(hrrr_ws_m, vad_ws_m, s=2, color='black')
ax1.set_xlim(0, 20); ax1.set_ylim(0, 20)
ax1.set_xlabel('HRRR Wind Speed (m/s)'); ax1.set_ylabel('Windcube VAD Wind Speed (m/s)')
ax1.text(0.5, 19, f"{len(hrrr_ws_m)} pts, mad: {s_mad:.1f}, sd: {s_sd:.1f}")
ax1.text(0.5, 17.5, f"fit: {s_fit[0]:.1f}x + {s_fit[1]:.2f}")

ax2.scatter(hrrr_dir_m[dmask], vad_dir_m[dmask], s=2, color='black')
ax2.set_xlim(0, 360); ax2.set_ylim(0, 360)
ax2.set_xlabel('HRRR Wind Dirn (deg)'); ax2.set_ylabel('Windcube VAD Wind Dirn (deg)')
ax2.text(10, 340, f"{int(dmask.sum())} pts, mad: {d_mad:.1f}, sd: {d_sd:.1f}")
ax2.text(10, 315, f"fit: {d_fit[0]:.1f}x + {d_fit[1]:.2f}")

plt.show()