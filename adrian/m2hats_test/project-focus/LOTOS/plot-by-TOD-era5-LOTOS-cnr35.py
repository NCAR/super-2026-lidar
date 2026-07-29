import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/lotos2025/era5/'  # CONFIRM this path
vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'
_first_vad = sorted(glob.glob(vad_base + 'VAD_*.nc'))[0]
_v = nc.Dataset(_first_vad)
site_alt = float(_v.variables['alt'][:])
_v.close()
print(f"Site elevation read from VAD: {site_alt:.1f} m MSL")

hours_s, sdiff_m = [], []
hours_d, ddiff_m = [], []

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
    # one ERA5 file per day; filename has no hour or ISS tag
    e_file = era5_base + 'era5_pressure_' + date + '_lotos2025.nc'
    if not os.path.exists(e_file):
        continue

    vad = nc.Dataset(vad_file)
    ws_vad = vad.variables['wind_speed'][:]
    wd_vad = vad.variables['wind_direction'][:]
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()

    ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
    vad_epoch = base_t + time_vad

    # read the whole day's ERA5 at once
    ds = nc.Dataset(e_file)
    u_all = ds.variables['u'][:, :, 0, 0]   # shape (24, 37)
    v_all = ds.variables['v'][:, :, 0, 0]
    z_all = ds.variables['z'][:, :, 0, 0]
    vt_all = ds.variables['valid_time'][:]  # shape (24,)
    ds.close()

    # loop over the 24 hourly time slices in this file
    for t in range(len(vt_all)):
        u = u_all[t]
        v = v_all[t]
        z = z_all[t]
        et = int(vt_all[t])

        era5_ws = np.sqrt(u**2 + v**2)
        era5_dir = np.degrees(np.arctan2(-u, -v)) % 360
        era5_agl = z / 9.80665 - site_alt

        hour = datetime.fromtimestamp(et, tz=timezone.utc).hour

        ti = np.argmin(np.abs(vad_epoch - et))
        if abs(vad_epoch[ti] - et) > 900:
            continue

        valid = ~np.ma.getmaskarray(ws_vad[ti])
        if not valid.any():
            continue

        h_valid = height[valid]
        idx_valid = np.where(valid)[0]

        for k in range(len(era5_agl)):
            if not (100 <= era5_agl[k] <= 2000):
                continue
            j = np.argmin(np.abs(h_valid - era5_agl[k]))
            if np.abs(h_valid[j] - era5_agl[k]) > 25:
                continue
            idx = idx_valid[j]

            if np.ma.is_masked(ws_vad[ti, idx]):
                continue
            # physical sanity guard (the _error VAD datasets contain garbage values)
            if not np.isfinite(ws_vad[ti, idx]) or ws_vad[ti, idx] < 0 or ws_vad[ti, idx] > 60:
                continue

            hours_s.append(hour)
            sdiff_m.append(float(ws_vad[ti, idx] - era5_ws[k]))

            if era5_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
                dd = ((float(wd_vad[ti, idx]) - era5_dir[k] + 180) % 360) - 180
                hours_d.append(hour)
                ddiff_m.append(dd)

hours_s = np.array(hours_s); sdiff_m = np.array(sdiff_m)
hours_d = np.array(hours_d); ddiff_m = np.array(ddiff_m)

print(f"Total speed points: {len(sdiff_m)}")
print(f"Total direction points: {len(ddiff_m)}")
if len(sdiff_m):
    print(f"sdiff range: {sdiff_m.min():.1f} to {sdiff_m.max():.1f}")

def hourly_means(hour_arr, diff_arr):
    means, stds, counts = [], [], []
    for h in range(24):
        sel = hour_arr == h
        if sel.any():
            means.append(np.mean(diff_arr[sel]))
            stds.append(np.std(diff_arr[sel]))
            counts.append(sel.sum())
        else:
            means.append(np.nan); stds.append(np.nan); counts.append(0)
    return np.arange(24), np.array(means), np.array(stds), np.array(counts)

hh, s_mean, s_std, s_cnt = hourly_means(hours_s, sdiff_m)
_,  d_mean, d_std, d_cnt = hourly_means(hours_d, ddiff_m)

# --- plot ---
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle('Mean Difference by Hour of Day (VAD - ERA5), LOTOS2025 using cnr35', fontsize=16)

ax1.errorbar(hh, s_mean, yerr=s_std, fmt='o-', color='steelblue',
             ecolor='lightgray', capsize=3, markersize=5)
ax1.axhline(0, color='red', linewidth=1)
ax1.set_ylabel('Wind Speed Diff (m/s)', fontsize=14)
ax1.set_title('Wind Speed', fontsize=15)
ax1.tick_params(labelsize=13)
ax1.grid(alpha=0.3)

ax2.errorbar(hh, d_mean, yerr=d_std, fmt='o-', color='indianred',
             ecolor='lightgray', capsize=3, markersize=5)
ax2.axhline(0, color='red', linewidth=1)
ax2.set_ylabel('Wind Dir Diff (deg)', fontsize=14)
ax2.set_xlabel('Hour of Day (UTC)', fontsize=14)
ax2.set_title('Wind Direction', fontsize=15)
ax2.tick_params(labelsize=13)
ax2.grid(alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.show()