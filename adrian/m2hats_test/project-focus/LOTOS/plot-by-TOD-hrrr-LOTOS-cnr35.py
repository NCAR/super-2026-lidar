import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

hrrr_base = '/scr/isf_apg/models/lotos2025/hrrr/'
vad_base = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'

# collect hour of day, ws diff, dir diff per matched point
hours_s, sdiff_m = [], []
hours_d, ddiff_m = [], []

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
    h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))

    if not h_files:
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

    for f in h_files:
        ds = nc.Dataset(f)
        hrrr_ws = ds.variables['wspd'][:]
        hrrr_dir = ds.variables['wdir'][:]
        hrrr_agl = ds.variables['height'][:]
        et = int(ds.variables['time'][0])
        ds.close()

        # hour of day (UTC) from the HRRR epoch timestamp
        hour = datetime.fromtimestamp(et, tz=timezone.utc).hour

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

            hours_s.append(hour)
            sdiff_m.append(ws_vad[ti, idx] - hrrr_ws[k])

            if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0:
                dd = ((wd_vad[ti, idx] - hrrr_dir[k] + 180) % 360) - 180
                hours_d.append(hour)
                ddiff_m.append(dd)

hours_s = np.array(hours_s)
sdiff_m = np.array(sdiff_m)
hours_d = np.array(hours_d)
ddiff_m = np.array(ddiff_m)

print(f"Total speed points: {len(sdiff_m)}")
print(f"Total direction points: {len(ddiff_m)}")

# average difference per hour of day
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
fig.suptitle('Mean Difference by Hour of Day (VAD - HRRR), LOTOS2025 using cnr35', fontsize=16)

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