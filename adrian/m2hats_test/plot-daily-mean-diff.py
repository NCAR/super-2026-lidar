import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# collect date string, ws diff, dir diff per matched point
# dates_s / sdiff_m: date string (YYYYMMDD) and VAD - HRRR wind speed difference, one entry per matched point
# dates_d / ddiff_m: date string (YYYYMMDD) and VAD - HRRR wind direction difference, one entry per matched point
#                    (only kept when both speeds are above the calm-wind cutoff)
dates_s, sdiff_m = [], []
dates_d, ddiff_m = [], []

# Loop over every VAD consensus wind file (one per day), sorted chronologically
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    # Extract the date string from the filename so we can find matching HRRR profile files
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))

    if not h_files:
        # No HRRR profiles for this day - skip to the next VAD file
        continue

    # --- Load VAD (lidar) data for this day ---
    vad = nc.Dataset(vad_file)
    ws_vad = vad.variables['wind_speed'][:]      # wind speed profile, dims: (time, height)
    wd_vad = vad.variables['wind_direction'][:]  # wind direction profile, dims: (time, height)
    height = vad.variables['height'][:]          # height levels for the VAD profile
    base_t = int(vad.variables['base_time'][:])  # reference epoch time for this file
    time_vad = vad.variables['time'][:]          # offsets (seconds) from base_time for each time step
    vad.close()

    # Mask fill values (-9999.0) so they don't get treated as real data
    ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)

    # Convert VAD time offsets into absolute epoch seconds for matching against HRRR times
    vad_epoch = base_t + time_vad

    # Loop over each HRRR profile file for this day (typically multiple forecast/analysis times)
    for f in h_files:
        ds = nc.Dataset(f)
        hrrr_ws = ds.variables['wspd'][:]     # HRRR wind speed profile
        hrrr_dir = ds.variables['wdir'][:]    # HRRR wind direction profile
        hrrr_agl = ds.variables['height'][:]  # HRRR heights, meters above ground level
        et = int(ds.variables['time'][0])     # epoch time of this HRRR profile
        ds.close()

        # Find the VAD time step closest to this HRRR profile's time
        ti = np.argmin(np.abs(vad_epoch - et))
        # Skip this HRRR file if the nearest VAD time is more than 15 minutes (900 s) away
        if abs(vad_epoch[ti] - et) > 900:
            continue

        # Identify which VAD height levels have valid (non-masked) data at this time step
        valid = ~np.ma.getmaskarray(ws_vad[ti])
        if not valid.any():
            # No valid VAD data at this time - nothing to match against
            continue

        h_valid = height[valid]          # VAD heights that have valid data
        idx_valid = np.where(valid)[0]   # original indices of those valid heights

        # Loop over each HRRR height level
        for k in range(len(hrrr_agl)):
            # Restrict comparison to the 100-2000 m AGL layer
            if not (100 <= hrrr_agl[k] <= 2000):
                continue
            # Find the closest valid VAD height to this HRRR height
            j = np.argmin(np.abs(h_valid - hrrr_agl[k]))
            # Require the height match to be within 25 m, otherwise skip (too far apart to compare)
            if np.abs(h_valid[j] - hrrr_agl[k]) > 25:
                continue
            # Map back to the index in the original (unfiltered) VAD height array
            idx = idx_valid[j]

            # --- Wind speed: record the date (as a string) and the VAD - HRRR difference for this matched point ---
            dates_s.append(date)
            sdiff_m.append(ws_vad[ti, idx] - hrrr_ws[k])

            # --- Wind direction: only compare when both speeds exceed 2 m/s,
            # since direction is poorly defined / noisy at very low wind speeds ---
            if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0:
                # Wrap the direction difference into the range [-180, 180) degrees
                dd = ((wd_vad[ti, idx] - hrrr_dir[k] + 180) % 360) - 180
                dates_d.append(date)
                ddiff_m.append(dd)

# Convert accumulated lists to numpy arrays for easier grouping/plotting
dates_s = np.array(dates_s)
sdiff_m = np.array(sdiff_m)
dates_d = np.array(dates_d)
ddiff_m = np.array(ddiff_m)

# --- Summary counts ---
print(f"Total speed points: {len(sdiff_m)}")
print(f"Total direction points: {len(ddiff_m)}")

# compute daily means
def daily_means(date_arr, diff_arr):
    # For each unique date string present, compute the mean and standard deviation of the
    # difference values collected that day, and convert the date string into a real datetime
    # object (needed for proper date-axis plotting/formatting below).
    udates = sorted(set(date_arr))
    means, stds, dts = [], [], []
    for d in udates:
        sel = date_arr == d
        means.append(np.mean(diff_arr[sel]))
        stds.append(np.std(diff_arr[sel]))
        dts.append(datetime.strptime(d, '%Y%m%d'))
    return np.array(dts), np.array(means), np.array(stds)

# Compute daily stats separately for wind speed and wind direction differences
s_dt, s_mean, s_std = daily_means(dates_s, sdiff_m)
d_dt, d_mean, d_std = daily_means(dates_d, ddiff_m)

# --- plot: 2 stacked panels sharing the date x-axis ---
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Daily Mean Difference (VAD - HRRR), M2HATS', fontsize=16)

# --- Top panel: wind speed time series ---
# Point/line at the daily mean difference, with error bars showing +/- 1 standard deviation for that day
ax1.errorbar(s_dt, s_mean, yerr=s_std, fmt='o-', color='steelblue',
             ecolor='lightgray', capsize=3, markersize=4)
ax1.axhline(0, color='red', linewidth=1)  # reference line at zero error
ax1.set_ylabel('Wind Speed Diff (m/s)', fontsize=14)
ax1.set_title('Wind Speed', fontsize=15)
ax1.tick_params(labelsize=13)
ax1.grid(alpha=0.3)

# --- Bottom panel: wind direction time series ---
# Point/line at the daily mean difference, with error bars showing +/- 1 standard deviation for that day
ax2.errorbar(d_dt, d_mean, yerr=d_std, fmt='o-', color='indianred',
             ecolor='lightgray', capsize=3, markersize=4)
ax2.axhline(0, color='red', linewidth=1)  # reference line at zero error
ax2.set_ylabel('Wind Dir Diff (deg)', fontsize=14)
ax2.set_xlabel('Date', fontsize=14)
ax2.set_title('Wind Direction', fontsize=15)
ax2.tick_params(labelsize=13)
ax2.grid(alpha=0.3)

# format x-axis dates
# Show dates as e.g. "Jul 15", with a tick every 5 days, and rotate labels so they don't overlap
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
fig.autofmt_xdate(rotation=45)

plt.tight_layout()
plt.show()