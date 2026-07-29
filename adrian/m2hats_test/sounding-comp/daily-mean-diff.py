# -*- coding: utf-8 -*-
"""
daily_mean_difference_timeseries.py
-----------------------------------
Daily mean signed wind difference (VAD - radiosonde) over the M2HATS
field project, plotted as a stacked time series:
  top    : wind speed difference (m/s)
  bottom : wind direction difference (deg, circular)

A "day" is keyed off the VAD daily file date. All matched levels for
that day are averaged into a single point.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob, os, warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

# -- paths --------------------------------------------------------------------
vad_base   = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
sonde_base = '/net/isf/radiosonde_archive/2023_m2hats/qc/ncdf_v1/'

# -- match parameters ---------------------------------------------------------
H_MIN, H_MAX = 100, 2000      # height range to compare (m AGL)
H_TOL        = 25             # max gate-to-level height difference (m)
T_TOL        = 30 * 60        # max time difference for a match (s) - 30 minutes
WS_MIN       = 2.0            # minimum wind speed for direction comparison (avoids noisy low-speed directions)

def units_to_epoch(units_str):
    # Parse a netCDF "seconds since YYYY-MM-DD HH:MM:SS UTC" units string into a Unix epoch,
    # so it can be added to the variable's offset value to get an absolute timestamp
    parts = units_str.strip().split('since')
    dt_str = parts[1].strip().replace(' UTC', '').replace(' Z', '')
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

# -- per-day accumulators -----------------------------------------------------
# One entry per day that has at least one matched point (unlike the point-level
# accumulators in other scripts, these hold a single summary value per day)
days        = []     # datetime (date) for each day with matches
ws_bias     = []     # daily mean (VAD - sonde) speed
ws_bias_sd  = []     # daily std of speed difference
ws_n        = []     # daily count (speed)
dir_bias    = []     # daily mean circular (VAD - sonde) direction
dir_bias_sd = []     # daily std of direction difference
dir_n       = []     # daily count (direction)

# -----------------------------------------------------------------------------
# Main loop: one iteration per VAD daily file
# -----------------------------------------------------------------------------
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    # Extract the date string from the filename so we can find matching radiosonde launch files
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')

    # sondes are flat in sonde_base, filename contains the date; there can be multiple
    # launches per day, hence the wildcard for launch time in the glob pattern
    s_files = sorted(glob.glob(os.path.join(sonde_base, 'NCAR_M2HATS_ISS1_RS41_v1_' + date + '_*_asc.nc')))
    if not s_files:
        # No radiosonde launches found for this day - skip to the next VAD file
        continue

    # --- Load VAD (lidar) data for this day ---
    vad      = nc.Dataset(vad_file)
    ws_vad   = vad.variables['wind_speed'][:]      # wind speed profile, dims: (time, height)
    wd_vad   = vad.variables['wind_direction'][:]  # wind direction profile, dims: (time, height)
    height   = vad.variables['height'][:]          # height levels for the VAD profile, AGL metres
    base_t   = int(vad.variables['base_time'][:])  # reference epoch time for this file
    time_vad = vad.variables['time'][:]            # offsets (seconds) from base_time for each time step
    vad.close()

    # Mask fill values (-9999.0) so they don't get treated as real data
    ws_vad    = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad    = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
    vad_epoch = base_t + time_vad  # Unix epoch per VAD sample

    # collect every matched difference for this day
    # (these are per-point differences that get averaged into a single daily value further below)
    day_ws_diff  = []
    day_dir_diff = []

    # Loop over each radiosonde launch file for this day
    for sf in s_files:
        ds = nc.Dataset(sf)
        sonde_wspd = ds.variables['wspd'][:]  # radiosonde wind speed profile (ascending leg)
        sonde_wdir = ds.variables['wdir'][:]  # radiosonde wind direction profile
        sonde_alt  = ds.variables['alt'][:]   # MSL metres
        # Parse the launch epoch from the launch_time units string plus its offset value
        lt_units   = ds.variables['launch_time'].units
        lt_offset  = float(ds.variables['launch_time'][:])
        sonde_time = units_to_epoch(lt_units) + lt_offset
        ds.close()

        # Mask any NaN/invalid values in the sonde speed and direction arrays
        sonde_wspd = np.ma.masked_invalid(sonde_wspd)
        sonde_wdir = np.ma.masked_invalid(sonde_wdir)

        # convert sonde alt MSL -> AGL using first valid alt as surface elev
        # (the first valid altitude reading is taken as the launch/surface elevation)
        valid_alt = sonde_alt[~np.ma.getmaskarray(np.ma.masked_invalid(sonde_alt))]
        if len(valid_alt) == 0:
            # No valid altitude data for this launch - can't convert to AGL, skip it
            continue
        surf_elev = float(valid_alt[0])
        sonde_agl = sonde_alt - surf_elev

        # Find the VAD time step closest to this sonde launch time
        ti = np.argmin(np.abs(vad_epoch - sonde_time))
        if abs(vad_epoch[ti] - sonde_time) > T_TOL:
            # Nearest VAD sample is still too far in time from the sonde launch - skip this launch
            continue

        # Identify which VAD height levels have valid (non-masked) data at this time step
        valid_vad = ~np.ma.getmaskarray(ws_vad[ti])
        if not valid_vad.any():
            # No valid VAD data at this time - nothing to match against
            continue
        h_valid   = height[valid_vad]          # VAD heights that have valid data
        idx_valid = np.where(valid_vad)[0]     # original indices of those valid heights

        # Loop over each sonde level and try to match it to a nearby VAD height gate
        for k in range(len(sonde_agl)):
            # Restrict comparison to the H_MIN-H_MAX AGL layer
            if not (H_MIN <= sonde_agl[k] <= H_MAX):
                continue
            if np.ma.is_masked(sonde_wspd[k]):
                # Skip sonde levels with no valid wind speed reading
                continue
            # Find the closest valid VAD height to this sonde level's height
            j = np.argmin(np.abs(h_valid - sonde_agl[k]))
            # Require the height match to be within H_TOL, otherwise skip (too far apart to compare)
            if np.abs(h_valid[j] - sonde_agl[k]) > H_TOL:
                continue
            # Map back to the index in the original (unfiltered) VAD height array
            idx = idx_valid[j]

            # --- Wind speed: record the VAD - sonde difference for this matched point ---
            day_ws_diff.append(float(ws_vad[ti, idx]) - float(sonde_wspd[k]))

            # --- Wind direction: only compare when both speeds exceed WS_MIN and the sonde
            # direction reading is valid, since direction is noisy at low wind speeds ---
            if sonde_wspd[k] > WS_MIN and ws_vad[ti, idx] > WS_MIN:
                if not np.ma.is_masked(sonde_wdir[k]):
                    # Wrap the direction difference into the range [-180, 180) degrees
                    cd = ((float(wd_vad[ti, idx]) - float(sonde_wdir[k]) + 180) % 360) - 180
                    day_dir_diff.append(cd)

    if not day_ws_diff:
        # No matched points at all for this day - don't add an entry for it
        continue

    # Record this day's date and the summary stats for its matched speed differences
    day_dt = datetime.strptime(date, '%Y%m%d')
    days.append(day_dt)

    dws = np.array(day_ws_diff)
    ws_bias.append(dws.mean())    # daily mean signed speed difference
    ws_bias_sd.append(dws.std())  # daily standard deviation of the speed difference
    ws_n.append(len(dws))         # number of matched speed points that day

    # Direction may have fewer (or zero) matches than speed, since it requires the WS_MIN cutoff
    if day_dir_diff:
        ddir = np.array(day_dir_diff)
        dir_bias.append(ddir.mean())    # daily mean circular direction difference
        dir_bias_sd.append(ddir.std())  # daily standard deviation of the direction difference
        dir_n.append(len(ddir))         # number of matched direction points that day
    else:
        # No direction matches this day (e.g. calm winds) - record as NaN rather than dropping the day,
        # so the speed panel still shows this date
        dir_bias.append(np.nan)
        dir_bias_sd.append(np.nan)
        dir_n.append(0)

#days        = np.array(days)
days = mdates.date2num(days)   # convert to matplotlib date numbers
ws_bias     = np.array(ws_bias)
ws_bias_sd  = np.array(ws_bias_sd)
dir_bias    = np.array(dir_bias)
dir_bias_sd = np.array(dir_bias_sd)

# --- Summary count ---
print("Days with matches: %d" % len(days))
if len(days) == 0:
    raise SystemExit("Zero days matched - check paths / tolerances.")

# overall (campaign-wide) means for reference lines
# nanmean is used so that days with no direction matches (NaN) don't break the direction average
ws_overall  = np.nanmean(ws_bias)
dir_overall = np.nanmean(dir_bias)
print("Campaign mean speed bias     : %+.2f m/s" % ws_overall)
print("Campaign mean direction bias : %+.2f deg" % dir_overall)

# -----------------------------------------------------------------------------
# Plot - stacked time series
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle('M2HATS  Daily Mean Wind Difference (VAD - Radiosonde)', fontsize=16)

# -- speed --------------------------------------------------------------------
ax1.axhline(0, color='gray', lw=0.8)  # reference line at zero bias
# Dotted reference line showing the campaign-wide mean speed bias
ax1.axhline(ws_overall, color='steelblue', ls=':', lw=1.2,
            label='campaign mean (%+.2f m/s)' % ws_overall)
# Daily mean speed bias, with error bars showing +/- 1 standard deviation for that day
ax1.errorbar(days, ws_bias, yerr=ws_bias_sd, fmt='o-', color='steelblue',
             ecolor='lightsteelblue', elinewidth=1.2, capsize=3, ms=5,
             label='daily mean +/- 1 SD')
ax1.set_ylabel('Speed difference (m/s)', fontsize=13)
ax1.set_title('Wind Speed Bias', fontsize=14)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3)

# -- direction ----------------------------------------------------------------
ax2.axhline(0, color='gray', lw=0.8)  # reference line at zero bias
# Dotted reference line showing the campaign-wide mean direction bias
ax2.axhline(dir_overall, color='darkorange', ls=':', lw=1.2,
            label='campaign mean (%+.1f deg)' % dir_overall)
# Daily mean direction bias, with error bars showing +/- 1 standard deviation for that day
# (days with no direction matches show as gaps, since their value is NaN)
ax2.errorbar(days, dir_bias, yerr=dir_bias_sd, fmt='o-', color='darkorange',
             ecolor='moccasin', elinewidth=1.2, capsize=3, ms=5,
             label='daily mean +/- 1 SD')
ax2.set_ylabel('Direction difference (deg)', fontsize=13)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_title('Wind Direction Bias', fontsize=14)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(alpha=0.3)

# date formatting on shared x-axis
# Tick mark on every Monday, labeled as e.g. "24 Jul", with rotated labels to avoid overlap
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.autofmt_xdate(rotation=45)

plt.tight_layout()
# Save the figure to disk in addition to showing it interactively
plt.savefig('daily_mean_difference.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: daily_mean_difference.png")

# -----------------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------------
# Print a per-day table of speed and direction bias/SD/counts, reconstructing the date
# string from the matplotlib date number stored in `days`
print("\n%-12s %10s %8s %6s   %10s %8s %6s" % (
    "Date", "WS_bias", "WS_sd", "WS_n", "DIR_bias", "DIR_sd", "DIR_n"))
for i in range(len(days)):
    print("%-12s %+9.2f %8.2f %6d   %+9.1f %8.1f %6d" % (
        mdates.num2date(days[i]).strftime('%Y-%m-%d'),
        ws_bias[i], ws_bias_sd[i], ws_n[i],
        dir_bias[i], dir_bias_sd[i], dir_n[i]))