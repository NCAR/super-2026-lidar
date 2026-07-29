# -*- coding: utf-8 -*-
"""
diurnal_mean_difference.py
--------------------------
Mean signed wind difference (VAD - radiosonde) as a function of the
diurnal cycle, binned by VAD sample hour (UTC, 24 hourly bins):
  top    : wind speed difference (m/s)
  bottom : wind direction difference (deg, circular)

Each matched level is assigned to the UTC hour of its VAD sample time,
then all matches across the whole campaign are averaged within each hour.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
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

# -- per-hour accumulators: lists of differences keyed by UTC hour 0..23 ------
# Every matched difference across the entire campaign is appended into the list for its
# corresponding UTC hour, regardless of which day it came from - these lists are reduced
# to per-hour mean/SD/count further below
ws_by_hour  = [[] for _ in range(24)]
dir_by_hour = [[] for _ in range(24)]

# -----------------------------------------------------------------------------
# Main matching loop
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

        # UTC hour of the matched VAD sample
        # this is the bin index (0-23) that every difference from this launch gets filed under below
        vad_hour = datetime.fromtimestamp(int(vad_epoch[ti]), tz=timezone.utc).hour

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

            # --- Wind speed: file the VAD - sonde difference into this match's UTC hour bin ---
            ws_by_hour[vad_hour].append(float(ws_vad[ti, idx]) - float(sonde_wspd[k]))

            # --- Wind direction: only compare when both speeds exceed WS_MIN and the sonde
            # direction reading is valid, since direction is noisy at low wind speeds ---
            if sonde_wspd[k] > WS_MIN and ws_vad[ti, idx] > WS_MIN:
                if not np.ma.is_masked(sonde_wdir[k]):
                    # Wrap the direction difference into the range [-180, 180) degrees
                    cd = ((float(wd_vad[ti, idx]) - float(sonde_wdir[k]) + 180) % 360) - 180
                    dir_by_hour[vad_hour].append(cd)

# -----------------------------------------------------------------------------
# Reduce to per-hour mean / SD / count
# -----------------------------------------------------------------------------
hours = np.arange(24)
# Pre-fill with NaN/0 so hours with no matches show as gaps rather than false zeros
ws_mean  = np.full(24, np.nan); ws_sd  = np.full(24, np.nan); ws_n  = np.zeros(24, dtype=int)
dir_mean = np.full(24, np.nan); dir_sd = np.full(24, np.nan); dir_n = np.zeros(24, dtype=int)

for h in range(24):
    if ws_by_hour[h]:
        a = np.array(ws_by_hour[h])
        ws_mean[h] = a.mean(); ws_sd[h] = a.std(); ws_n[h] = len(a)
    if dir_by_hour[h]:
        a = np.array(dir_by_hour[h])
        dir_mean[h] = a.mean(); dir_sd[h] = a.std(); dir_n[h] = len(a)

# --- Summary counts ---
total_ws  = int(ws_n.sum())
total_dir = int(dir_n.sum())
print("Total speed matches     : %d" % total_ws)
print("Total direction matches : %d" % total_dir)
if total_ws == 0:
    raise SystemExit("Zero matches - check paths / tolerances.")

# Campaign-wide (all-hours) mean, ignoring any hours with no data
ws_overall  = np.nanmean(ws_mean)
dir_overall = np.nanmean(dir_mean)

# -----------------------------------------------------------------------------
# Plot - stacked diurnal cycle
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle('M2HATS  Diurnal Mean Wind Difference (VAD - Radiosonde), by UTC hour',
             fontsize=16)

# -- speed --------------------------------------------------------------------
ax1.axhline(0, color='gray', lw=0.8)  # reference line at zero bias
# Dotted reference line showing the all-hours mean speed bias
ax1.axhline(ws_overall, color='steelblue', ls=':', lw=1.2,
            label='all-hours mean (%+.2f m/s)' % ws_overall)
# Hourly mean speed bias, with error bars showing +/- 1 standard deviation for that hour
ax1.errorbar(hours, ws_mean, yerr=ws_sd, fmt='o-', color='steelblue',
             ecolor='lightsteelblue', elinewidth=1.2, capsize=3, ms=6,
             label='hourly mean +/- 1 SD')
ax1.set_ylabel('Speed difference (m/s)', fontsize=13)
ax1.set_title('Wind Speed Bias', fontsize=14)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3)
# annotate sample counts above each point
# (shows how many matched points contributed to each hourly mean, so sparse hours are obvious)
for h in hours:
    if ws_n[h] > 0:
        ax1.annotate("%d" % ws_n[h], (h, ws_mean[h]),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8, color='dimgray')

# -- direction ----------------------------------------------------------------
ax2.axhline(0, color='gray', lw=0.8)  # reference line at zero bias
# Dotted reference line showing the all-hours mean direction bias
ax2.axhline(dir_overall, color='darkorange', ls=':', lw=1.2,
            label='all-hours mean (%+.1f deg)' % dir_overall)
# Hourly mean direction bias, with error bars showing +/- 1 standard deviation for that hour
ax2.errorbar(hours, dir_mean, yerr=dir_sd, fmt='o-', color='darkorange',
             ecolor='moccasin', elinewidth=1.2, capsize=3, ms=6,
             label='hourly mean +/- 1 SD')
ax2.set_ylabel('Direction difference (deg)', fontsize=13)
ax2.set_xlabel('Hour of day (UTC)', fontsize=13)
ax2.set_title('Wind Direction Bias', fontsize=14)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(alpha=0.3)
# annotate sample counts above each direction point, same purpose as the speed panel above
for h in hours:
    if dir_n[h] > 0:
        ax2.annotate("%d" % dir_n[h], (h, dir_mean[h]),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8, color='dimgray')

# Tick every 2 hours across the 24-hour day, with a bit of padding on each side
ax2.set_xticks(np.arange(0, 24, 2))
ax2.set_xlim(-0.5, 23.5)

plt.tight_layout()
# Save the figure to disk in addition to showing it interactively
plt.savefig('diurnal_mean_difference.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: diurnal_mean_difference.png")

# -----------------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------------
# Print a table of speed and direction bias/SD/counts for each UTC hour, showing "nan"
# for any hour that had zero matches
print("\n%4s %10s %8s %6s   %10s %8s %6s" % (
    "Hour", "WS_bias", "WS_sd", "WS_n", "DIR_bias", "DIR_sd", "DIR_n"))
for h in range(24):
    wsb = "%+9.2f" % ws_mean[h]  if ws_n[h]  else "      nan"
    wss = "%8.2f"  % ws_sd[h]    if ws_n[h]  else "     nan"
    dib = "%+9.1f" % dir_mean[h] if dir_n[h] else "      nan"
    dis = "%8.1f"  % dir_sd[h]   if dir_n[h] else "     nan"
    print("%4d %s %s %6d   %s %s %6d" % (h, wsb, wss, ws_n[h], dib, dis, dir_n[h]))