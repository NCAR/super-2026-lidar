# -*- coding: utf-8 -*-
"""
histogram_outliers_by_pressure.py
---------------------------------
Histograms of total matched points and outliers, binned by pressure
(fixed 50 hPa bands), for VAD-vs-radiosonde wind speed and direction.

Outlier definitions:
  * wind speed     : |VAD - sonde| > 2.0 m/s
  * wind direction : circular |VAD - sonde| > 30 deg
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

# -- outlier thresholds -------------------------------------------------------
WS_OUTLIER  = 2.0             # m/s
DIR_OUTLIER = 30.0            # deg

# -- pressure binning ---------------------------------------------------------
P_BIN       = 50.0            # hPa band width

# -- accumulators (now carry pressure with each matched pair) -----------------
# ws_diff_list / ws_pres_list: |VAD - sonde| wind speed difference and matching pressure (hPa),
#                              one entry per matched point
# dir_diff_list / dir_pres_list: circular |VAD - sonde| direction difference and matching pressure,
#                                 one entry per matched point (only kept when both speeds are above WS_MIN)
ws_diff_list, ws_pres_list   = [], []   # |speed diff| and pressure for all speed matches
dir_diff_list, dir_pres_list = [], []   # circular dir diff and pressure for dir matches

def units_to_epoch(units_str):
    # Parse a netCDF "seconds since YYYY-MM-DD HH:MM:SS UTC" units string into a Unix epoch,
    # so it can be added to the variable's offset value to get an absolute timestamp
    parts = units_str.strip().split('since')
    dt_str = parts[1].strip().replace(' UTC', '').replace(' Z', '')
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

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
        sonde_alt  = ds.variables['alt'][:]            # MSL metres
        sonde_pres = ds.variables['pres'][:]           # hPa, used for pressure binning below
        # Parse the launch epoch from the launch_time units string plus its offset value
        lt_units   = ds.variables['launch_time'].units
        lt_offset  = float(ds.variables['launch_time'][:])
        sonde_time = units_to_epoch(lt_units) + lt_offset
        ds.close()

        # Mask any NaN/invalid values in the sonde speed, direction, and pressure arrays
        sonde_wspd = np.ma.masked_invalid(sonde_wspd)
        sonde_wdir = np.ma.masked_invalid(sonde_wdir)
        sonde_pres = np.ma.masked_invalid(sonde_pres)

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
            if np.ma.is_masked(sonde_wspd[k]) or np.ma.is_masked(sonde_pres[k]):
                # Skip sonde levels with no valid wind speed or pressure reading
                continue

            # Find the closest valid VAD height to this sonde level's height
            j = np.argmin(np.abs(h_valid - sonde_agl[k]))
            # Require the height match to be within H_TOL, otherwise skip (too far apart to compare)
            if np.abs(h_valid[j] - sonde_agl[k]) > H_TOL:
                continue
            # Map back to the index in the original (unfiltered) VAD height array
            idx = idx_valid[j]

            # --- Wind speed: store the absolute VAD - sonde difference and the pressure at this level ---
            p = float(sonde_pres[k])
            ws_diff_list.append(abs(float(ws_vad[ti, idx]) - float(sonde_wspd[k])))
            ws_pres_list.append(p)

            # --- Wind direction: only store when both speeds exceed WS_MIN and the sonde
            # direction reading is valid, since direction is noisy at low wind speeds ---
            if sonde_wspd[k] > WS_MIN and ws_vad[ti, idx] > WS_MIN:
                if not np.ma.is_masked(sonde_wdir[k]):
                    # Wrap the direction difference into the range [-180, 180) degrees, then take the
                    # absolute value since we only need magnitude for the outlier/histogram logic below
                    cd = ((float(wd_vad[ti, idx]) - float(sonde_wdir[k]) + 180) % 360) - 180
                    dir_diff_list.append(abs(cd))
                    dir_pres_list.append(p)

# -----------------------------------------------------------------------------
# Convert to arrays
# -----------------------------------------------------------------------------
ws_diff  = np.array(ws_diff_list);  ws_pres  = np.array(ws_pres_list)
dir_diff = np.array(dir_diff_list); dir_pres = np.array(dir_pres_list)

# --- Summary counts ---
print("Total speed matches     : %d" % len(ws_diff))
print("Total direction matches : %d" % len(dir_diff))
if len(ws_diff) == 0:
    raise SystemExit("Zero matches - check paths / tolerances.")

# -----------------------------------------------------------------------------
# Build pressure bins (50 hPa bands spanning the matched data)
# -----------------------------------------------------------------------------
# Combine speed and direction pressures (if any direction matches exist) so both histograms
# share the same bin edges, spanning the full observed pressure range
all_pres = np.concatenate([ws_pres, dir_pres]) if len(dir_pres) else ws_pres
p_lo = np.floor(all_pres.min() / P_BIN) * P_BIN  # round down to the nearest P_BIN boundary
p_hi = np.ceil(all_pres.max()  / P_BIN) * P_BIN  # round up to the nearest P_BIN boundary
edges = np.arange(p_lo, p_hi + P_BIN, P_BIN)
centers = (edges[:-1] + edges[1:]) / 2.0  # bin midpoints, used as the y-axis positions in the plot below

# total and outlier counts per bin
# For each pressure bin: how many matched points fall in it, and of those, how many exceed
# the outlier threshold for that variable
ws_total,  _ = np.histogram(ws_pres, bins=edges)
ws_out,    _ = np.histogram(ws_pres[ws_diff > WS_OUTLIER], bins=edges)
dir_total, _ = np.histogram(dir_pres, bins=edges)
dir_out,   _ = np.histogram(dir_pres[dir_diff > DIR_OUTLIER], bins=edges)

# -----------------------------------------------------------------------------
# Plot - horizontal bars (pressure decreasing upward, atmosphere-style)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
fig.suptitle('M2HATS - VAD vs Radiosonde:  matched points & outliers by pressure',
             fontsize=16)

bar_h = P_BIN * 0.8  # slightly narrower than the full bin width, to leave a visible gap between bars

# -- wind speed ----------------------------------------------------------------
# Solid bars: total matched speed points in each pressure bin
ax1.barh(centers, ws_total, height=bar_h, color='lightsteelblue',
         edgecolor='steelblue', label='total matched')
# Hatched outline bars (no fill): the subset of those points that are speed outliers,
# drawn on the same bins so the outlier count can be visually compared to the total
ax1.barh(centers, ws_out, height=bar_h, color='none',
         edgecolor='crimson', hatch='///', linewidth=1.2,
         label='outliers (>%.1f m/s)' % WS_OUTLIER)
ax1.set_xlabel('Number of matched points', fontsize=13)
ax1.set_ylabel('Pressure (hPa)', fontsize=13)
ax1.set_title('Wind Speed', fontsize=14)
ax1.invert_yaxis()                       # high pressure (low alt) at bottom
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# annotate outlier percentage per bin
# Places a "% outliers" label just past the end of each total bar, for bins that have any matches
for c, t, o in zip(centers, ws_total, ws_out):
    if t > 0:
        ax1.text(t + ws_total.max() * 0.01, c, "%.0f%%" % (100.0 * o / t),
                 va='center', fontsize=9, color='crimson')

# -- wind direction ------------------------------------------------------------
# Solid bars: total matched direction points in each pressure bin
ax2.barh(centers, dir_total, height=bar_h, color='moccasin',
         edgecolor='darkorange', label='total matched')
# Hatched outline bars (no fill): the subset of those points that are direction outliers
ax2.barh(centers, dir_out, height=bar_h, color='none',
         edgecolor='crimson', hatch='///', linewidth=1.2,
         label='outliers (>%.0f deg)' % DIR_OUTLIER)
ax2.set_xlabel('Number of matched points', fontsize=13)
ax2.set_title('Wind Direction', fontsize=14)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=11)
ax2.grid(axis='x', alpha=0.3)

# annotate outlier percentage per bin, same purpose as the speed panel above
for c, t, o in zip(centers, dir_total, dir_out):
    if t > 0:
        ax2.text(t + dir_total.max() * 0.01, c, "%.0f%%" % (100.0 * o / t),
                 va='center', fontsize=9, color='crimson')

plt.tight_layout()
# Save the figure to disk in addition to showing it interactively
plt.savefig('outliers_by_pressure.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: outliers_by_pressure.png")

# -----------------------------------------------------------------------------
# Text summary table
# -----------------------------------------------------------------------------
# Print a per-pressure-band table of total/outlier counts and outlier percentage,
# for both wind speed and wind direction
print("\n%-16s %8s %8s %8s   %8s %8s %8s" % (
    "P-band (hPa)", "WS_tot", "WS_out", "WS_%", "DIR_tot", "DIR_out", "DIR_%"))
for i in range(len(centers)):
    lo, hi = edges[i], edges[i + 1]
    wsp = 100.0 * ws_out[i]  / ws_total[i]  if ws_total[i]  else 0.0
    dip = 100.0 * dir_out[i] / dir_total[i] if dir_total[i] else 0.0
    print("%6.0f - %6.0f %8d %8d %7.0f%%   %8d %8d %7.0f%%" % (
        lo, hi, ws_total[i], ws_out[i], wsp, dir_total[i], dir_out[i], dip))