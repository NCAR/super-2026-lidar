# -*- coding: utf-8 -*-
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

# -- paths --------------------------------------------------------------------
vad_base   = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
sonde_base = '/net/isf/radiosonde_archive/2023_m2hats/qc/ncdf_v1/'

# -- height filter (m AGL) ----------------------------------------------------
H_MIN, H_MAX = 100, 2000      # height range to compare
H_TOL        = 25             # max gate-to-level height difference (m)
T_TOL        = 30 * 60        # max time difference for a match (s) - 30 minutes
WS_MIN       = 2.0            # minimum wind speed for direction comparison (avoids noisy low-speed directions)

# -- accumulators -------------------------------------------------------------
# sonde_ws_m / vad_ws_m: matched radiosonde / VAD wind speed values, one entry per matched level
# sonde_dir_m / vad_dir_m: matched radiosonde / VAD wind direction values (only kept when both
#                          speeds exceed WS_MIN and the sonde direction value is valid)
sonde_ws_m, vad_ws_m   = [], []
sonde_dir_m, vad_dir_m = [], []

def units_to_epoch(units_str):
    """Parse a 'seconds since YYYY-MM-DD HH:MM:SS UTC' string to a Unix epoch."""
    # e.g. "seconds since 2023-07-23 17:27:36 UTC"
    # Split off everything after the word "since" to get the reference date/time string
    parts = units_str.strip().split('since')
    # Strip out the UTC/Z suffix so it can be parsed as a plain datetime string
    dt_str = parts[1].strip().replace(' UTC', '').replace(' Z', '')
    fmt = '%Y-%m-%d %H:%M:%S'
    dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

# -----------------------------------------------------------------------------
# Main loop: iterate over VAD daily files
# -----------------------------------------------------------------------------
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    # Extract the date string from the filename so we can find matching radiosonde launch files
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')

    # sondes are flat in sonde_base, filename contains the date
    # (there can be multiple launches on the same day, hence the glob pattern with a wildcard for launch time)
    s_files = sorted(glob.glob(os.path.join(sonde_base, 'NCAR_M2HATS_ISS1_RS41_v1_' + date + '_*_asc.nc')))
    if not s_files:
        # No radiosonde launches found for this day - skip to the next VAD file
        continue

    # -- read VAD --------------------------------------------------------------
    vad      = nc.Dataset(vad_file)
    ws_vad   = vad.variables['wind_speed'][:]        # shape (time, height)
    wd_vad   = vad.variables['wind_direction'][:]    # shape (time, height)
    height   = vad.variables['height'][:]            # AGL, metres
    base_t   = int(vad.variables['base_time'][:])    # reference epoch time for this file
    time_vad = vad.variables['time'][:]              # offsets (seconds) from base_time for each time step
    vad.close()

    # Mask fill values (-9999.0) so they don't get treated as real data
    ws_vad    = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad    = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
    vad_epoch = base_t + time_vad                    # Unix epoch per VAD sample

    # -- loop over sonde launches (ascending only) ----------------------------
    for sf in s_files:
        ds = nc.Dataset(sf)

        sonde_wspd = ds.variables['wspd'][:]  # radiosonde wind speed profile (ascending leg)
        sonde_wdir = ds.variables['wdir'][:]  # radiosonde wind direction profile
        sonde_alt  = ds.variables['alt'][:]   # MSL metres

        # parse launch epoch from the units string of launch_time
        # (the netCDF "units" attribute encodes the reference time, and the variable's value is
        # an offset in seconds from that reference - together they give the absolute launch epoch)
        lt_units   = ds.variables['launch_time'].units   # "seconds since YYYY-MM-DD HH:MM:SS UTC"
        lt_offset  = float(ds.variables['launch_time'][:])
        sonde_time = units_to_epoch(lt_units) + lt_offset

        ds.close()

        # Mask any NaN/invalid values in the sonde speed and direction arrays
        sonde_wspd = np.ma.masked_invalid(sonde_wspd)
        sonde_wdir = np.ma.masked_invalid(sonde_wdir)

        # convert sonde alt MSL -> AGL using first valid alt as surface elev
        # (the first valid altitude reading is taken as the launch/surface elevation, since the
        # sonde starts on the ground before ascending)
        valid_alt = sonde_alt[~np.ma.getmaskarray(np.ma.masked_invalid(sonde_alt))]
        if len(valid_alt) == 0:
            # No valid altitude data for this launch - can't convert to AGL, skip it
            continue
        surf_elev  = float(valid_alt[0])
        sonde_agl  = sonde_alt - surf_elev

        # -- find nearest VAD time sample -------------------------------------
        ti = np.argmin(np.abs(vad_epoch - sonde_time))
        if abs(vad_epoch[ti] - sonde_time) > T_TOL:
            # Nearest VAD sample is still too far in time from the sonde launch - report and skip
            print("  No VAD match for %s  (dt=%.0f min)" % (
                os.path.basename(sf), abs(vad_epoch[ti] - sonde_time) / 60.0))
            continue

        # Identify which VAD height levels have valid (non-masked) data at this time step
        valid_vad = ~np.ma.getmaskarray(ws_vad[ti])
        if not valid_vad.any():
            # No valid VAD data at this time - nothing to match against
            continue

        h_valid   = height[valid_vad]          # VAD heights that have valid data
        idx_valid = np.where(valid_vad)[0]     # original indices of those valid heights

        # -- match sonde levels to VAD gates ----------------------------------
        n_matched = 0
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
            # --- Store the matched sonde/VAD speed pair ---
            sonde_ws_m.append(float(sonde_wspd[k]))
            vad_ws_m.append(float(ws_vad[ti, idx]))
            n_matched += 1

            # --- Wind direction: only store when both speeds exceed WS_MIN and the sonde
            # direction reading is valid, since direction is noisy at low wind speeds ---
            if sonde_wspd[k] > WS_MIN and ws_vad[ti, idx] > WS_MIN:
                if not np.ma.is_masked(sonde_wdir[k]):
                    sonde_dir_m.append(float(sonde_wdir[k]))
                    vad_dir_m.append(float(wd_vad[ti, idx]))

        # Per-launch summary: surface elevation used, which VAD time step matched, and how many levels matched
        print("  %s  surf_elev=%.0f m  VAD ti=%d  matched %d levels" % (
            os.path.basename(sf), surf_elev, ti, n_matched))

# -----------------------------------------------------------------------------
# Convert to arrays
# -----------------------------------------------------------------------------
sonde_ws_m  = np.array(sonde_ws_m,  dtype=float)
vad_ws_m    = np.array(vad_ws_m,    dtype=float)
sonde_dir_m = np.array(sonde_dir_m, dtype=float)
vad_dir_m   = np.array(vad_dir_m,   dtype=float)

# --- Summary counts ---
print("\nMatched wind-speed points : %d" % len(sonde_ws_m))
print("Matched wind-dir  points  : %d" % len(sonde_dir_m))

# Sanity check: bail out early if nothing matched at all (likely a path/tolerance issue)
if len(sonde_ws_m) == 0:
    raise SystemExit("Zero matches - check sonde path, variable names, and height / time tolerances.")

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
# --- Wind speed error stats ---
sdiff  = vad_ws_m - sonde_ws_m         # VAD - sonde wind speed difference, one value per matched point
s_mad  = np.mean(np.abs(sdiff))        # mean absolute difference
s_sd   = np.std(sdiff)                 # standard deviation of the difference
s_bias = np.mean(sdiff)                # mean (signed) difference - shows systematic over/under-estimation
s_fit  = np.polyfit(sonde_ws_m, vad_ws_m, 1)  # linear fit of VAD vs sonde speed (slope, intercept)

# --- Wind direction error stats ---
# Drop any NaNs before computing direction differences
dmask  = ~np.isnan(vad_dir_m) & ~np.isnan(sonde_dir_m)
# Wrap the direction difference into the range [-180, 180) degrees to handle the 0/360 boundary correctly
cdiff  = ((vad_dir_m[dmask] - sonde_dir_m[dmask] + 180) % 360) - 180
d_mad  = np.mean(np.abs(cdiff))        # mean absolute difference (circular)
d_sd   = np.std(cdiff)                 # standard deviation of the circular difference
d_bias = np.mean(cdiff)                # mean (signed) circular difference - systematic direction bias
d_fit  = np.polyfit(sonde_dir_m[dmask], vad_dir_m[dmask], 1)  # linear fit of VAD vs sonde direction (slope, intercept)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
# Side-by-side scatter plots: sonde value vs VAD value, with a 1:1 reference line, for speed and direction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('M2HATS  23 Jul - 24 Sep 2023\nWindcube VAD vs Radiosonde', fontsize=16)

# -- wind speed ----------------------------------------------------------------
ax1.scatter(sonde_ws_m, vad_ws_m, s=4, color='steelblue', alpha=0.6, label='matched pairs')
ax1.plot([0, 20], [0, 20], 'k--', lw=0.8, label='1:1')
ax1.set_xlim(0, 20); ax1.set_ylim(0, 20)
ax1.set_xlabel('Radiosonde Wind Speed (m/s)', fontsize=20)
ax1.set_ylabel('Windcube VAD Wind Speed (m/s)', fontsize=20)
ax1.tick_params(labelsize=15)
# Annotate with point count, error stats (bias, mean absolute difference, standard deviation), and the linear fit
ax1.text(0.5, 19.0, "%d pts" % len(sonde_ws_m), fontsize=15)
ax1.text(0.5, 17.8, "bias: %+.2f  MAD: %.2f  SD: %.2f m/s" % (s_bias, s_mad, s_sd), fontsize=15)
ax1.text(0.5, 16.6, "fit: %.2fx + %.2f" % (s_fit[0], s_fit[1]), fontsize=15)
ax1.legend(fontsize=13, markerscale=2)

# -- wind direction ------------------------------------------------------------
ax2.scatter(sonde_dir_m[dmask], vad_dir_m[dmask], s=4, color='darkorange', alpha=0.6, label='matched pairs')
ax2.plot([0, 360], [0, 360], 'k--', lw=0.8, label='1:1')
ax2.set_xlim(0, 360); ax2.set_ylim(0, 360)
ax2.set_xlabel('Radiosonde Wind Direction (deg)', fontsize=20)
ax2.set_ylabel('Windcube VAD Wind Direction (deg)', fontsize=20)
ax2.tick_params(labelsize=15)
# Annotate with point count, error stats (bias, mean absolute difference, standard deviation), and the linear fit
ax2.text(10, 345, "%d pts  (both >%.1f m/s)" % (int(dmask.sum()), WS_MIN), fontsize=15)
ax2.text(10, 328, "bias: %+.1f  MAD: %.1f  SD: %.1f deg" % (d_bias, d_mad, d_sd), fontsize=15)
ax2.text(10, 311, "fit: %.2fx + %.1f" % (d_fit[0], d_fit[1]), fontsize=15)
ax2.legend(fontsize=13, markerscale=2)

plt.tight_layout()
# Save the figure to disk in addition to showing it interactively
plt.savefig('vad_vs_sonde.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: vad_vs_sonde.png")