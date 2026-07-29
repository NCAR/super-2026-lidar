import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# --- Accumulators across all files/times/heights ---
# u_hrrr_m / u_vad_m: matched HRRR / VAD zonal (east-west) wind component, one entry per matched point
# v_hrrr_m / v_vad_m: matched HRRR / VAD meridional (north-south) wind component, one entry per matched point
# dir_hrrr_m / dir_vad_m: matched HRRR / VAD wind direction, only kept when both speeds are above the
#                         calm-wind cutoff and the VAD direction value is valid
u_hrrr_m, u_vad_m, v_hrrr_m, v_vad_m = [], [], [], []
dir_hrrr_m, dir_vad_m = [], []

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
    u_vad = vad.variables['u'][:]                # zonal (east-west) wind component profile
    v_vad = vad.variables['v'][:]                # meridional (north-south) wind component profile
    ws_vad = vad.variables['wind_speed'][:]      # wind speed profile, used for the direction speed cutoff
    wd_vad = vad.variables['wind_direction'][:]  # wind direction profile
    height = vad.variables['height'][:]          # height levels for the VAD profile
    base_t = int(vad.variables['base_time'][:])  # reference epoch time for this file
    time_vad = vad.variables['time'][:]          # offsets (seconds) from base_time for each time step
    vad.close()

    # Mask fill values (-9999.0) so they don't get treated as real data
    u_vad = np.ma.masked_where(u_vad == -9999.0, u_vad)
    v_vad = np.ma.masked_where(v_vad == -9999.0, v_vad)
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

        # HRRR only provides speed/direction, so derive the u/v components from them
        # (meteorological convention: u is positive eastward, v is positive northward,
        # and wind direction is "from" direction, hence the negative signs)
        wdir_rad = np.radians(hrrr_dir)
        hrrr_u = -hrrr_ws * np.sin(wdir_rad)
        hrrr_v = -hrrr_ws * np.cos(wdir_rad)

        # Find the VAD time step closest to this HRRR profile's time
        ti = np.argmin(np.abs(vad_epoch - et))
        # Skip this HRRR file if the nearest VAD time is more than 15 minutes (900 s) away
        if abs(vad_epoch[ti] - et) > 900:
            continue

        # Identify which VAD height levels have valid (non-masked) u AND v data at this time step
        valid = ~np.ma.getmaskarray(u_vad[ti]) & ~np.ma.getmaskarray(v_vad[ti])
        if not valid.any():
            # No valid VAD data at this time - nothing to match against
            continue

        h_valid = height[valid]          # VAD heights that have valid u/v data
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

            # Extra safety check: even though `valid` already required non-masked u/v at this
            # time step, re-check the specific matched index in case of partial masking
            if np.ma.is_masked(u_vad[ti, idx]) or np.ma.is_masked(v_vad[ti, idx]):
                continue

            # --- Store the matched u and v component pairs ---
            u_hrrr_m.append(float(hrrr_u[k])); u_vad_m.append(float(u_vad[ti, idx]))
            v_hrrr_m.append(float(hrrr_v[k])); v_vad_m.append(float(v_vad[ti, idx]))

            # direction: only when both speeds > 2 m/s and direction is valid
            # (direction is poorly defined / noisy at very low wind speeds)
            if (hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0
                    and not np.ma.is_masked(wd_vad[ti, idx])):
                dir_hrrr_m.append(float(hrrr_dir[k]))
                dir_vad_m.append(float(wd_vad[ti, idx]))

# Convert accumulated lists to numpy arrays for easier math/plotting
u_hrrr_m = np.array(u_hrrr_m); u_vad_m = np.array(u_vad_m)
v_hrrr_m = np.array(v_hrrr_m); v_vad_m = np.array(v_vad_m)
dir_hrrr_m = np.array(dir_hrrr_m); dir_vad_m = np.array(dir_vad_m)

# --- Summary counts ---
print(f"u/v matched points: {len(u_hrrr_m)}")
print(f"direction matched points: {len(dir_hrrr_m)}")

# differences and stats
u_diff = u_vad_m - u_hrrr_m  # VAD - HRRR difference in the u (zonal) component
v_diff = v_vad_m - v_hrrr_m  # VAD - HRRR difference in the v (meridional) component
u_mad, u_sd = np.mean(np.abs(u_diff)), np.std(u_diff)  # mean absolute difference and standard deviation for u
v_mad, v_sd = np.mean(np.abs(v_diff)), np.std(v_diff)  # mean absolute difference and standard deviation for v
u_fit = np.polyfit(u_hrrr_m, u_vad_m, 1)  # linear fit of VAD vs HRRR u (slope, intercept) - not plotted, kept for diagnostics
v_fit = np.polyfit(v_hrrr_m, v_vad_m, 1)  # linear fit of VAD vs HRRR v (slope, intercept) - not plotted, kept for diagnostics

# circular direction difference, wrapped to +/-180
# (needed because direction is cyclic - e.g. 359 deg vs 1 deg is a 2-degree difference, not 358)
dir_diff = ((dir_vad_m - dir_hrrr_m + 180) % 360) - 180
dir_mad, dir_sd = np.mean(np.abs(dir_diff)), np.std(dir_diff)  # mean absolute difference and standard deviation for direction
dir_fit = np.polyfit(dir_hrrr_m, dir_vad_m, 1)  # linear fit of VAD vs HRRR direction (slope, intercept) - not plotted, kept for diagnostics

# Quick sanity-check print of the range of u/v differences seen
print(f"u_diff range: {u_diff.min():.1f} to {u_diff.max():.1f}")
print(f"v_diff range: {v_diff.min():.1f} to {v_diff.max():.1f}")

# --- plot: 2x3 grid (scatter top, histogram bottom; u, v, direction) ---
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('U, V, and Direction Comparison (M2HATS)', fontsize=17)

lim = 20  # axis limits (m/s) for the u/v scatter plots
# Use a symmetric histogram range wide enough to cover the largest u or v difference seen
dmax = max(np.abs(u_diff).max(), np.abs(v_diff).max())
uv_edges = np.linspace(-dmax, dmax, 80)  # shared bin edges for the u and v difference histograms

# --- top row: scatter plots (HRRR value vs VAD value, with a 1:1 reference line) ---
# u component scatter: perfect agreement would fall on the dashed 1:1 line
axes[0,0].scatter(u_hrrr_m, u_vad_m, s=3, color='black', alpha=0.3)
axes[0,0].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='1:1')
axes[0,0].set_xlim(-lim, lim); axes[0,0].set_ylim(-lim, lim)
axes[0,0].set_xlabel('HRRR u (m/s)'); axes[0,0].set_ylabel('VAD u (m/s)')
axes[0,0].set_title('U Component')
# Annotate with point count and summary error stats (mean absolute difference, standard deviation)
axes[0,0].text(0.05, 0.95, f"{len(u_hrrr_m)} pts\nmad: {u_mad:.2f}\nsd: {u_sd:.2f}",
               transform=axes[0,0].transAxes, va='top', fontsize=11)
axes[0,0].legend(fontsize=11, loc='lower right')

# v component scatter: perfect agreement would fall on the dashed 1:1 line
axes[0,1].scatter(v_hrrr_m, v_vad_m, s=3, color='black', alpha=0.3)
axes[0,1].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='1:1')
axes[0,1].set_xlim(-lim, lim); axes[0,1].set_ylim(-lim, lim)
axes[0,1].set_xlabel('HRRR v (m/s)'); axes[0,1].set_ylabel('VAD v (m/s)')
axes[0,1].set_title('V Component')
# Annotate with point count and summary error stats (mean absolute difference, standard deviation)
axes[0,1].text(0.05, 0.95, f"{len(v_hrrr_m)} pts\nmad: {v_mad:.2f}\nsd: {v_sd:.2f}",
               transform=axes[0,1].transAxes, va='top', fontsize=11)
axes[0,1].legend(fontsize=11, loc='lower right')

# direction scatter: fixed 0-360 degree axes, with a 1:1 reference line
axes[0,2].scatter(dir_hrrr_m, dir_vad_m, s=3, color='black', alpha=0.3)
axes[0,2].plot([0, 360], [0, 360], 'r--', linewidth=1, label='1:1')
axes[0,2].set_xlim(0, 360); axes[0,2].set_ylim(0, 360)
axes[0,2].set_xlabel('HRRR direction (deg)'); axes[0,2].set_ylabel('VAD direction (deg)')
axes[0,2].set_title('Wind Direction')
# Annotate with point count and summary error stats (mean absolute difference, standard deviation)
axes[0,2].text(0.05, 0.95, f"{len(dir_hrrr_m)} pts\nmad: {dir_mad:.1f}\nsd: {dir_sd:.1f}",
               transform=axes[0,2].transAxes, va='top', fontsize=11)
axes[0,2].legend(fontsize=11, loc='lower right')

# --- bottom row: difference histograms (VAD - HRRR, binned into counts) ---
# u difference histogram
axes[1,0].hist(u_diff, bins=uv_edges, color='steelblue', edgecolor='black')
axes[1,0].axvline(0, color='red', linewidth=1)  # reference line at zero error
axes[1,0].set_xlim(-dmax, dmax)
axes[1,0].set_xlabel('VAD - HRRR u (m/s)'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title('U Difference')

# v difference histogram
axes[1,1].hist(v_diff, bins=uv_edges, color='indianred', edgecolor='black')
axes[1,1].axvline(0, color='red', linewidth=1)  # reference line at zero error
axes[1,1].set_xlim(-dmax, dmax)
axes[1,1].set_xlabel('VAD - HRRR v (m/s)'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('V Difference')

# direction histogram uses the circular difference, fixed -180 to 180 range
dir_edges = np.linspace(-180, 180, 73)  # 5-degree bins
axes[1,2].hist(dir_diff, bins=dir_edges, color='seagreen', edgecolor='black')
axes[1,2].axvline(0, color='red', linewidth=1)  # reference line at zero error
axes[1,2].set_xlim(-180, 180)
axes[1,2].set_xlabel('VAD - HRRR direction (deg, circular)'); axes[1,2].set_ylabel('Count')
axes[1,2].set_title('Direction Difference')

plt.tight_layout()
plt.show()