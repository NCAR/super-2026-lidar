import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# THIS is roughly the same as plot-by-uv-dir.py, it was just a previous iteration

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# --- Accumulators across all files/times/heights ---
# u_hrrr_m / u_vad_m: matched HRRR / VAD zonal (east-west) wind component, one entry per matched point
# v_hrrr_m / v_vad_m: matched HRRR / VAD meridional (north-south) wind component, one entry per matched point
u_hrrr_m, u_vad_m, v_hrrr_m, v_vad_m = [], [], [], []

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
    u_vad = vad.variables['u'][:]  # zonal (east-west) wind component profile
    v_vad = vad.variables['v'][:]  # meridional (north-south) wind component profile
    height = vad.variables['height'][:]          # height levels for the VAD profile
    base_t = int(vad.variables['base_time'][:])  # reference epoch time for this file
    time_vad = vad.variables['time'][:]          # offsets (seconds) from base_time for each time step
    vad.close()

    # Mask fill values (-9999.0) so they don't get treated as real data
    u_vad = np.ma.masked_where(u_vad == -9999.0, u_vad)
    v_vad = np.ma.masked_where(v_vad == -9999.0, v_vad)

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

        # derive HRRR u/v from speed and direction (meteorological convention)
        # (u is positive eastward, v is positive northward; the negative signs account for
        # wind direction being reported as the "from" direction)
        wdir_rad = np.radians(hrrr_dir)
        hrrr_u = -hrrr_ws * np.sin(wdir_rad)
        hrrr_v = -hrrr_ws * np.cos(wdir_rad)

        # Find the VAD time step closest to this HRRR profile's time
        ti = np.argmin(np.abs(vad_epoch - et))
        # Skip this HRRR file if the nearest VAD time is more than 15 minutes (900 s) away
        if abs(vad_epoch[ti] - et) > 900:
            continue

        # valid where both u and v are unmasked
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

# Convert accumulated lists to numpy arrays for easier math/plotting
u_hrrr_m = np.array(u_hrrr_m); u_vad_m = np.array(u_vad_m)
v_hrrr_m = np.array(v_hrrr_m); v_vad_m = np.array(v_vad_m)

# --- Summary count ---
print(f"Matched points: {len(u_hrrr_m)}")

# differences and stats
u_diff = u_vad_m - u_hrrr_m  # VAD - HRRR difference in the u (zonal) component
v_diff = v_vad_m - v_hrrr_m  # VAD - HRRR difference in the v (meridional) component
u_mad, u_sd = np.mean(np.abs(u_diff)), np.std(u_diff)  # mean absolute difference and standard deviation for u
v_mad, v_sd = np.mean(np.abs(v_diff)), np.std(v_diff)  # mean absolute difference and standard deviation for v
u_fit = np.polyfit(u_hrrr_m, u_vad_m, 1)  # linear fit of VAD vs HRRR u (slope, intercept), shown in the scatter annotation
v_fit = np.polyfit(v_hrrr_m, v_vad_m, 1)  # linear fit of VAD vs HRRR v (slope, intercept), shown in the scatter annotation

# report actual ranges so we can size the histograms
print(f"u_diff range: {u_diff.min():.1f} to {u_diff.max():.1f}")
print(f"v_diff range: {v_diff.min():.1f} to {v_diff.max():.1f}")

# --- plot: 2x2 grid (scatter on top, difference histogram on bottom; u left, v right) ---
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle('U and V Wind Component Comparison (M2HATS)', fontsize=17)

lim = 20  # axis limit for scatter (m/s)

# --- top-left: u scatter (HRRR vs VAD, with 1:1 reference line) ---
axes[0,0].scatter(u_hrrr_m, u_vad_m, s=3, color='black', alpha=0.3)
axes[0,0].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='1:1')
axes[0,0].set_xlim(-lim, lim); axes[0,0].set_ylim(-lim, lim)
axes[0,0].set_xlabel('HRRR u (m/s)'); axes[0,0].set_ylabel('VAD u (m/s)')
axes[0,0].set_title('U Component')
# Annotate with point count, error stats (mad, sd), and the linear fit equation
axes[0,0].text(0.05, 0.95, f"{len(u_hrrr_m)} pts\nmad: {u_mad:.2f}\nsd: {u_sd:.2f}\nfit: {u_fit[0]:.2f}x+{u_fit[1]:.2f}",
               transform=axes[0,0].transAxes, va='top', fontsize=12)
axes[0,0].legend(fontsize=12, loc='lower right')

# --- top-right: v scatter (HRRR vs VAD, with 1:1 reference line) ---
axes[0,1].scatter(v_hrrr_m, v_vad_m, s=3, color='black', alpha=0.3)
axes[0,1].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='1:1')
axes[0,1].set_xlim(-lim, lim); axes[0,1].set_ylim(-lim, lim)
axes[0,1].set_xlabel('HRRR v (m/s)'); axes[0,1].set_ylabel('VAD v (m/s)')
axes[0,1].set_title('V Component')
# Annotate with point count, error stats (mad, sd), and the linear fit equation
axes[0,1].text(0.05, 0.95, f"{len(v_hrrr_m)} pts\nmad: {v_mad:.2f}\nsd: {v_sd:.2f}\nfit: {v_fit[0]:.2f}x+{v_fit[1]:.2f}",
               transform=axes[0,1].transAxes, va='top', fontsize=12)
axes[0,1].legend(fontsize=12, loc='lower right')

# common symmetric range covering both components' extremes
dmax = max(np.abs(u_diff).max(), np.abs(v_diff).max())
edges = np.linspace(-dmax, dmax, 80)  # 80 bins across the full range

# --- bottom-left: u difference histogram (VAD - HRRR u, binned into counts) ---
axes[1,0].hist(u_diff, bins=edges, color='steelblue', edgecolor='black')
axes[1,0].axvline(0, color='red', linewidth=1)  # reference line at zero error
axes[1,0].set_xlim(-dmax, dmax)
axes[1,0].set_xlabel('VAD - HRRR u (m/s)'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title('U Difference')

# --- bottom-right: v difference histogram (VAD - HRRR v, binned into counts) ---
axes[1,1].hist(v_diff, bins=edges, color='indianred', edgecolor='black')
axes[1,1].axvline(0, color='red', linewidth=1)  # reference line at zero error
axes[1,1].set_xlim(-dmax, dmax)
axes[1,1].set_xlabel('VAD - HRRR v (m/s)'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('V Difference')

plt.tight_layout()
plt.show()