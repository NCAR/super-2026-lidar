import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# collect wind speed and directional difference per matched point
# ws_m:    VAD wind speed at each matched point (used as the x-axis, to look at low-speed behavior)
# ddiff_m: VAD - HRRR wind direction difference at each matched point
ws_m, ddiff_m = [], []

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

            # directional difference vs. VAD wind speed (no 2 m/s cutoff here -
            # we want to SEE the low-speed behavior)
            # Wrap the direction difference into the range [-180, 180) degrees to handle the 0/360 boundary correctly
            dd = ((wd_vad[ti, idx] - hrrr_dir[k] + 180) % 360) - 180
            ws_m.append(ws_vad[ti, idx])
            ddiff_m.append(dd)

# Convert accumulated lists to numpy arrays for easier plotting/binning
ws_m = np.array(ws_m)
ddiff_m = np.array(ddiff_m)

# --- Summary count ---
print(f"Total points: {len(ws_m)}")

# --- plot ---
# Scatter of |direction difference| against VAD wind speed, with a red dashed line marking the
# 2 m/s cutoff normally used elsewhere to exclude calm-wind direction comparisons - here it's
# left in as a visual reference rather than a filter
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle('Directional Difference vs. Wind Speed (M2HATS)', fontsize=18)

ax.scatter(ws_m, np.abs(ddiff_m), s=3, color='black', alpha=0.3)
ax.axvline(2.0, color='red', linewidth=1, linestyle='--', label='2 m/s cutoff')
ax.set_xlabel('VAD Wind Speed (m/s)', fontsize=16)
ax.set_ylabel('|VAD - HRRR Direction| (deg)', fontsize=16)
ax.set_xlim(0, 20)
ax.set_ylim(0, 180)
ax.tick_params(labelsize=15)
ax.legend(fontsize=15)
ax.grid(alpha=0.3)

# overlay binned mean to show the trend clearly
# Bin the VAD wind speed into 1 m/s wide bins (0-20 m/s) and compute the mean |direction diff|
# within each bin, to make the relationship between speed and direction error easier to read
# through the scatter of individual points
ws_bins = np.arange(0, 21, 1)
centers = (ws_bins[:-1] + ws_bins[1:]) / 2  # bin midpoints, used as x-values for the trend line
bin_means = []
for i in range(len(ws_bins) - 1):
    sel = (ws_m >= ws_bins[i]) & (ws_m < ws_bins[i+1])
    # Use NaN for empty bins so the trend line shows a gap rather than a false zero
    bin_means.append(np.mean(np.abs(ddiff_m[sel])) if sel.any() else np.nan)
ax.plot(centers, bin_means, 'o-', color='steelblue', markersize=6,
        linewidth=2, label='binned mean |diff|')
ax.legend(fontsize=15)  # re-call legend so the binned-mean line is included alongside the cutoff line

plt.tight_layout()
plt.show()

# Associated with wsp-vs-diff_plot.png file