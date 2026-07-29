import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
ws_outlier = 2.0    # m/s threshold for speed outliers
dir_outlier = 30.0  # degree threshold for direction outliers

# collect per matched point
# ws_snr / ws_hrrr / ws_vad_v: VAD SNR, HRRR wind speed, and VAD wind speed at each matched point
# dir_snr / dir_hrrr / dir_vad_v: VAD SNR, HRRR direction, and VAD direction at each matched point
#                                 (only kept when both speeds are above the calm-wind cutoff)
ws_snr, ws_hrrr, ws_vad_v = [], [], []          # speed: snr, hrrr ws, vad ws
dir_snr, dir_hrrr, dir_vad_v = [], [], []        # direction: snr, hrrr dir, vad dir

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
    snr_vad = vad.variables['mean_snr'][:]       # mean SNR (dB) profile, used to group outliers by signal quality
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

            # skip masked elements defensively
            # (extra safety check beyond the `valid` mask, in case of partial masking at this index)
            if np.ma.is_masked(ws_vad[ti, idx]):
                continue

            # --- Wind speed: store SNR, HRRR speed, and VAD speed for this matched point ---
            ws_snr.append(float(snr_vad[ti, idx]))
            ws_hrrr.append(float(hrrr_ws[k]))
            ws_vad_v.append(float(ws_vad[ti, idx]))

            # --- Wind direction: only store when both speeds exceed 2 m/s and the VAD direction
            # value is valid, since direction is poorly defined / noisy at very low wind speeds ---
            if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
                dir_snr.append(float(snr_vad[ti, idx]))
                dir_hrrr.append(float(hrrr_dir[k]))
                dir_vad_v.append(float(wd_vad[ti, idx]))

# Convert accumulated lists to numpy arrays for easier math/filtering/plotting
ws_snr = np.array(ws_snr); ws_hrrr = np.array(ws_hrrr); ws_vad_v = np.array(ws_vad_v)
dir_snr = np.array(dir_snr); dir_hrrr = np.array(dir_hrrr); dir_vad_v = np.array(dir_vad_v)

# --- Summary counts ---
print(f"Speed points: {len(ws_snr)}, Direction points: {len(dir_snr)}")

# --- linear fits and residuals ---
# speed: fit vad vs hrrr, residual = vad - fit(hrrr)
# The residual captures how far each point falls from the overall VAD-vs-HRRR trend line,
# as opposed to the raw diff below which just measures VAD - HRRR directly
s_fit = np.polyfit(ws_hrrr, ws_vad_v, 1)   # slope, intercept of the best-fit line: vad_speed ~ hrrr_speed
s_pred = np.polyval(s_fit, ws_hrrr)        # predicted VAD speed for each point, from the fit
s_resid = ws_vad_v - s_pred                # residual: how far the actual VAD speed is from that prediction

# direction: fit vad vs hrrr, residual = vad - fit(hrrr)
d_fit = np.polyfit(dir_hrrr, dir_vad_v, 1)  # slope, intercept of the best-fit line: vad_dir ~ hrrr_dir
d_pred = np.polyval(d_fit, dir_hrrr)        # predicted VAD direction for each point, from the fit
d_resid = dir_vad_v - d_pred                # residual: how far the actual VAD direction is from that prediction

print(f"Speed fit: {s_fit[0]:.2f}x + {s_fit[1]:.2f}")
print(f"Direction fit: {d_fit[0]:.2f}x + {d_fit[1]:.2f}")

# --- outlier definitions ---
# speed outlier: |raw diff| > threshold
# (this uses the simple VAD - HRRR difference, not the fit residual, to flag outliers)
s_rawdiff = ws_vad_v - ws_hrrr
ws_out = np.abs(s_rawdiff) > ws_outlier

# direction outlier: circular diff > threshold
# Wrap the direction difference into the range [-180, 180) degrees to handle the 0/360 boundary correctly
d_rawdiff = ((dir_vad_v - dir_hrrr + 180) % 360) - 180
dir_out = np.abs(d_rawdiff) > dir_outlier

print(f"Speed outliers: {ws_out.sum()}, Direction outliers: {dir_out.sum()}")

# --- plot: 2x2 grid ---
# This figure looks at the same outliers (defined above by raw diff, not by fit residual) from
# two different angles: how they're distributed across SNR values (top row), and how they're
# distributed across fit-residual values (bottom row) - i.e. whether outliers cluster at low SNR,
# and whether they also stand out as large residuals relative to the overall trend line.
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Outliers grouped by SNR and by Fit Residual (M2HATS)', fontsize=17)

# Shared SNR bins across both speed and direction panels, spanning the full range seen in either dataset
snr_bins = np.linspace(min(ws_snr.min(), dir_snr.min()), max(ws_snr.max(), dir_snr.max()), 30)

# --- top-left: speed outliers by SNR ---
# Gray bars: all matched speed points binned by their SNR value
axes[0,0].hist(ws_snr, bins=snr_bins, color='lightgray', edgecolor='black', label='All matched')
# Blue bars: subset of those points that are speed outliers, same SNR bins (drawn on top)
axes[0,0].hist(ws_snr[ws_out], bins=snr_bins, color='steelblue', edgecolor='black',
               label=f'Outliers (|diff|>{ws_outlier} m/s)')
axes[0,0].set_xlabel('mean_snr (dB)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Wind Speed Outliers by SNR')
axes[0,0].legend(fontsize=12)

# --- top-right: direction outliers by SNR ---
# Gray bars: all matched direction points binned by their SNR value
axes[0,1].hist(dir_snr, bins=snr_bins, color='lightgray', edgecolor='black', label='All matched')
# Red bars: subset of those points that are direction outliers, same SNR bins (drawn on top)
axes[0,1].hist(dir_snr[dir_out], bins=snr_bins, color='indianred', edgecolor='black',
               label=f'Outliers (|diff|>{dir_outlier} deg)')
axes[0,1].set_xlabel('mean_snr (dB)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Wind Direction Outliers by SNR')
axes[0,1].legend(fontsize=12)

# --- bottom-left: speed outliers by fit residual ---
# Bins sized to the actual range of speed residuals seen
s_res_bins = np.linspace(s_resid.min(), s_resid.max(), 30)
# Gray bars: all matched speed points binned by their fit residual
axes[1,0].hist(s_resid, bins=s_res_bins, color='lightgray', edgecolor='black', label='All matched')
# Blue bars: subset of those points that are (raw-diff) speed outliers, same residual bins (drawn on top)
axes[1,0].hist(s_resid[ws_out], bins=s_res_bins, color='steelblue', edgecolor='black',
               label=f'Outliers (|diff|>{ws_outlier} m/s)')
axes[1,0].axvline(0, color='red', linewidth=1)  # reference line at zero residual (right on the fit line)
axes[1,0].set_xlabel('Speed fit residual (m/s)')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Wind Speed Outliers by Fit Residual')
axes[1,0].legend(fontsize=12)

# --- bottom-right: direction outliers by fit residual ---
# Bins sized to the actual range of direction residuals seen
d_res_bins = np.linspace(d_resid.min(), d_resid.max(), 30)
# Gray bars: all matched direction points binned by their fit residual
axes[1,1].hist(d_resid, bins=d_res_bins, color='lightgray', edgecolor='black', label='All matched')
# Red bars: subset of those points that are (raw-diff) direction outliers, same residual bins (drawn on top)
axes[1,1].hist(d_resid[dir_out], bins=d_res_bins, color='indianred', edgecolor='black',
               label=f'Outliers (|diff|>{dir_outlier} deg)')
axes[1,1].axvline(0, color='red', linewidth=1)  # reference line at zero residual (right on the fit line)
axes[1,1].set_xlabel('Direction fit residual (deg)')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Wind Direction Outliers by Fit Residual')
axes[1,1].legend(fontsize=12)

plt.tight_layout()
plt.show()

# Associated with snr-resid-histogram.png file