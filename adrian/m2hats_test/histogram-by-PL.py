import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# --- Outlier thresholds ---
ws_outlier = 2.0    # m/s threshold: |VAD - HRRR| wind speed difference beyond this is flagged as an outlier
dir_outlier = 30.0  # degree threshold: |VAD - HRRR| wind direction difference beyond this is flagged as an outlier

# --- Accumulators across all files/times/heights ---
# pres_m:  HRRR pressure (hPa) at each matched wind-speed point
# sdiff_m: VAD - HRRR wind speed difference at each matched point
# ddiff_m: VAD - HRRR wind direction difference (only computed when both speeds are above the calm-wind cutoff)
# has_dir: HRRR pressure (hPa) at each matched wind-direction point (parallel to ddiff_m)
pres_m, sdiff_m, ddiff_m, has_dir = [], [], [], []

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
        hrrr_pres = ds.variables['pres'][:]   # HRRR pressure levels corresponding to each height
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

            # --- Wind speed comparison ---
            sd = ws_vad[ti, idx] - hrrr_ws[k]  # VAD - HRRR wind speed difference
            pres_m.append(hrrr_pres[k])
            sdiff_m.append(sd)

            # --- Wind direction comparison ---
            # Only compare directions when both VAD and HRRR wind speeds exceed 2 m/s,
            # since direction is poorly defined / noisy at very low wind speeds
            if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0:
                # Wrap the direction difference into the range [-180, 180) degrees
                dd = ((wd_vad[ti, idx] - hrrr_dir[k] + 180) % 360) - 180
                ddiff_m.append(dd)
                has_dir.append(hrrr_pres[k])

# Convert accumulated lists to numpy arrays for easier filtering/plotting
pres_m = np.array(pres_m)
sdiff_m = np.array(sdiff_m)
ddiff_m = np.array(ddiff_m)
has_dir = np.array(has_dir)

# --- Summary counts ---
print(f"Total speed points: {len(pres_m)}")
print(f"Total direction points: {len(ddiff_m)}")

# Boolean masks flagging which matched points are outliers
ws_out = np.abs(sdiff_m) > ws_outlier
dir_out = np.abs(ddiff_m) > dir_outlier
print(f"Speed outliers (|diff| > {ws_outlier} m/s): {ws_out.sum()}")
print(f"Direction outliers (|diff| > {dir_outlier} deg): {dir_out.sum()}")

# Pressure bins (hPa) used for both histograms, spanning the M2HATS ISS1 site's typical range
bins = np.arange(650, 875, 25)

# --- Plot: 2 side-by-side panels, with outlier counts overlaid on top of all-matched counts ---
plt.rcParams.update({'font.size': 18})  # base font size for everything
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Matched Points and Outliers by Pressure Level (M2HATS)', fontsize=16)

# --- Left panel: wind speed ---
# Gray bars: all matched points per pressure bin
ax1.hist(pres_m, bins=bins, color='lightgray', edgecolor='black', label='All matched')
# Blue bars: subset of matched points flagged as speed outliers, same bins (drawn on top)
ax1.hist(pres_m[ws_out], bins=bins, color='steelblue', edgecolor='black', label=f'Outliers (|diff| > {ws_outlier} m/s)')
ax1.set_xlabel('Pressure (hPa)', fontsize=20)
ax1.set_ylabel('Count', fontsize=20)
ax1.set_title('Wind Speed', fontsize=18)
ax1.tick_params(labelsize=16)
ax1.legend(fontsize=16)
ax1.invert_xaxis()  # higher pressure (lower altitude) on the left, matching typical atmospheric profile convention

# --- Right panel: wind direction ---
# Gray bars: all matched direction points per pressure bin
ax2.hist(has_dir, bins=bins, color='lightgray', edgecolor='black', label='All matched')
# Red bars: subset of matched points flagged as direction outliers, same bins (drawn on top)
ax2.hist(has_dir[dir_out], bins=bins, color='indianred', edgecolor='black', label=f'Outliers (|diff| > {dir_outlier} deg)')
ax2.set_xlabel('Pressure (hPa)', fontsize=20)
ax2.set_ylabel('Count', fontsize=20)
ax2.set_title('Wind Direction', fontsize=18)
ax2.tick_params(labelsize=16)
ax2.legend(fontsize=16)
ax2.invert_xaxis()  # same pressure-axis orientation as the speed panel

plt.tight_layout()
plt.show()