import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# --- Accumulators across all files/times/heights ---
# hrrr_ws_m / vad_ws_m: matched HRRR / VAD wind speed values (one entry per matched point)
# hrrr_dir_m / vad_dir_m: matched HRRR / VAD wind direction values (only kept when both speeds are above the calm-wind cutoff)
hrrr_ws_m, vad_ws_m, hrrr_dir_m, vad_dir_m = [], [], [], []

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
  height = vad.variables['height'][:]            # AGL
  base_t = int(vad.variables['base_time'][:])  # reference epoch time for this file
  time_vad = vad.variables['time'][:]          # offsets (seconds) from base_time for each time step
  vad.close()

  # Mask fill values (-9999.0) so they don't get treated as real data
  ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
  wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)

  # Convert VAD time offsets into absolute epoch seconds for matching against HRRR times
  vad_epoch = base_t + time_vad
  
  # HRRR pressure levels per hour
  # Loop over each HRRR profile file for this day (typically multiple forecast/analysis times)
  for f in h_files:
    ds = nc.Dataset(f)
    hrrr_ws = ds.variables['wspd'][:]     # HRRR wind speed profile
    hrrr_dir = ds.variables['wdir'][:]    # HRRR wind direction profile
    hrrr_agl = ds.variables['height'][:]  # HRRR heights, meters above ground level
    et = int(ds.variables['time'][0])     # epoch time of this HRRR profile
    ds.close()
    
    # (left commented out: an older approach that derived speed/direction/AGL height from
    # raw u/v wind components and geopotential z, before the file provided wspd/wdir/height directly)
#    hrrr_ws = np.sqrt(u**2 + v**2)
#    hrrr_dir = np.degrees(np.arctan2(-u, -v)) % 360
#    hrrr_agl = z / 9.80665 - surf_alt
    
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

      # --- Wind speed: store the matched HRRR/VAD pair ---
      hrrr_ws_m.append(hrrr_ws[k]); vad_ws_m.append(ws_vad[ti, idx])

      # --- Wind direction: only store the matched pair when both speeds exceed 2 m/s,
      # since direction is poorly defined / noisy at very low wind speeds ---
      if hrrr_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0:
        hrrr_dir_m.append(hrrr_dir[k]); vad_dir_m.append(wd_vad[ti, idx])
      
# Convert accumulated lists to numpy arrays for easier math/plotting
hrrr_ws_m = np.array(hrrr_ws_m); vad_ws_m = np.array(vad_ws_m)
hrrr_dir_m = np.array(hrrr_dir_m); vad_dir_m = np.array(vad_dir_m)

# --- Summary counts ---
print(f"Matched points: {len(hrrr_ws_m)}")
print(f"Matched points: {len(hrrr_dir_m)}")

# Sanity check: bail out early if nothing matched at all (likely a height/time alignment bug)
if len(hrrr_ws_m) == 0:
  raise SystemExit("Still zero matches - check surf_alt and hrrr_agl values.")
  
# --- Wind speed error stats ---
sdiff = vad_ws_m - hrrr_ws_m           # VAD - HRRR wind speed difference, one value per matched point
s_mad, s_sd = np.mean(np.abs(sdiff)), np.std(sdiff)   # mean absolute difference and standard deviation of the difference
s_fit = np.polyfit(hrrr_ws_m, vad_ws_m, 1)             # linear fit of VAD vs HRRR speed (slope, intercept) - not plotted here but available for diagnostics

# --- Wind direction error stats ---
# Drop any NaNs before computing direction differences
dmask = ~np.isnan(vad_dir_m) & ~np.isnan(hrrr_dir_m)
# Wrap the direction difference into the range [-180, 180) degrees to handle the 0/360 boundary correctly
cdiff = ((vad_dir_m[dmask] - hrrr_dir_m[dmask] + 180) % 360) - 180
d_mad, d_sd = np.mean(np.abs(cdiff)), np.std(cdiff)   # mean absolute difference and standard deviation of the wrapped difference
d_fit = np.polyfit(hrrr_dir_m[dmask], vad_dir_m[dmask], 1)  # linear fit of VAD vs HRRR direction (slope, intercept) - not plotted here but available for diagnostics

# --- Plot: 2 side-by-side panels, histogram of VAD-HRRR differences (i.e. the error binned into counts) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('M2HATS - Difference Histogram')

# --- Left panel: wind speed difference histogram ---
# Bins the speed error (VAD - HRRR, in m/s) into 50 bins and counts how many matched points fall in each
ax1.hist(sdiff, bins=50, color='black', edgecolor='none')
ax1.axvline(0, color='red', linewidth=1)  # reference line at zero error
ax1.set_xlabel('VAD - HRRR Wind Speed (m/s)')
ax1.set_ylabel('Count')
ax1.set_title('Wind Speed Difference')
# Annotate the panel with the point count and summary error stats (mean absolute difference, standard deviation)
ax1.text(0.05, 0.95, f"{len(hrrr_ws_m)} pts\nmad: {s_mad:.1f}, sd: {s_sd:.1f}",
         transform=ax1.transAxes, va='top')

# --- Right panel: wind direction difference histogram ---
# Bins the (wrapped) direction error (VAD - HRRR, in degrees) into 50 bins and counts how many matched points fall in each
ax2.hist(cdiff, bins=50, color='black', edgecolor='none')
ax2.axvline(0, color='red', linewidth=1)  # reference line at zero error
ax2.set_xlabel('VAD - HRRR Wind Direction (deg)')
ax2.set_ylabel('Count')
ax2.set_title('Wind Direction Difference')
# Annotate the panel with the point count and summary error stats (mean absolute difference, standard deviation)
ax2.text(0.05, 0.95, f"{int(dmask.sum())} pts\nmad: {d_mad:.1f}, sd: {d_sd:.1f}",
         transform=ax2.transAxes, va='top')

plt.tight_layout()
plt.show()
# Associated with histogram.png file