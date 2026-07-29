import netCDF4 as nc
import numpy as np
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Data locations for the M2HATS campaign ---
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

# thresholds to test (None = no SNR filtering)
# Each value is an SNR cutoff (dB) - only VAD points with mean_snr >= threshold will be kept
thresholds = [None, -27.0, -25.0, -23.0, -20.0]

# pre-load all matched points ONCE with their SNR, so we can filter after the fact
# this avoids re-reading every file for every threshold
# all_snr:      VAD mean SNR (dB) at each matched point
# all_sdiff:    VAD - HRRR wind speed difference at each matched point
# all_ddiff_snr: VAD - HRRR wind direction difference at each matched point (computed for all points up front;
#                the speed>2 m/s requirement is applied later via dir_speed_ok rather than during collection)
# all_ws_pair:  (hrrr_ws, vad_ws) tuple at each matched point, used later to build the direction speed-cutoff mask
all_snr, all_sdiff, all_ddiff_snr, all_ws_pair = [], [], [], []

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
    snr_vad = vad.variables['mean_snr'][:]       # mean SNR (dB) profile, used later for threshold filtering
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

        # only require the basic mask here; SNR filtering applied later
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

            # record SNR, speed diff, the wind-speed pair, and dir diff for each match
            all_snr.append(snr_vad[ti, idx])                       # SNR at this matched point, for later threshold filtering
            all_sdiff.append(ws_vad[ti, idx] - hrrr_ws[k])         # VAD - HRRR wind speed difference
            all_ws_pair.append((hrrr_ws[k], ws_vad[ti, idx]))      # raw (HRRR, VAD) speed pair, used to build the >2 m/s direction mask below
            # Wrap the direction difference into the range [-180, 180) degrees to handle the 0/360 boundary correctly
            dd = ((wd_vad[ti, idx] - hrrr_dir[k] + 180) % 360) - 180
            all_ddiff_snr.append(dd)

# Convert accumulated lists to numpy arrays for easier filtering/stats
all_snr = np.array(all_snr)
all_sdiff = np.array(all_sdiff)
all_ddiff_snr = np.array(all_ddiff_snr)
all_ws_pair = np.array(all_ws_pair)  # shape (N, 2): [hrrr_ws, vad_ws]

print(f"Total matched points collected: {len(all_snr)}\n")

# direction comparison requires both speeds > 2 m/s
# (direction is poorly defined / noisy at very low wind speeds, so those points are excluded from direction stats)
dir_speed_ok = (all_ws_pair[:, 0] > 2.0) & (all_ws_pair[:, 1] > 2.0)

# --- evaluate each threshold ---
# Table header: for each SNR cutoff, show point counts and error stats (mean absolute difference, standard deviation)
# for both wind speed and wind direction
print(f"{'SNR cutoff':>12} | {'ws pts':>7} | {'ws mad':>7} | {'ws sd':>7} | {'dir pts':>8} | {'dir mad':>8} | {'dir sd':>7}")
print("-" * 78)
for thresh in thresholds:
    if thresh is None:
        # No SNR filtering - keep every matched point
        keep = np.ones(len(all_snr), dtype=bool)
        label = "none"
    else:
        # Keep only points whose VAD SNR meets or exceeds this threshold
        keep = all_snr >= thresh
        label = f"{thresh:.0f} dB"

    # wind speed stats
    sd = all_sdiff[keep]
    ws_mad = np.mean(np.abs(sd))  # mean absolute difference for wind speed
    ws_sd = np.std(sd)            # standard deviation of the wind speed difference

    # wind direction stats (apply speed cutoff on top of SNR keep)
    dkeep = keep & dir_speed_ok
    dd = all_ddiff_snr[dkeep]
    dir_mad = np.mean(np.abs(dd))  # mean absolute difference for wind direction
    dir_sd = np.std(dd)            # standard deviation of the wind direction difference

    # Print one row of the table for this SNR threshold
    print(f"{label:>12} | {keep.sum():>7} | {ws_mad:>7.2f} | {ws_sd:>7.2f} | "
          f"{dkeep.sum():>8} | {dir_mad:>8.2f} | {dir_sd:>7.2f}")