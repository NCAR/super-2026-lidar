import netCDF4 as nc
import numpy as np
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- dataset configs: one VAD/HRRR/ERA5 path set per campaign ---
lotos_vad  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'
lotos_hrrr = '/scr/isf_apg/models/lotos2025/hrrr/'
lotos_era5 = '/scr/isf_apg/models/lotos2025/era5/'

m2hats_vad  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
m2hats_hrrr = '/scr/isf_apg/models/m2hats/hrrr/'
m2hats_era5 = '/scr/isf_apg/models/m2hats/era5/'

# Site elevations, needed to convert ERA5 geopotential height (MSL) into AGL
# height for matching. LOTOS reads it from the VAD file; M2HATS's VAD 'alt'
# is masked/unusable so it's hardcoded from ERA5 surface geopotential instead.
_fv = sorted(glob.glob(lotos_vad + 'VAD_*.nc'))[0]
_v = nc.Dataset(_fv); lotos_alt = float(_v.variables['alt'][:]); _v.close()
m2hats_alt = 1739.0


def read_vad(vad_file):
    """Open a single VAD NetCDF file and return wind speed/direction (masked
    where flagged as -9999.0), height levels, and absolute epoch time for
    each profile (base_time + per-record time offset)."""
    vad = nc.Dataset(vad_file)
    ws = np.ma.masked_where(vad.variables['wind_speed'][:] == -9999.0, vad.variables['wind_speed'][:])
    wd = np.ma.masked_where(vad.variables['wind_direction'][:] == -9999.0, vad.variables['wind_direction'][:])
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()
    return ws, wd, height, base_t + time_vad  # last item = absolute epoch times


def match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                  hours_s, hours_d):
    """Same matching logic as the plot script, but only records hours.

    This is a data-availability check, not a comparison: it doesn't compute
    or store any speed/direction differences, it just appends the hour-of-day
    of each successfully matched point to `hours_s` (and, when the direction
    is also usable, to `hours_d` too). Used to tally how many comparison
    points fall in each UTC hour, per campaign/model combination, to help
    decide on an appropriate CNR/QC mask."""
    hour = datetime.fromtimestamp(et, tz=timezone.utc).hour

    # Find the VAD profile closest in time to this model time slice
    ti = np.argmin(np.abs(vad_epoch - et))
    # Reject the match if the nearest VAD profile is more than 15 min (900 s) away
    if abs(vad_epoch[ti] - et) > 900:
        return
    # Only consider VAD height levels that aren't masked/missing at this time
    valid = ~np.ma.getmaskarray(ws_vad[ti])
    if not valid.any():
        return
    h_valid = height[valid]          # VAD heights with valid data at time ti
    idx_valid = np.where(valid)[0]   # original array indices of those valid heights

    # Loop over model height levels and try to pair each with the nearest
    # valid VAD height level
    for k in range(len(m_agl)):
        # Restrict comparison to the 100-2000 m AGL range
        if not (100 <= m_agl[k] <= 2000):
            continue
        # Nearest valid VAD height level to this model level
        j = np.argmin(np.abs(h_valid - m_agl[k]))
        # Reject the pairing if the height difference exceeds 25 m tolerance
        if np.abs(h_valid[j] - m_agl[k]) > 25:
            continue
        idx = idx_valid[j]  # index back into the full VAD height/array space
        if np.ma.is_masked(ws_vad[ti, idx]):
            continue

        # Record this as a usable wind-speed comparison point for this hour
        hours_s.append(hour)

        # Direction is only usable when both speeds exceed 2 m/s (direction
        # is unreliable at low speeds) and the VAD direction isn't masked
        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            hours_d.append(hour)


def run_lotos_hrrr():
    """Tally hourly speed/direction point counts for LOTOS2025 VAD vs HRRR."""
    hs, hd = [], []
    for vad_file in sorted(glob.glob(lotos_vad + 'VAD_*.nc')):
        # Extract the date string from the VAD filename (e.g. VAD_20250715.nc -> 20250715)
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        # HRRR filenames mix ISS1/ISS2 site suffixes across this project, so wildcard the suffix
        h_files = sorted(glob.glob(lotos_hrrr + date + '/hrrr_profile_' + date + '_*_ISS*.nc'))
        if not h_files:
            continue  # no HRRR data for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        # A given day can have multiple HRRR profile files, so loop over all of them
        for f in h_files:
            ds = nc.Dataset(f)
            m_ws  = ds.variables['wspd'][:]
            m_dir = ds.variables['wdir'][:]
            m_agl = ds.variables['height'][:]
            et = int(ds.variables['time'][0])  # single valid time for this HRRR profile file
            ds.close()
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch, hs, hd)
    return np.array(hs), np.array(hd)


def run_lotos_era5():
    """Tally hourly speed/direction point counts for LOTOS2025 VAD vs ERA5
    (LOTOS: one ERA5 file per day, 24 hourly time slices inside)."""
    hs, hd = [], []
    for vad_file in sorted(glob.glob(lotos_vad + 'VAD_*.nc')):
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        e_file = lotos_era5 + 'era5_pressure_' + date + '_lotos2025.nc'
        if not os.path.exists(e_file):
            continue  # no ERA5 file for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        ds = nc.Dataset(e_file)
        # Index [:, :, 0, 0] selects the single (lat, lon) grid point nearest
        # the site, keeping (time, pressure_level) dimensions
        u_all = ds.variables['u'][:, :, 0, 0]   # zonal wind component
        v_all = ds.variables['v'][:, :, 0, 0]   # meridional wind component
        z_all = ds.variables['z'][:, :, 0, 0]   # geopotential
        vt_all = ds.variables['valid_time'][:]  # epoch time for each hourly step
        ds.close()
        # ERA5 provides hourly time steps with multiple pressure levels each;
        # process one hourly time step at a time
        for t in range(len(vt_all)):
            u, v, z = u_all[t], v_all[t], z_all[t]
            m_ws  = np.sqrt(u**2 + v**2)                   # wind speed from components
            m_dir = np.degrees(np.arctan2(-u, -v)) % 360   # meteorological wind direction (from-direction, 0-360 deg)
            m_agl = z / 9.80665 - lotos_alt                # geopotential -> geopotential height -> AGL
            match_profile(m_ws, m_dir, m_agl, int(vt_all[t]), ws_vad, wd_vad, height, vad_epoch, hs, hd)
    return np.array(hs), np.array(hd)


def run_m2hats_hrrr():
    """Tally hourly speed/direction point counts for M2HATS VAD vs HRRR
    (M2HATS uses ISS1, per-hour profile files)."""
    hs, hd = [], []
    for vad_file in sorted(glob.glob(m2hats_vad + '30min_winds_*.nc')):
        # Extract the date string from the VAD filename (e.g. 30min_winds_20230715.nc -> 20230715)
        date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
        h_files = sorted(glob.glob(m2hats_hrrr + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))
        if not h_files:
            continue  # no HRRR data for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        for f in h_files:
            ds = nc.Dataset(f)
            m_ws  = ds.variables['wspd'][:]
            m_dir = ds.variables['wdir'][:]
            m_agl = ds.variables['height'][:]
            et = int(ds.variables['time'][0])  # single valid time for this HRRR profile file
            ds.close()
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch, hs, hd)
    return np.array(hs), np.array(hd)


def run_m2hats_era5():
    """Tally hourly speed/direction point counts for M2HATS VAD vs ERA5
    (M2HATS: per-hour ERA5 files in date subdirectories, unlike LOTOS's
    one-file-per-day layout)."""
    hs, hd = [], []
    for vad_file in sorted(glob.glob(m2hats_vad + '30min_winds_*.nc')):
        date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
        e_files = sorted(glob.glob(m2hats_era5 + date + '/era5_pressure_' + date + '_*_ISS1.nc'))
        if not e_files:
            continue  # no ERA5 data for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        # Loop over each hourly ERA5 profile file for this day
        for f in e_files:
            ds = nc.Dataset(f)
            # Index [0, :, 0, 0] selects the single time step and single (lat, lon)
            # grid point nearest the site, keeping only the pressure-level dimension
            u = ds.variables['u'][0, :, 0, 0]   # zonal wind component
            v = ds.variables['v'][0, :, 0, 0]   # meridional wind component
            z = ds.variables['z'][0, :, 0, 0]   # geopotential
            et = int(ds.variables['valid_time'][0])  # single valid time for this ERA5 file
            ds.close()
            m_ws  = np.sqrt(u**2 + v**2)                   # wind speed from components
            m_dir = np.degrees(np.arctan2(-u, -v)) % 360   # meteorological wind direction (from-direction, 0-360 deg)
            m_agl = z / 9.80665 - m2hats_alt               # geopotential -> geopotential height -> AGL
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch, hs, hd)
    return np.array(hs), np.array(hd)


def hour_counts(hour_arr):
    """Count how many entries in hour_arr fall in each UTC hour (0-23)."""
    return np.array([(hour_arr == h).sum() for h in range(24)])


# --- run all four campaign/model combinations and tabulate ---
print("Running LOTOS x HRRR..."); lh_s, lh_d = run_lotos_hrrr()
print("Running LOTOS x ERA5..."); le_s, le_d = run_lotos_era5()
print("Running M2HATS x HRRR..."); mh_s, mh_d = run_m2hats_hrrr()
print("Running M2HATS x ERA5..."); me_s, me_d = run_m2hats_era5()

# Per combination: (hourly speed-point counts, hourly direction-point counts)
counts = {
    'LOTOS-HRRR':  (hour_counts(lh_s), hour_counts(lh_d)),
    'LOTOS-ERA5':  (hour_counts(le_s), hour_counts(le_d)),
    'M2HATS-HRRR': (hour_counts(mh_s), hour_counts(mh_d)),
    'M2HATS-ERA5': (hour_counts(me_s), hour_counts(me_d)),
}

labels = list(counts.keys())

# Print one table per variable (speed, direction), with a column per
# campaign/model combination and a row per UTC hour, plus summary stats
# (min/median/total across hours) at the bottom to help judge data coverage
# and guide the choice of CNR/QC mask.
for metric, mi in [('WIND SPEED', 0), ('WIND DIRECTION', 1)]:
    print(f"\n{'='*70}")
    print(f"POINT COUNTS PER HOUR - {metric}")
    print(f"{'='*70}")
    header = f"{'hr':>4} |" + "".join(f" {lab:>12} |" for lab in labels)
    print(header)
    print("-" * len(header))
    for h in range(24):
        row = f"{h:>4} |"
        for lab in labels:
            row += f" {counts[lab][mi][h]:>12} |"
        print(row)
    print("-" * len(header))
    # summary stats to guide the mask choice
    row_min = f"{'min':>4} |"
    row_med = f"{'med':>4} |"
    row_tot = f"{'tot':>4} |"
    for lab in labels:
        c = counts[lab][mi]
        row_min += f" {c.min():>12} |"
        row_med += f" {int(np.median(c)):>12} |"
        row_tot += f" {c.sum():>12} |"
    print(row_min)
    print(row_med)
    print(row_tot)