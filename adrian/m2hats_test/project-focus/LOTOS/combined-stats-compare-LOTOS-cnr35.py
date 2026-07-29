import netCDF4 as nc
import numpy as np
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

# --- Base directories for VAD lidar data and the two model datasets ---
vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'
hrrr_base = '/scr/isf_apg/models/lotos2025/hrrr/'
era5_base = '/scr/isf_apg/models/lotos2025/era5/'

# Read site elevation once from the first VAD file, since it's needed later
# to convert ERA5 geopotential height (MSL) into AGL height for matching.
_first_vad = sorted(glob.glob(vad_base + 'VAD_*.nc'))[0]
_v = nc.Dataset(_first_vad)
site_alt = float(_v.variables['alt'][:])  # site elevation in meters MSL
_v.close()
print(f"Site elevation read from VAD: {site_alt:.1f} m MSL\n")


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


def match_one(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
              hours_s, sdiff, hours_d, ddiff):
    """Match a single model time slice (one HRRR profile file, or one ERA5
    hourly step) against the closest-in-time VAD profile, then pair up
    height levels between model and VAD and append the speed/direction
    differences (VAD minus model) to the running lists.

    Parameters
    ----------
    m_ws, m_dir, m_agl : model wind speed, wind direction, and AGL heights
        for this single time slice.
    et : epoch time (UTC seconds) of this model time slice.
    ws_vad, wd_vad, height, vad_epoch : full VAD arrays for the day, as
        returned by read_vad().
    hours_s, sdiff, hours_d, ddiff : output lists (mutated in place) that
        accumulate hour-of-day and difference values for speed and
        direction respectively.
    """
    # Hour-of-day bucket (UTC) used later for diurnal-cycle statistics
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

        # Skip if the matched VAD wind speed value is masked
        if np.ma.is_masked(ws_vad[ti, idx]):
            continue

        # --- Wind speed difference (VAD - model) ---
        hours_s.append(hour)
        sdiff.append(float(ws_vad[ti, idx] - m_ws[k]))

        # --- Wind direction difference (VAD - model), only computed when
        # both speeds exceed 2 m/s (direction is unreliable at low speeds)
        # and the VAD direction value isn't masked ---
        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            # Wrap direction difference into [-180, 180] degrees
            dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180
            hours_d.append(hour)
            ddiff.append(dd)


def run_hrrr():
    """Loop over all VAD days, find the matching HRRR profile file(s) for
    each day, and accumulate VAD-vs-HRRR differences across the whole
    campaign using match_one()."""
    hours_s, sdiff, hours_d, ddiff = [], [], [], []
    for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
        # Extract the date string from the VAD filename (e.g. VAD_20250715.nc -> 20250715)
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        # HRRR filenames mix ISS1/ISS2 site suffixes across this project,
        # so wildcard the suffix to catch whichever site tag was used
        h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS*.nc'))
        if not h_files:
            continue  # no HRRR data for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        # A given day can have multiple HRRR profile files (e.g. multiple
        # forecast/init times), so loop over all of them
        for f in h_files:
            ds = nc.Dataset(f)
            m_ws  = ds.variables['wspd'][:]
            m_dir = ds.variables['wdir'][:]
            m_agl = ds.variables['height'][:]
            et = int(ds.variables['time'][0])  # single valid time for this HRRR profile file
            ds.close()
            match_one(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                      hours_s, sdiff, hours_d, ddiff)
    return np.array(hours_s), np.array(sdiff), np.array(hours_d), np.array(ddiff)


def run_era5():
    """Loop over all VAD days, load the corresponding ERA5 pressure-level
    file for each day, derive wind speed/direction/AGL height from u/v/z,
    and accumulate VAD-vs-ERA5 differences across the whole campaign using
    match_one()."""
    hours_s, sdiff, hours_d, ddiff = [], [], [], []
    for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        e_file = era5_base + 'era5_pressure_' + date + '_lotos2025.nc'
        if not os.path.exists(e_file):
            continue  # no ERA5 file for this day, skip it
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)

        ds = nc.Dataset(e_file)
        # Index [:, :, 0, 0] selects the single (lat, lon) grid point nearest
        # the site, keeping (time, pressure_level) dimensions -> shape (24, 37)
        u_all = ds.variables['u'][:, :, 0, 0]   # (24, 37) zonal wind component
        v_all = ds.variables['v'][:, :, 0, 0]   # (24, 37) meridional wind component
        z_all = ds.variables['z'][:, :, 0, 0]   # (24, 37) geopotential
        vt_all = ds.variables['valid_time'][:]  # epoch time for each of the 24 hourly steps
        ds.close()

        # ERA5 provides hourly time steps with multiple pressure levels each;
        # process one hourly time step at a time
        for t in range(len(vt_all)):
            u, v, z = u_all[t], v_all[t], z_all[t]
            m_ws  = np.sqrt(u**2 + v**2)                     # wind speed from components
            m_dir = np.degrees(np.arctan2(-u, -v)) % 360     # meteorological wind direction (from-direction, 0-360 deg)
            m_agl = z / 9.80665 - site_alt                   # geopotential -> geopotential height -> AGL (subtract site elevation)
            match_one(m_ws, m_dir, m_agl, int(vt_all[t]), ws_vad, wd_vad, height, vad_epoch,
                      hours_s, sdiff, hours_d, ddiff)
    return np.array(hours_s), np.array(sdiff), np.array(hours_d), np.array(ddiff)


# --- run both model comparisons across the full campaign ---
print("Running HRRR comparison...")
h_hs, h_sd, h_hd, h_dd = run_hrrr()
print("Running ERA5 comparison...")
e_hs, e_sd, e_hd, e_dd = run_era5()


# --- output helpers ---

def overall(diff):
    """Compute summary statistics (mean, std, mean absolute difference, min,
    max, range, and sample count) for an array of differences. Returns None
    if there are no points to summarize."""
    if len(diff) == 0:
        return None
    return dict(mean=np.mean(diff), std=np.std(diff), mad=np.mean(np.abs(diff)),
                mn=diff.min(), mx=diff.max(), rng=diff.max() - diff.min(), n=len(diff))


def hour_means(hour_arr, diff_arr):
    """Bin the difference values by hour-of-day (0-23) and compute the mean,
    range, and count of points in each hour bin. Hours with no data get NaN
    for mean/range and a count of 0."""
    means, rngs, counts = [], [], []
    for h in range(24):
        sel = hour_arr == h
        if sel.any():
            d = diff_arr[sel]
            means.append(np.mean(d)); rngs.append(d.max() - d.min()); counts.append(sel.sum())
        else:
            means.append(np.nan); rngs.append(np.nan); counts.append(0)
    return np.array(means), np.array(rngs), np.array(counts)


def print_overall_side_by_side(title, hrrr_diff, era5_diff):
    """Print a small side-by-side table comparing overall HRRR vs ERA5
    summary statistics for a given variable (e.g. wind speed or direction)."""
    h, e = overall(hrrr_diff), overall(era5_diff)
    print(f"\n{title}")
    print(f"{'metric':>10} | {'HRRR':>10} | {'ERA5':>10}")
    print("-" * 38)
    for key, lbl in [('mean','mean'),('std','std'),('mad','mad'),('mn','min'),('mx','max'),('rng','range')]:
        hv = f"{h[key]:.2f}" if h else "--"
        ev = f"{e[key]:.2f}" if e else "--"
        print(f"{lbl:>10} | {hv:>10} | {ev:>10}")
    print(f"{'n points':>10} | {(h['n'] if h else 0):>10} | {(e['n'] if e else 0):>10}")


def print_hourly_side_by_side(title, h_hour, h_diff, e_hour, e_diff):
    """Print an hour-by-hour (0-23, UTC) side-by-side table comparing HRRR
    vs ERA5 mean difference, range, and sample count for a given variable."""
    hm, hr, hc = hour_means(h_hour, h_diff)
    em, er, ec = hour_means(e_hour, e_diff)
    print(f"\n{title}")
    print(f"{'':>4} | {'--- HRRR ---':^26} | {'--- ERA5 ---':^26}")
    print(f"{'hr':>4} | {'mean':>8} {'range':>8} {'n':>6} | {'mean':>8} {'range':>8} {'n':>6}")
    print("-" * 64)
    for h in range(24):
        def fmt(m, r, c):
            # Show placeholders instead of NaN when there's no data for this hour
            return f"{'--':>8} {'--':>8} {c:>6}" if c == 0 else f"{m:>8.2f} {r:>8.2f} {c:>6}"
        print(f"{h:>4} | {fmt(hm[h], hr[h], hc[h])} | {fmt(em[h], er[h], ec[h])}")


# --- report results ---
print(f"\n{'='*64}")
print("VAD vs HRRR and ERA5 - LOTOS2025  (vad_cnr40_error dataset)")
print(f"{'='*64}")

# Overall (campaign-wide) summary stats for speed and direction
print_overall_side_by_side("WIND SPEED overall [m/s]", h_sd, e_sd)
print_overall_side_by_side("WIND DIRECTION overall [deg]", h_dd, e_dd)
# Diurnal-cycle breakdown (by UTC hour) for speed and direction
print_hourly_side_by_side("WIND SPEED difference by hour [m/s]", h_hs, h_sd, e_hs, e_sd)
print_hourly_side_by_side("WIND DIRECTION difference by hour [deg]", h_hd, h_dd, e_hd, e_dd)