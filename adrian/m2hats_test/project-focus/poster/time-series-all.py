import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
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

# site elevations (only used for ERA5 height conversion)
_fv = sorted(glob.glob(lotos_vad + 'VAD_*.nc'))[0]
_v = nc.Dataset(_fv); lotos_alt = float(_v.variables['alt'][:]); _v.close()
m2hats_alt = 1739.0  # M2HATS VAD 'alt' is masked; ERA5 surface geopotential value
print(f"LOTOS site elevation: {lotos_alt:.1f} m | M2HATS site elevation: {m2hats_alt:.1f} m\n")


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
                  hours_s, sdiff, hours_d, ddiff):
    """Match one model time slice to the VAD; append signed hourly differences.

    Unlike the point-count-only version of this script, this keeps the
    actual signed difference (VAD - model) for speed and direction, so the
    caller can later average them by hour to build the diurnal-cycle
    comparison plot."""
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

        # --- Wind speed difference (VAD - model) ---
        hours_s.append(hour)
        sdiff.append(float(ws_vad[ti, idx] - m_ws[k]))

        # --- Wind direction difference (VAD - model), only computed when
        # both speeds exceed 2 m/s (direction is unreliable at low speeds)
        # and the VAD direction value isn't masked ---
        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180  # wrap into [-180, 180]
            hours_d.append(hour)
            ddiff.append(dd)


def run_lotos_hrrr():
    """Collect hourly speed/direction differences for LOTOS2025 VAD vs HRRR."""
    hs, sd, hd, dd = [], [], [], []
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
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                          hs, sd, hd, dd)
    return map(np.array, (hs, sd, hd, dd))


def run_lotos_era5():
    """Collect hourly speed/direction differences for LOTOS2025 VAD vs ERA5
    (LOTOS: one ERA5 file per day, 24 hourly time slices inside)."""
    hs, sd, hd, dd = [], [], [], []
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
            match_profile(m_ws, m_dir, m_agl, int(vt_all[t]), ws_vad, wd_vad, height, vad_epoch,
                          hs, sd, hd, dd)
    return map(np.array, (hs, sd, hd, dd))


def run_m2hats_hrrr():
    """Collect hourly speed/direction differences for M2HATS VAD vs HRRR
    (M2HATS uses ISS1, per-hour profile files)."""
    hs, sd, hd, dd = [], [], [], []
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
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                          hs, sd, hd, dd)
    return map(np.array, (hs, sd, hd, dd))


def run_m2hats_era5():
    """Collect hourly speed/direction differences for M2HATS VAD vs ERA5
    (M2HATS: per-hour ERA5 files in date subdirectories, unlike LOTOS's
    one-file-per-day layout)."""
    hs, sd, hd, dd = [], [], [], []
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
            match_profile(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                          hs, sd, hd, dd)
    return map(np.array, (hs, sd, hd, dd))


def hourly_means(hour_arr, diff_arr):
    """Compute the mean difference for each UTC hour (0-23). Hours with no
    matched points get NaN, so the plotted line simply has a gap there."""
    means = np.full(24, np.nan)
    for h in range(24):
        sel = hour_arr == h
        if sel.any():
            means[h] = np.mean(diff_arr[sel])
    return means


# --- run all four campaign/model comparisons ---
# Each dataset tuple carries: (legend label, line color, line style,
# hours_speed, speed_diffs, hours_dir, dir_diffs). Colors are shared per
# campaign (LOTOS=blue, M2HATS=orange, both Okabe-Ito colorblind-safe) and
# line style distinguishes the model within a campaign (solid=HRRR, dashed=ERA5).
datasets = []
print("Running LOTOS x HRRR...")
hs, sd, hd, dd = run_lotos_hrrr()
datasets.append(('LOTOS - HRRR', '#0072B2', '-', hs, sd, hd, dd))
print("Running LOTOS x ERA5...")
hs, sd, hd, dd = run_lotos_era5()
datasets.append(('LOTOS - ERA5', '#0072B2', '--', hs, sd, hd, dd))
print("Running M2HATS x HRRR...")
hs, sd, hd, dd = run_m2hats_hrrr()
datasets.append(('M2HATS - HRRR', '#D55E00', '-', hs, sd, hd, dd))
print("Running M2HATS x ERA5...")
hs, sd, hd, dd = run_m2hats_era5()
datasets.append(('M2HATS - ERA5', '#D55E00', '--', hs, sd, hd, dd))

# Quick sanity-check print of point counts for each of the four comparisons
for label, _, _, hs, sd, hd, dd in datasets:
    print(f"{label}: {len(sd)} speed pts, {len(dd)} direction pts")

# --- plot: two stacked panels (speed on top, direction below), one line per
# campaign/model comparison, showing the hourly mean difference (VAD - model) ---
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
fig.suptitle('Mean Difference by Hour of Day (VAD - Model)', fontsize=16)

hh = np.arange(24)

for label, color, ls, hs, sd, hd, dd in datasets:
    ax1.plot(hh, hourly_means(hs, sd), marker='o', markersize=4,
             color=color, linestyle=ls, label=label)
    ax2.plot(hh, hourly_means(hd, dd), marker='o', markersize=4,
             color=color, linestyle=ls, label=label)

ax1.axhline(0, color='black', linewidth=1)  # reference line for zero difference
ax1.set_ylabel('Wind Speed Diff (m/s)', fontsize=14)
ax1.set_title('Wind Speed', fontsize=15)
ax1.tick_params(labelsize=13)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=12)

ax2.axhline(0, color='black', linewidth=1)  # reference line for zero difference
ax2.set_ylabel('Wind Dir Diff (deg)', fontsize=14)
ax2.set_xlabel('Hour of Day (UTC)', fontsize=14)
ax2.set_title('Wind Direction', fontsize=15)
ax2.tick_params(labelsize=13)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=12)
ax2.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('timeseries_by_hour_all_datasets.png', dpi=300, bbox_inches='tight')
plt.show()