import netCDF4 as nc
import numpy as np
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

n_guarded = 0

print(">>> RUNNING GUARDED VERSION <<<")

vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr40_error/'
hrrr_base = '/scr/isf_apg/models/lotos2025/hrrr/'
era5_base = '/scr/isf_apg/models/lotos2025/era5/'

# read site elevation once from the first VAD file
_first_vad = sorted(glob.glob(vad_base + 'VAD_*.nc'))[0]
_v = nc.Dataset(_first_vad)
site_alt = float(_v.variables['alt'][:])
_v.close()
print(f"Site elevation read from VAD: {site_alt:.1f} m MSL\n")


def read_vad(vad_file):
    vad = nc.Dataset(vad_file)
    ws = np.ma.masked_where(vad.variables['wind_speed'][:] == -9999.0, vad.variables['wind_speed'][:])
    wd = np.ma.masked_where(vad.variables['wind_direction'][:] == -9999.0, vad.variables['wind_direction'][:])
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()
    return ws, wd, height, base_t + time_vad


def match_one(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
              hours_s, sdiff, hours_d, ddiff):
    """Match a single model time slice against the VAD and append diffs."""
    global n_guarded
    hour = datetime.fromtimestamp(et, tz=timezone.utc).hour

    ti = np.argmin(np.abs(vad_epoch - et))
    if abs(vad_epoch[ti] - et) > 900:
        return

    valid = ~np.ma.getmaskarray(ws_vad[ti])
    if not valid.any():
        return

    h_valid = height[valid]
    idx_valid = np.where(valid)[0]

    for k in range(len(m_agl)):
        if not (100 <= m_agl[k] <= 2000):
            continue
        j = np.argmin(np.abs(h_valid - m_agl[k]))
        if np.abs(h_valid[j] - m_agl[k]) > 25:
            continue
        idx = idx_valid[j]

        if np.ma.is_masked(ws_vad[ti, idx]):
            continue
        # physical sanity guard for the _error dataset
        if not np.isfinite(ws_vad[ti, idx]) or ws_vad[ti, idx] < 0 or ws_vad[ti, idx] > 60:
            global n_guarded
            n_guarded += 1
            continue

        hours_s.append(hour)
        sdiff.append(float(ws_vad[ti, idx] - m_ws[k]))

        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180
            hours_d.append(hour)
            ddiff.append(dd)


def run_hrrr():
    hours_s, sdiff, hours_d, ddiff = [], [], [], []
    for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        # mixed ISS1/ISS2 suffixes in this project -> wildcard the suffix
        h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS*.nc'))
        if not h_files:
            continue
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
        for f in h_files:
            ds = nc.Dataset(f)
            m_ws  = ds.variables['wspd'][:]
            m_dir = ds.variables['wdir'][:]
            m_agl = ds.variables['height'][:]
            et = int(ds.variables['time'][0])
            ds.close()
            match_one(m_ws, m_dir, m_agl, et, ws_vad, wd_vad, height, vad_epoch,
                      hours_s, sdiff, hours_d, ddiff)
    return np.array(hours_s), np.array(sdiff), np.array(hours_d), np.array(ddiff)


def run_era5():
    hours_s, sdiff, hours_d, ddiff = [], [], [], []
    for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
        date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
        e_file = era5_base + 'era5_pressure_' + date + '_lotos2025.nc'
        if not os.path.exists(e_file):
            continue
        ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)

        ds = nc.Dataset(e_file)
        u_all = ds.variables['u'][:, :, 0, 0]   # (24, 37)
        v_all = ds.variables['v'][:, :, 0, 0]
        z_all = ds.variables['z'][:, :, 0, 0]
        vt_all = ds.variables['valid_time'][:]
        ds.close()

        for t in range(len(vt_all)):
            u, v, z = u_all[t], v_all[t], z_all[t]
            m_ws  = np.sqrt(u**2 + v**2)
            m_dir = np.degrees(np.arctan2(-u, -v)) % 360
            m_agl = z / 9.80665 - site_alt
            match_one(m_ws, m_dir, m_agl, int(vt_all[t]), ws_vad, wd_vad, height, vad_epoch,
                      hours_s, sdiff, hours_d, ddiff)
    return np.array(hours_s), np.array(sdiff), np.array(hours_d), np.array(ddiff)


# --- run both ---
print("Running HRRR comparison...")
h_hs, h_sd, h_hd, h_dd = run_hrrr()
print("Running ERA5 comparison...")
e_hs, e_sd, e_hd, e_dd = run_era5()

# --- diagnostics ---
print(f"HRRR sdiff min/max: {h_sd.min():.1f} / {h_sd.max():.1f}")
print(f"ERA5 sdiff min/max: {e_sd.min():.1f} / {e_sd.max():.1f}")
print(f"Points rejected by sanity guard: {n_guarded}")


# --- output helpers ---
def overall(diff):
    if len(diff) == 0:
        return None
    return dict(mean=np.mean(diff), std=np.std(diff), mad=np.mean(np.abs(diff)),
                mn=diff.min(), mx=diff.max(), rng=diff.max() - diff.min(), n=len(diff))


def hour_means(hour_arr, diff_arr):
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
    hm, hr, hc = hour_means(h_hour, h_diff)
    em, er, ec = hour_means(e_hour, e_diff)
    print(f"\n{title}")
    print(f"{'':>4} | {'--- HRRR ---':^26} | {'--- ERA5 ---':^26}")
    print(f"{'hr':>4} | {'mean':>8} {'range':>8} {'n':>6} | {'mean':>8} {'range':>8} {'n':>6}")
    print("-" * 64)
    for h in range(24):
        def fmt(m, r, c):
            return f"{'--':>8} {'--':>8} {c:>6}" if c == 0 else f"{m:>8.2f} {r:>8.2f} {c:>6}"
        print(f"{h:>4} | {fmt(hm[h], hr[h], hc[h])} | {fmt(em[h], er[h], ec[h])}")


print(f"\n{'='*64}")
print("VAD vs HRRR and ERA5 - LOTOS2025  (vad_cnr40_error dataset, guarded)")
print(f"{'='*64}")

print_overall_side_by_side("WIND SPEED overall [m/s]", h_sd, e_sd)
print_overall_side_by_side("WIND DIRECTION overall [deg]", h_dd, e_dd)
print_hourly_side_by_side("WIND SPEED difference by hour [m/s]", h_hs, h_sd, e_hs, e_sd)
print_hourly_side_by_side("WIND DIRECTION difference by hour [deg]", h_hd, h_dd, e_hd, e_dd)