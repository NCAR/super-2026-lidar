import netCDF4 as nc
import numpy as np
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

vad_base  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_cnr_33_witherror/'
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
era5_base = '/scr/isf_apg/models/m2hats/era5/'
site_alt  = 1623.0  # ISS1 site elevation (m MSL) - REPLACE with actual value (only used by ERA5)


def read_hrrr(f):
    """Return (ws, dir, agl, epoch) for an HRRR profile file."""
    ds = nc.Dataset(f)
    ws  = ds.variables['wspd'][:]
    wd  = ds.variables['wdir'][:]
    agl = ds.variables['height'][:]
    et  = int(ds.variables['time'][0])
    ds.close()
    return ws, wd, agl, et


def read_era5(f):
    """Return (ws, dir, agl, epoch) for an ERA5 pressure-level file."""
    ds = nc.Dataset(f)
    u = ds.variables['u'][0, :, 0, 0]
    v = ds.variables['v'][0, :, 0, 0]
    z = ds.variables['z'][0, :, 0, 0]
    et = int(ds.variables['valid_time'][0])
    ds.close()
    ws  = np.sqrt(u**2 + v**2)
    wd  = np.degrees(np.arctan2(-u, -v)) % 360
    agl = z / 9.80665 - site_alt
    return ws, wd, agl, et


def run_comparison(model_base, glob_tmpl, reader):
    """Collect hourly speed/direction differences vs VAD for one model."""
    hours_s, sdiff = [], []
    hours_d, ddiff = [], []

    for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
        date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
        h_files = sorted(glob.glob(model_base + date + glob_tmpl.format(date=date)))
        if not h_files:
            continue

        vad = nc.Dataset(vad_file)
        ws_vad = vad.variables['wind_speed'][:]
        wd_vad = vad.variables['wind_direction'][:]
        height = vad.variables['height'][:]
        base_t = int(vad.variables['base_time'][:])
        time_vad = vad.variables['time'][:]
        vad.close()

        ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
        wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
        vad_epoch = base_t + time_vad

        for f in h_files:
            m_ws, m_dir, m_agl, et = reader(f)
            hour = datetime.fromtimestamp(et, tz=timezone.utc).hour

            ti = np.argmin(np.abs(vad_epoch - et))
            if abs(vad_epoch[ti] - et) > 900:
                continue

            valid = ~np.ma.getmaskarray(ws_vad[ti])
            if not valid.any():
                continue

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

                hours_s.append(hour)
                sdiff.append(float(ws_vad[ti, idx] - m_ws[k]))

                if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
                    dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180
                    hours_d.append(hour)
                    ddiff.append(dd)

    return (np.array(hours_s), np.array(sdiff),
            np.array(hours_d), np.array(ddiff))


# --- run both models ---
print("Running HRRR comparison...")
h_hs, h_sd, h_hd, h_dd = run_comparison(
    hrrr_base, '/hrrr_profile_{date}_*_ISS1.nc', read_hrrr)

print("Running ERA5 comparison...")
e_hs, e_sd, e_hd, e_dd = run_comparison(
    era5_base, '/era5_pressure_{date}_*_ISS1.nc', read_era5)


# --- helpers for side-by-side output ---
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


def print_overall_side_by_side(title, hrrr_diff, era5_diff, unit):
    h, e = overall(hrrr_diff), overall(era5_diff)
    print(f"\n{title}")
    print(f"{'metric':>10} | {'HRRR':>10} | {'ERA5':>10}")
    print("-" * 38)
    keys = [('mean', 'mean'), ('std', 'std'), ('mad', 'mad'),
            ('mn', 'min'), ('mx', 'max'), ('rng', 'range')]
    for key, lbl in keys:
        hv = f"{h[key]:.2f}" if h else "--"
        ev = f"{e[key]:.2f}" if e else "--"
        print(f"{lbl:>10} | {hv:>10} | {ev:>10}")
    hn = h['n'] if h else 0
    en = e['n'] if e else 0
    print(f"{'n points':>10} | {hn:>10} | {en:>10}")


def print_hourly_side_by_side(title, h_hour, h_diff, e_hour, e_diff, unit):
    hm, hr, hc = hour_means(h_hour, h_diff)
    em, er, ec = hour_means(e_hour, e_diff)
    print(f"\n{title}")
    print(f"{'':>4} | {'--- HRRR ---':^26} | {'--- ERA5 ---':^26}")
    print(f"{'hr':>4} | {'mean':>8} {'range':>8} {'n':>6} | {'mean':>8} {'range':>8} {'n':>6}")
    print("-" * 64)
    for h in range(24):
        def fmt(m, r, c):
            if c == 0:
                return f"{'--':>8} {'--':>8} {c:>6}"
            return f"{m:>8.2f} {r:>8.2f} {c:>6}"
        print(f"{h:>4} | {fmt(hm[h], hr[h], hc[h])} | {fmt(em[h], er[h], ec[h])}")


print(f"\n{'='*64}")
print("VAD vs HRRR and ERA5 - M2HATS")
print(f"{'='*64}")

print_overall_side_by_side("WIND SPEED overall [m/s]", h_sd, e_sd, "m/s")
print_overall_side_by_side("WIND DIRECTION overall [deg]", h_dd, e_dd, "deg")

print_hourly_side_by_side("WIND SPEED difference by hour [m/s]",
                          h_hs, h_sd, e_hs, e_sd, "m/s")
print_hourly_side_by_side("WIND DIRECTION difference by hour [deg]",
                          h_hd, h_dd, e_hd, e_dd, "deg")