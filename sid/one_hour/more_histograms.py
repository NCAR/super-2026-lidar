import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/m2hats/era5/'
vad_base  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

SPD_TOL = 2.0    # m/s
DIR_TOL = 30.0   # deg
SPD_MIN = 2.0    # m/s

era5_ws_m, vad_ws_m, era5_dir_m, vad_dir_m = [], [], [], []
snr_m, resid_m, corr_m = [], [], []
w_m, unp_m, vnp_m, wnp_m = [], [], [], []

for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    p_files = sorted(glob.glob(era5_base + date + '/era5_pressure_' + date + '_*_ISS1.nc'))
    s_files = sorted(glob.glob(era5_base + date + '/era5_surface_'  + date + '_*_ISS1.nc'))
    if not p_files or not s_files:
        continue

    sds = nc.Dataset(s_files[0])
    surf_alt = float(sds.variables['z'][0, 0, 0]) / 9.80665
    sds.close()

    vad = nc.Dataset(vad_file)
    ws_vad   = vad.variables['wind_speed'][:]
    wd_vad   = vad.variables['wind_direction'][:]
    w_vad    = np.asarray(vad.variables['w'][:])
    corr     = np.asarray(vad.variables['correlation'][:])
    resid    = np.asarray(vad.variables['residual'][:])
    snr      = np.asarray(vad.variables['mean_snr'][:])
    unp      = np.asarray(vad.variables['u_npoints'][:])
    vnp      = np.asarray(vad.variables['v_npoints'][:])
    wnp      = np.asarray(vad.variables['w_npoints'][:])
    height   = vad.variables['height'][:]
    base_t   = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()

    ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)

    good = (~np.ma.getmaskarray(ws_vad)) & (np.asarray(ws_vad) >= SPD_MIN)

    vad_epoch = base_t + time_vad

    for f in p_files:
        ds = nc.Dataset(f)
        u = ds.variables['u'][0, :, 0, 0]
        v = ds.variables['v'][0, :, 0, 0]
        z = ds.variables['z'][0, :, 0, 0]
        et = int(ds.variables['valid_time'][0])
        ds.close()

        era5_ws  = np.sqrt(u**2 + v**2)
        era5_dir = np.degrees(np.arctan2(-u, -v)) % 360
        era5_agl = z / 9.80665 - surf_alt

        ti = np.argmin(np.abs(vad_epoch - et))
        if abs(vad_epoch[ti] - et) > 900:
            continue
        valid = good[ti]
        if not valid.any():
            continue
        h_valid = height[valid]
        idx_valid = np.where(valid)[0]

        for k in range(len(era5_agl)):
            if not (100 <= era5_agl[k] <= 2000):
                continue
            j = np.argmin(np.abs(h_valid - era5_agl[k]))
            if np.abs(h_valid[j] - era5_agl[k]) > 25:
                continue
            idx = idx_valid[j]
            era5_ws_m.append(era5_ws[k])
            vad_ws_m.append(ws_vad[ti, idx])
            era5_dir_m.append(era5_dir[k])
            vad_dir_m.append(wd_vad[ti, idx])
            snr_m.append(snr[ti, idx])
            resid_m.append(resid[ti, idx])
            corr_m.append(corr[ti, idx])
            w_m.append(w_vad[ti, idx])
            unp_m.append(unp[ti, idx])
            vnp_m.append(vnp[ti, idx])
            wnp_m.append(wnp[ti, idx])

era5_ws_m  = np.array(era5_ws_m);  vad_ws_m  = np.array(vad_ws_m)
era5_dir_m = np.array(era5_dir_m); vad_dir_m = np.array(vad_dir_m)
snr_m   = np.array(snr_m)
resid_m = np.array(resid_m)
corr_m  = np.array(corr_m)
w_m     = np.array(w_m)
unp_m   = np.array(unp_m)
vnp_m   = np.array(vnp_m)
wnp_m   = np.array(wnp_m)
print(f"Matched points: {len(era5_ws_m)}")

if len(era5_ws_m) == 0:
    raise SystemExit("Zero matches.")

# --- disagreement flags ---
spd_diff = np.abs(vad_ws_m - era5_ws_m)
dir_diff = np.abs(((vad_dir_m - era5_dir_m + 180) % 360) - 180)
disagree = (spd_diff > SPD_TOL) | (dir_diff > DIR_TOL)
print(f"Overall disagreement: {100*disagree.mean():.1f}%")

# --- binned disagreement fraction per metric ---
def plot_metric(metric, name, units, nbins=20):
    ok = (metric != -9999.0) & ~np.isnan(metric)
    m, d = metric[ok], disagree[ok]
    if len(m) == 0:
        print(f"No valid data for {name}, skipping")
        return
    edges = np.linspace(np.percentile(m, 1), np.percentile(m, 99), nbins + 1)
    if edges[0] == edges[-1]:
        print(f"{name} has no spread, skipping")
        return
    centers, fracs, counts = [], [], []
    for i in range(nbins):
        sel = (m >= edges[i]) & (m < edges[i+1])
        if sel.sum() < 10:
            continue
        centers.append(0.5*(edges[i]+edges[i+1]))
        fracs.append(d[sel].mean())
        counts.append(sel.sum())

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.bar(centers, 100*np.array(fracs),
            width=0.9*(edges[1]-edges[0]), color='crimson', alpha=0.8)
    ax1.set_xlabel(f'{name} ({units})')
    ax1.set_ylabel('Disagreement (%)', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax2 = ax1.twinx()
    ax2.plot(centers, counts, 'k.-', markersize=4, linewidth=0.8)
    ax2.set_ylabel('Points per bin', color='black')
    ax1.set_title(f'M2HATS: ERA5-VAD disagreement vs {name}\n'
                  f'(speed >{SPD_TOL:g} m/s or dir >{DIR_TOL:g} deg)')
    plt.tight_layout()

plot_metric(snr_m,         'Mean SNR',               'unitless')
plot_metric(resid_m,       'Fit residual',           'm/s')
plot_metric(corr_m,        'Correlation',            'unitless')
plot_metric(np.abs(w_m),   '|Vertical wind|',        'm/s')
plot_metric(unp_m,         'u consensus points',     'count')
plot_metric(vnp_m,         'v consensus points',     'count')
plot_metric(wnp_m,         'w consensus points',     'count')
plt.show()

