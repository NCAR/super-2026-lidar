import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/m2hats/era5/'
vad_base  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'

SPD_TOL   = 2.0     # m/s, agreement threshold
DIR_TOL   = 30.0    # deg, agreement threshold
SPD_MIN   = 2.0     # m/s, drop low winds (direction meaningless below)
CORR_MIN  = 0.8
RESID_MAX = 2.25
SNR_MIN   = -28.0

era5_ws_m, vad_ws_m, era5_dir_m, vad_dir_m, plev_m = [], [], [], [], []

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
    corr     = np.asarray(vad.variables['correlation'][:])
    resid    = np.asarray(vad.variables['residual'][:])
    snr      = np.asarray(vad.variables['mean_snr'][:])
    height   = vad.variables['height'][:]
    base_t   = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()

    ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)

    good = (~np.ma.getmaskarray(ws_vad)) & \
           (corr  >= CORR_MIN) & (corr  != -9999.0) & \
           (resid <= RESID_MAX) & (resid != -9999.0) & \
           (snr   >= SNR_MIN)  & (snr   != -9999.0) & \
           (np.asarray(ws_vad) >= SPD_MIN)

    vad_epoch = base_t + time_vad

    for f in p_files:
        ds = nc.Dataset(f)
        u = ds.variables['u'][0, :, 0, 0]
        v = ds.variables['v'][0, :, 0, 0]
        z = ds.variables['z'][0, :, 0, 0]
        plev = ds.variables['pressure_level'][:]
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
            plev_m.append(float(plev[k]))

era5_ws_m  = np.array(era5_ws_m)
vad_ws_m   = np.array(vad_ws_m)
era5_dir_m = np.array(era5_dir_m)
vad_dir_m  = np.array(vad_dir_m)
plev_m     = np.array(plev_m)
print(f"Matched points: {len(era5_ws_m)}")

if len(era5_ws_m) == 0:
    raise SystemExit("Zero matches — check QC thresholds, surf_alt, era5_agl.")

# --- LAD fit (reporting) ---
def lad_fit(x, y):
    def l1_loss(p):
        return np.sum(np.abs(y - (p[0]*x + p[1])))
    return minimize(l1_loss, x0=[1.0, 0.0], method='Nelder-Mead').x
s_fit = lad_fit(era5_ws_m, vad_ws_m)

# --- agreement: speed within 2 m/s AND direction within 30 deg ---
spd_diff = np.abs(vad_ws_m - era5_ws_m)
dir_diff = np.abs(((vad_dir_m - era5_dir_m + 180) % 360) - 180)
spd_out  = spd_diff > SPD_TOL
dir_out  = dir_diff > DIR_TOL
both_ok  = (~spd_out) & (~dir_out)

levels = np.array(sorted(np.unique(plev_m), reverse=True))
def per_level(mask):
    return np.array([np.sum((plev_m == L) & mask) for L in levels])
total_counts = np.array([np.sum(plev_m == L) for L in levels])
spd_counts   = per_level(spd_out)
dir_counts   = per_level(dir_out)
spd_frac = np.divide(spd_counts, total_counts, out=np.zeros(len(levels)), where=total_counts > 0)
dir_frac = np.divide(dir_counts, total_counts, out=np.zeros(len(levels)), where=total_counts > 0)

print(f"LAD fit: {s_fit[0]:.3f}x + {s_fit[1]:.3f}")
print(f"Speed disagree (>{SPD_TOL:g} m/s): {spd_out.sum()}/{len(spd_out)} ({100*spd_out.mean():.1f}%)")
print(f"Dir disagree   (>{DIR_TOL:g} deg): {dir_out.sum()}/{len(dir_out)} ({100*dir_out.mean():.1f}%)")
print(f"Agree on both:  {both_ok.sum()}/{len(both_ok)} ({100*both_ok.mean():.1f}%)")

# --- plot: two panels ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
fig.suptitle('M2HATS: model-VAD disagreement by pressure level')
ypos = np.arange(len(levels))

ax1.barh(ypos, total_counts, color='lightgray', label='all matched pts')
ax1.barh(ypos, spd_counts,   color='crimson',   label=f'speed off >{SPD_TOL:g} m/s')
ax1.set_yticks(ypos); ax1.set_yticklabels([f"{int(L)}" for L in levels])
ax1.invert_yaxis()
ax1.set_xlabel('Number of points'); ax1.set_ylabel('ERA5 pressure level (hPa)')
ax1.set_title('Speed')
for i, fr in enumerate(spd_frac):
    ax1.text(total_counts[i] + 2, i, f"{100*fr:.0f}%", va='center', fontsize=8)
ax1.legend(loc='lower right', fontsize=8)

ax2.barh(ypos, total_counts, color='lightgray', label='all matched pts')
ax2.barh(ypos, dir_counts,   color='darkorange', label=f'dir off >{DIR_TOL:g} deg')
ax2.set_xlabel('Number of points')
ax2.set_title('Direction')
for i, fr in enumerate(dir_frac):
    ax2.text(total_counts[i] + 2, i, f"{100*fr:.0f}%", va='center', fontsize=8)
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()

