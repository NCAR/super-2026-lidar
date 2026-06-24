import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/lotos2025/era5/'
vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'

TOL_ABS   = 1.0     # m/s, outlier band
SPD_MIN   = 2.0     # m/s, drop low winds
CORR_MIN  = 0.8
RESID_MAX = 2.25
SNR_MIN   = -28.0

era5_ws_m, vad_ws_m, plev_m = [], [], []

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
    p_files = sorted(glob.glob(era5_base + date + '/era5_pressure_' + date + '_*.nc'))
    s_files = sorted(glob.glob(era5_base + date + '/era5_surface_'  + date + '_*.nc'))
    if not p_files or not s_files:
        continue

    sds = nc.Dataset(s_files[0])
    surf_alt = float(sds.variables['z'][0, 0, 0]) / 9.80665
    sds.close()

    vad = nc.Dataset(vad_file)
    u_vad    = vad.variables['u'][:]
    v_vad    = vad.variables['v'][:]
    corr     = np.asarray(vad.variables['correlation'][:])
    resid    = np.asarray(vad.variables['residual'][:])
    snr      = np.asarray(vad.variables['mean_snr'][:])
    height   = vad.variables['height'][:]
    base_t   = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()

    u_vad = np.ma.masked_where((u_vad == -9999.0) | np.isnan(u_vad), u_vad)
    v_vad = np.ma.masked_where((v_vad == -9999.0) | np.isnan(v_vad), v_vad)
    ws_vad = np.sqrt(u_vad**2 + v_vad**2)

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
            plev_m.append(float(plev[k]))

era5_ws_m = np.array(era5_ws_m)
vad_ws_m  = np.array(vad_ws_m)
plev_m    = np.array(plev_m)
print(f"Matched points: {len(era5_ws_m)}")

if len(era5_ws_m) == 0:
    raise SystemExit("Zero matches — check QC thresholds, paths, surf_alt.")

# --- LAD fit (reporting) ---
def lad_fit(x, y):
    def l1_loss(p):
        return np.sum(np.abs(y - (p[0]*x + p[1])))
    return minimize(l1_loss, x0=[1.0, 0.0], method='Nelder-Mead').x
s_fit = lad_fit(era5_ws_m, vad_ws_m)

# --- outliers: |VAD - ERA5| > 1 m/s ---
outlier = np.abs(vad_ws_m - era5_ws_m) > TOL_ABS

levels = np.array(sorted(np.unique(plev_m), reverse=True))
total_counts   = np.array([np.sum(plev_m == L) for L in levels])
outlier_counts = np.array([np.sum((plev_m == L) & outlier) for L in levels])
frac = np.divide(outlier_counts, total_counts,
                 out=np.zeros_like(outlier_counts, float), where=total_counts > 0)

print(f"LAD fit: {s_fit[0]:.3f}x + {s_fit[1]:.3f}")
print(f"Total outliers (>{TOL_ABS:g} m/s): {outlier.sum()} of {len(outlier)} ({100*outlier.mean():.1f}%)")

# --- plot ---
fig, ax = plt.subplots(figsize=(8, 7))
ypos = np.arange(len(levels))
ax.barh(ypos, total_counts,   color='lightgray', label='all matched pts (post-QC)')
ax.barh(ypos, outlier_counts, color='crimson',   label=f'outliers (>{TOL_ABS:g} m/s)')
ax.set_yticks(ypos)
ax.set_yticklabels([f"{int(L)}" for L in levels])
ax.invert_yaxis()
ax.set_xlabel('Number of points')
ax.set_ylabel('ERA5 pressure level (hPa)')
ax.set_title(f'LOTOS-2025: wind-speed outliers (>{TOL_ABS:g} m/s) by pressure level')
for i, fr in enumerate(frac):
    ax.text(total_counts[i] + 2, i, f"{100*fr:.0f}%", va='center', fontsize=8)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

