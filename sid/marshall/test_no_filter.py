import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/lotos2025/era5/'
vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'

era5_ws_m, vad_ws_m, era5_dir_m, vad_dir_m = [], [], [], []

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
    height   = vad.variables['height'][:]
    base_t   = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()

    u_vad = np.ma.masked_where((u_vad == -9999.0) | np.isnan(u_vad), u_vad)
    v_vad = np.ma.masked_where((v_vad == -9999.0) | np.isnan(v_vad), v_vad)
    ws_vad = np.sqrt(u_vad**2 + v_vad**2)
    wd_vad = np.degrees(np.arctan2(-u_vad, -v_vad)) % 360
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
        valid = ~np.ma.getmaskarray(ws_vad[ti])
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

era5_ws_m  = np.array(era5_ws_m)
vad_ws_m   = np.array(vad_ws_m)
era5_dir_m = np.array(era5_dir_m)
vad_dir_m  = np.array(vad_dir_m)
print(f"Matched points: {len(era5_ws_m)}")

if len(era5_ws_m) == 0:
    raise SystemExit("Zero matches — check paths and surf_alt.")

def lad_fit(x, y):
    def l1_loss(p):
        return np.sum(np.abs(y - (p[0]*x + p[1])))
    return minimize(l1_loss, x0=[1.0, 0.0], method='Nelder-Mead').x

sdiff = vad_ws_m - era5_ws_m
s_mad = np.mean(np.abs(sdiff)); s_sd = np.std(sdiff)
s_fit = lad_fit(era5_ws_m, vad_ws_m)

dmask = (~np.isnan(vad_dir_m)) & (~np.isnan(era5_dir_m))
cdiff = ((vad_dir_m[dmask] - era5_dir_m[dmask] + 180) % 360) - 180
d_mad = np.mean(np.abs(cdiff)); d_sd = np.std(cdiff)
d_fit = lad_fit(era5_dir_m[dmask], vad_dir_m[dmask])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('LOTOS-2025  ERA5 vs Windcube VAD')

x_ws  = np.array([0, 20])
x_dir = np.array([0, 360])

# speed panel
ax1.scatter(era5_ws_m, vad_ws_m, s=2, color='black', zorder=1)
ax1.plot(x_ws, s_fit[0]*x_ws + s_fit[1], 'r-',  linewidth=1.5, label='ladfit', zorder=2)
ax1.plot(x_ws, x_ws * 1.05, 'b--', linewidth=1, label='+5%', zorder=2)
ax1.plot(x_ws, x_ws * 0.95, 'b--', linewidth=1, label='-5%', zorder=2)
ax1.plot(x_ws, x_ws, 'g-', linewidth=1, label='1:1', zorder=2)
ax1.set_xlim(0, 20); ax1.set_ylim(0, 20)
ax1.set_xlabel('ERA5 Speed (m/s)'); ax1.set_ylabel('Windcube VAD Speed (m/s)')
ax1.text(0.5, 19.0, f"{len(era5_ws_m)} pts, mad: {s_mad:.1f}, sd {s_sd:.1f}")
ax1.text(0.5, 17.5, f"ladfit: {s_fit[0]:.1f}x + {s_fit[1]:.2f}")
ax1.legend(loc='lower right', fontsize=8)

# direction panel
ax2.scatter(era5_dir_m[dmask], vad_dir_m[dmask], s=2, color='black', zorder=1)
ax2.plot(x_dir, d_fit[0]*x_dir + d_fit[1], 'r-',  linewidth=1.5, label='ladfit', zorder=2)
ax2.plot(x_dir, x_dir * 1.05, 'b--', linewidth=1, label='+5%', zorder=2)
ax2.plot(x_dir, x_dir * 0.95, 'b--', linewidth=1, label='-5%', zorder=2)
ax2.plot(x_dir, x_dir, 'g-', linewidth=1, label='1:1', zorder=2)
ax2.set_xlim(0, 360); ax2.set_ylim(0, 360)
ax2.set_xlabel('ERA5 Dirn (deg)'); ax2.set_ylabel('Windcube VAD Dirn (deg)')
ax2.text(10, 340, f"{int(dmask.sum())} pts, mad: {d_mad:.1f}, sd {d_sd:.1f}")
ax2.text(10, 315, f"ladfit: {d_fit[0]:.1f}x + {d_fit[1]:.0f}")
ax2.legend(loc='lower right', fontsize=8)

plt.show()

