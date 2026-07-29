import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

vad_base  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
era5_base = '/scr/isf_apg/models/m2hats/era5/'

# M2HATS VAD 'alt' variable is masked/unusable -> hardcode from ERA5 surface geopotential
site_alt = 1739.0  # m MSL, ERA5 surface geopotential at M2HATS ISS1
print(f"Site elevation (hardcoded): {site_alt:.1f} m MSL\n")

rows = []  # one dict per matched point

def read_vad(vad_file):
    vad = nc.Dataset(vad_file)
    ws = np.ma.masked_where(vad.variables['wind_speed'][:] == -9999.0, vad.variables['wind_speed'][:])
    wd = np.ma.masked_where(vad.variables['wind_direction'][:] == -9999.0, vad.variables['wind_direction'][:])
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad.close()
    return ws, wd, height, base_t + time_vad


def shear_at(profile_ws, k):
    """Local wind-speed shear using neighbors, absolute value."""
    lo = max(0, k - 1); hi = min(len(profile_ws) - 1, k + 1)
    if hi == lo:
        return np.nan
    return abs(float(profile_ws[hi] - profile_ws[lo]))


def lapse_at(temp_profile, agl_profile, k):
    """Local temperature lapse (dT/dz), K per km."""
    lo = max(0, k - 1); hi = min(len(temp_profile) - 1, k + 1)
    dz = agl_profile[hi] - agl_profile[lo]
    if dz == 0 or not np.isfinite(dz):
        return np.nan
    return float((temp_profile[hi] - temp_profile[lo]) / dz) * 1000.0


def collect(model_name, m_ws, m_dir, m_agl, m_temp, et,
            ws_vad, wd_vad, height, vad_epoch):
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

        # targets are ABSOLUTE errors
        ws_err = abs(float(ws_vad[ti, idx] - m_ws[k]))

        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180
            dir_err = abs(dd)
        else:
            dir_err = np.nan

        rows.append(dict(
            model = 0 if model_name == 'hrrr' else 1,
            hour = hour,
            height = float(m_agl[k]),
            model_ws = float(m_ws[k]),
            shear = shear_at(m_ws, k),
            lapse = lapse_at(m_temp, m_agl, k) if m_temp is not None else np.nan,
            day = et // 86400,
            ws_err = ws_err,
            dir_err = dir_err,
        ))


# --- HRRR pass (per-hour profile files, M2HATS uses ISS1) ---
print("Collecting HRRR...")
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))
    if not h_files:
        continue
    ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
    for f in h_files:
        ds = nc.Dataset(f)
        m_ws  = ds.variables['wspd'][:]
        m_dir = ds.variables['wdir'][:]
        m_agl = ds.variables['height'][:]
        m_temp = ds.variables['temp'][:] if 'temp' in ds.variables else None
        et = int(ds.variables['time'][0])
        ds.close()
        collect('hrrr', m_ws, m_dir, m_agl, m_temp, et, ws_vad, wd_vad, height, vad_epoch)

# --- ERA5 pass (M2HATS: per-hour files in date subdirectories) ---
print("Collecting ERA5...")
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    e_files = sorted(glob.glob(era5_base + date + '/era5_pressure_' + date + '_*_ISS1.nc'))
    if not e_files:
        continue
    ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
    for f in e_files:
        ds = nc.Dataset(f)
        u = ds.variables['u'][0, :, 0, 0]
        v = ds.variables['v'][0, :, 0, 0]
        z = ds.variables['z'][0, :, 0, 0]
        tt = ds.variables['t'][0, :, 0, 0]
        et = int(ds.variables['valid_time'][0])
        ds.close()
        m_ws  = np.sqrt(u**2 + v**2)
        m_dir = np.degrees(np.arctan2(-u, -v)) % 360
        m_agl = z / 9.80665 - site_alt
        collect('era5', m_ws, m_dir, m_agl, tt, et,
                ws_vad, wd_vad, height, vad_epoch)

# --- build arrays ---
feature_names = ['model', 'hour', 'height', 'model_ws', 'shear', 'lapse']
X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
y_ws  = np.array([r['ws_err'] for r in rows], dtype=float)
y_dir = np.array([r['dir_err'] for r in rows], dtype=float)
groups = np.array([r['day'] for r in rows])

print(f"\nTotal rows: {len(rows)}")


def train_and_report(X, y, groups, target_name):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xm, ym, gm = X[mask], y[mask], groups[mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(Xm, ym, gm))

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(Xm[tr], ym[tr])

    r2_tr = rf.score(Xm[tr], ym[tr])
    r2_te = rf.score(Xm[te], ym[te])

    perm = permutation_importance(rf, Xm[te], ym[te], n_repeats=10,
                                  random_state=42, n_jobs=-1)

    # terminal output
    print(f"\n{'='*55}\nTARGET: {target_name}   ({mask.sum()} usable rows)\n{'='*55}")
    print(f"R^2 train: {r2_tr:.3f}   R^2 test: {r2_te:.3f}")
    print("\nFeature importance (impurity):")
    for name, imp in sorted(zip(feature_names, rf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:>10}: {imp:.3f}")
    print("\nPermutation importance (test set):")
    for name, m, s in sorted(zip(feature_names, perm.importances_mean, perm.importances_std),
                             key=lambda x: -x[1]):
        print(f"  {name:>10}: {m:.3f} +/- {s:.3f}")

    return dict(name=target_name, n=int(mask.sum()), r2_tr=r2_tr, r2_te=r2_te,
                perm_mean=perm.importances_mean, perm_std=perm.importances_std,
                impurity=rf.feature_importances_)


def make_table_figure(result, filename):
    """Render one target's results as a poster-quality table figure."""
    order = np.argsort(result['perm_mean'])[::-1]

    col_labels = ['Feature', 'Permutation\nimportance', 'Impurity\nimportance']
    cell_text = []
    for i in order:
        cell_text.append([
            feature_names[i],
            f"{result['perm_mean'][i]:.3f} +/- {result['perm_std'][i]:.3f}",
            f"{result['impurity'][i]:.3f}",
        ])

    n_rows = len(cell_text) + 1
    fig, ax = plt.subplots(figsize=(7, 0.45 * n_rows + 1.1))
    ax.axis('off')

    title = (f"{result['name']}\n"
             f"n = {result['n']:,}   |   "
             f"$R^2$ train = {result['r2_tr']:.3f}   |   "
             f"$R^2$ test = {result['r2_te']:.3f}")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='upper center',
                     colWidths=[0.28, 0.42, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#e8e8e8')
            cell.set_height(cell.get_height() * 1.6)
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {filename}")


# --- run both targets and produce table figures ---
res_ws  = train_and_report(X, y_ws,  groups, "Absolute wind speed error (m/s) - M2HATS consensus")
res_dir = train_and_report(X, y_dir, groups, "Absolute wind direction error (deg) - M2HATS consensus")

make_table_figure(res_ws,  'rf_importance_wind_speed_m2hats_consensus.png')
make_table_figure(res_dir, 'rf_importance_wind_direction_m2hats_consensus.png')