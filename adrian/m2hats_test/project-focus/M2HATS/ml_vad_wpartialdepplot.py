import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import glob, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.model_selection import GroupShuffleSplit

# --- Base directories for VAD lidar data and the two model datasets ---
vad_base  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
era5_base = '/scr/isf_apg/models/m2hats/era5/'

# M2HATS VAD 'alt' variable is masked/unusable -> hardcode from ERA5 surface geopotential.
# Needed to convert ERA5 geopotential height (MSL) into AGL height for matching.
site_alt = 1739.0  # m MSL, ERA5 surface geopotential at M2HATS ISS1
print(f"Site elevation (hardcoded): {site_alt:.1f} m MSL\n")

rows = []  # one dict per matched VAD/model point, later turned into the ML feature/target arrays

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


def shear_at(profile_ws, k):
    """Local wind-speed shear using neighbors, absolute value.
    Approximates d(speed)/d(index) using the levels just above/below k
    (clamped at the profile edges), used as an ML feature."""
    lo = max(0, k - 1); hi = min(len(profile_ws) - 1, k + 1)
    if hi == lo:
        return np.nan
    return abs(float(profile_ws[hi] - profile_ws[lo]))


def lapse_at(temp_profile, agl_profile, k):
    """Local temperature lapse (dT/dz), K per km.
    Uses the levels just above/below k (clamped at the profile edges),
    used as an ML feature (only available when temperature data exists)."""
    lo = max(0, k - 1); hi = min(len(temp_profile) - 1, k + 1)
    dz = agl_profile[hi] - agl_profile[lo]
    if dz == 0 or not np.isfinite(dz):
        return np.nan
    return float((temp_profile[hi] - temp_profile[lo]) / dz) * 1000.0


def collect(model_name, m_ws, m_dir, m_agl, m_temp, et,
            ws_vad, wd_vad, height, vad_epoch):
    """Match a single model time slice (one HRRR profile file, or one ERA5
    profile file) against the closest-in-time VAD profile, pair up height
    levels, compute absolute speed/direction errors plus engineered
    features (shear, lapse, hour, day, etc.), and append one row per
    matched height level to the module-level `rows` list."""
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

        # targets are ABSOLUTE errors (unlike the signed diffs used elsewhere)
        ws_err = abs(float(ws_vad[ti, idx] - m_ws[k]))

        # Direction error only computed when both speeds exceed 2 m/s
        # (direction is unreliable at low speeds) and VAD direction isn't masked
        if m_ws[k] > 2.0 and ws_vad[ti, idx] > 2.0 and not np.ma.is_masked(wd_vad[ti, idx]):
            dd = ((float(wd_vad[ti, idx]) - m_dir[k] + 180) % 360) - 180  # wrap into [-180, 180]
            dir_err = abs(dd)
        else:
            dir_err = np.nan

        # One row of features + targets for this matched (time, height) point
        rows.append(dict(
            model = 0 if model_name == 'hrrr' else 1,     # encode model source as 0/1 feature
            hour = hour,
            height = float(m_agl[k]),
            model_ws = float(m_ws[k]),
            shear = shear_at(m_ws, k),
            lapse = lapse_at(m_temp, m_agl, k) if m_temp is not None else np.nan,
            day = et // 86400,   # integer day-of-epoch, used as the grouping key for train/test split
            ws_err = ws_err,
            dir_err = dir_err,
        ))


# --- HRRR pass (per-hour profile files, M2HATS uses ISS1) ---
print("Collecting HRRR...")
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    # Extract the date string from the VAD filename (e.g. 30min_winds_20230715.nc -> 20230715)
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))
    if not h_files:
        continue  # no HRRR data for this day, skip it
    ws_vad, wd_vad, height, vad_epoch = read_vad(vad_file)
    # A given day can have multiple HRRR profile files (e.g. multiple hourly
    # profiles), so loop over all of them
    for f in h_files:
        ds = nc.Dataset(f)
        m_ws  = ds.variables['wspd'][:]
        m_dir = ds.variables['wdir'][:]
        m_agl = ds.variables['height'][:]
        m_temp = ds.variables['temp'][:] if 'temp' in ds.variables else None  # not all HRRR files carry temperature
        et = int(ds.variables['time'][0])  # single valid time for this HRRR profile file
        ds.close()
        collect('hrrr', m_ws, m_dir, m_agl, m_temp, et, ws_vad, wd_vad, height, vad_epoch)

# --- ERA5 pass (M2HATS: per-hour files in date subdirectories, unlike LOTOS's
# one-file-per-day-with-24-slices layout) ---
print("Collecting ERA5...")
for vad_file in sorted(glob.glob(vad_base + '30min_winds_*.nc')):
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    e_files = sorted(glob.glob(era5_base + date + '/era5_pressure_' + date + '_*_ISS1.nc'))
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
        tt = ds.variables['t'][0, :, 0, 0]  # temperature (used for lapse rate feature)
        et = int(ds.variables['valid_time'][0])  # single valid time for this ERA5 file
        ds.close()
        m_ws  = np.sqrt(u**2 + v**2)                   # wind speed from components
        m_dir = np.degrees(np.arctan2(-u, -v)) % 360   # meteorological wind direction (from-direction, 0-360 deg)
        m_agl = z / 9.80665 - site_alt                 # geopotential -> geopotential height -> AGL
        collect('era5', m_ws, m_dir, m_agl, tt, et,
                ws_vad, wd_vad, height, vad_epoch)

# --- build feature/target arrays for the ML models ---
feature_names = ['model', 'hour', 'height', 'model_ws', 'shear', 'lapse']
X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
y_ws  = np.array([r['ws_err'] for r in rows], dtype=float)    # target 1: |wind speed error|
y_dir = np.array([r['dir_err'] for r in rows], dtype=float)   # target 2: |wind direction error|
groups = np.array([r['day'] for r in rows])                   # day-of-epoch, used to group-split train/test by day

print(f"\nTotal rows: {len(rows)}")


def train_and_report(X, y, groups, target_name):
    """Fit a RandomForestRegressor on the given target, using a single
    group-aware train/test split (grouped by day, so no day's data leaks
    across the split), then print R^2 scores plus impurity-based and
    permutation-based feature importances. Returns a dict with everything
    needed downstream to build the table and PDP figures."""
    # Drop rows with any NaN feature or NaN target (e.g. missing lapse rate,
    # or direction errors skipped due to the low-speed threshold)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xm, ym, gm = X[mask], y[mask], groups[mask]

    # Split by day (group) rather than by row, so the same day never
    # appears in both train and test sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(Xm, ym, gm))

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(Xm[tr], ym[tr])

    r2_tr = rf.score(Xm[tr], ym[tr])
    r2_te = rf.score(Xm[te], ym[te])

    # Permutation importance computed on the held-out test set (more
    # reliable than impurity importance, especially with correlated features)
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
                impurity=rf.feature_importances_,
                model=rf, X_test=Xm[te], y_test=ym[te])


def make_table_figure(result, filename):
    """Render one target's results as a poster-quality table figure,
    listing each feature's permutation and impurity importance, sorted by
    permutation importance (descending), with the R^2 scores in the title."""
    order = np.argsort(result['perm_mean'])[::-1]  # sort features by permutation importance, descending

    col_labels = ['Feature', 'Permutation\nimportance', 'Impurity\nimportance']
    cell_text = []
    for i in order:
        cell_text.append([
            feature_names[i],
            f"{result['perm_mean'][i]:.3f} +/- {result['perm_std'][i]:.3f}",
            f"{result['impurity'][i]:.3f}",
        ])

    n_rows = len(cell_text) + 1  # +1 for the header row
    fig, ax = plt.subplots(figsize=(7, 0.45 * n_rows + 1.1))
    ax.axis('off')  # this figure is a table, not a plot axis

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

    # Style the header row and alternate row shading for readability
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


def make_pdp_figure(result, pdp_features, filename, ylabel):
    """Partial dependence plots for chosen features, one panel each (old
    sklearn API). Shows how the model's predicted error changes as each
    selected feature varies, holding the others at their marginal
    distribution."""
    idx = [feature_names.index(f) for f in pdp_features]

    fig, axes = plt.subplots(1, len(idx), figsize=(5.5 * len(idx), 4.5))
    if len(idx) == 1:
        axes = [axes]  # normalize to a list so the loop below works for a single feature too

    plot_partial_dependence(result['model'], result['X_test'], features=idx,
                            feature_names=feature_names, ax=axes)

    # Consistent axis styling/labeling across all PDP subplots
    for ax in np.ravel(axes):
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.grid(alpha=0.3)

    fig.suptitle(result['name'], fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {filename}")


# --- run both targets, produce tables and PDPs ---
res_ws  = train_and_report(X, y_ws,  groups, "Absolute wind speed error (m/s) - M2HATS consensus")
res_dir = train_and_report(X, y_dir, groups, "Absolute wind direction error (deg) - M2HATS consensus")

make_table_figure(res_ws,  'rf_table_wind_speed_m2hats_consensus_v2.png')
make_table_figure(res_dir, 'rf_table_wind_direction_m2hats_consensus_v2.png')

# PDPs for the top direction features (the hour/lapse/model_ws trio) and speed's top two
make_pdp_figure(res_dir, ['lapse', 'hour', 'model_ws'],
                'rf_pdp_wind_direction_m2hats_consensus_v2.png',
                'Predicted |direction error| (deg)')
make_pdp_figure(res_ws, ['model_ws', 'height'],
                'rf_pdp_wind_speed_m2hats_consensus_v2.png',
                'Predicted |speed error| (m/s)')