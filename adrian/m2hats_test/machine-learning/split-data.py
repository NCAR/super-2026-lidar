"""
prepare_ml_split.py
--------------------
Loads matched HRRR / WindCube VAD wind data and produces an 80/20
train/test split that is stratified by *date* (not by individual
observations) so that no single day bleeds across both sets.

Outputs
-------
  ml_train.npz  –  training set  (~80 % of days)
  ml_test.npz   –  test set      (~20 % of days)

Each .npz contains the arrays:
  u_hrrr, v_hrrr   – HRRR u / v components  (m/s)
  u_vad,  v_vad    – VAD  u / v components  (m/s)
  height            – height AGL             (m)
  epoch             – Unix epoch of the VAD match (s)
  date_label        – YYYYMMDD string for each observation
"""

import netCDF4 as nc
import numpy as np
import glob
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# -- paths ------------------------------------------------------------------
HRRR_BASE = '/scr/isf_apg/models/m2hats/hrrr/'
VAD_BASE  = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
OUT_DIR   = '.'          # directory for ml_train.npz / ml_test.npz

# -- tuneable constants ------------------------------------------------------
HEIGHT_MIN  = 100    # m AGL – lowest height to include
HEIGHT_MAX  = 2000   # m AGL – highest height to include
HEIGHT_TOL  = 25     # m     – max allowed mismatch between VAD and HRRR gate
TIME_TOL    = 900    # s     – max allowed time offset for a valid match
FILL_VALUE  = -9999.0
TRAIN_FRAC  = 0.80
RANDOM_SEED = 42

# -- containers (one entry per matched observation) -------------------------
records = {
    'u_hrrr':     [],
    'v_hrrr':     [],
    'u_vad':      [],
    'v_vad':      [],
    'height':     [],
    'epoch':      [],
    'date_label': [],
}

# -- main data-loading loop -------------------------------------------------
vad_files = sorted(glob.glob(VAD_BASE + '30min_winds_*.nc'))
print(f"Found {len(vad_files)} VAD file(s) to process …")

for vad_file in vad_files:
    date = os.path.basename(vad_file).replace('30min_winds_', '').replace('.nc', '')
    h_files = sorted(glob.glob(HRRR_BASE + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))

    if not h_files:
        continue

    # -- load VAD ----------------------------------------------------------
    vad    = nc.Dataset(vad_file)
    u_vad  = vad.variables['u'][:]
    v_vad  = vad.variables['v'][:]
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_v = vad.variables['time'][:]
    vad.close()

    u_vad    = np.ma.masked_where(u_vad == FILL_VALUE, u_vad)
    v_vad    = np.ma.masked_where(v_vad == FILL_VALUE, v_vad)
    vad_epoch = base_t + time_v

    # -- loop over HRRR files for this date --------------------------------
    for f in h_files:
        ds       = nc.Dataset(f)
        hrrr_ws  = ds.variables['wspd'][:]
        hrrr_dir = ds.variables['wdir'][:]
        hrrr_agl = ds.variables['height'][:]
        et       = int(ds.variables['time'][0])
        ds.close()

        # meteorological ? Cartesian
        wdir_rad = np.radians(hrrr_dir)
        hrrr_u   = -hrrr_ws * np.sin(wdir_rad)
        hrrr_v   = -hrrr_ws * np.cos(wdir_rad)

        # nearest VAD time
        ti = np.argmin(np.abs(vad_epoch - et))
        if abs(vad_epoch[ti] - et) > TIME_TOL:
            continue

        valid = (~np.ma.getmaskarray(u_vad[ti]) &
                 ~np.ma.getmaskarray(v_vad[ti]))
        if not valid.any():
            continue

        h_valid   = height[valid]
        idx_valid = np.where(valid)[0]

        for k in range(len(hrrr_agl)):
            if not (HEIGHT_MIN <= hrrr_agl[k] <= HEIGHT_MAX):
                continue

            j = np.argmin(np.abs(h_valid - hrrr_agl[k]))
            if np.abs(h_valid[j] - hrrr_agl[k]) > HEIGHT_TOL:
                continue

            idx = idx_valid[j]
            if np.ma.is_masked(u_vad[ti, idx]) or np.ma.is_masked(v_vad[ti, idx]):
                continue

            records['u_hrrr'].append(float(hrrr_u[k]))
            records['v_hrrr'].append(float(hrrr_v[k]))
            records['u_vad'].append(float(u_vad[ti, idx]))
            records['v_vad'].append(float(v_vad[ti, idx]))
            records['height'].append(float(hrrr_agl[k]))
            records['epoch'].append(et)
            records['date_label'].append(date)

# -- convert to arrays ------------------------------------------------------
for key in ('u_hrrr', 'v_hrrr', 'u_vad', 'v_vad', 'height', 'epoch'):
    records[key] = np.array(records[key])
records['date_label'] = np.array(records['date_label'])

n_total = len(records['u_hrrr'])
print(f"Total matched observations: {n_total}")

if n_total == 0:
    raise RuntimeError("No matched observations found – check your data paths.")

# -- date-based 80/20 split -------------------------------------------------
# Splitting by date prevents temporal leakage: observations from the same
# day share atmospheric conditions, so mixing them across train/test would
# give an over-optimistic evaluation of model generalisation.

unique_dates = np.unique(records['date_label'])
n_dates      = len(unique_dates)
print(f"Unique dates: {n_dates}")

rng          = np.random.default_rng(RANDOM_SEED)
shuffled     = rng.permutation(unique_dates)
n_train      = int(np.ceil(n_dates * TRAIN_FRAC))   # round up so train >= 80 %
train_dates  = set(shuffled[:n_train])
test_dates   = set(shuffled[n_train:])

train_mask = np.array([d in train_dates for d in records['date_label']])
test_mask  = ~train_mask

# -- save -------------------------------------------------------------------
def save_split(path, mask, label):
    np.savez(
        path,
        u_hrrr     = records['u_hrrr'][mask],
        v_hrrr     = records['v_hrrr'][mask],
        u_vad      = records['u_vad'][mask],
        v_vad      = records['v_vad'][mask],
        height     = records['height'][mask],
        epoch      = records['epoch'][mask],
        date_label = records['date_label'][mask],
    )
    n     = mask.sum()
    dates = np.unique(records['date_label'][mask])
    print(f"\n{label}")
    print(f"  Observations : {n:>7d}  ({100*n/n_total:.1f} %)")
    print(f"  Days         : {len(dates):>7d}  ({100*len(dates)/n_dates:.1f} %)")
    print(f"  Date range   : {dates[0]}  ?  {dates[-1]}")
    print(f"  Saved to     : {path}")

save_split(os.path.join(OUT_DIR, 'ml_train.npz'), train_mask, 'TRAINING SET')
save_split(os.path.join(OUT_DIR, 'ml_test.npz'),  test_mask,  'TEST SET')

# -- quick sanity-check summary ---------------------------------------------
print("\n-- Summary --------------------------------------------------------")
train = np.load(os.path.join(OUT_DIR, 'ml_train.npz'), allow_pickle=True)
test  = np.load(os.path.join(OUT_DIR, 'ml_test.npz'),  allow_pickle=True)

for label, ds in [('Train', train), ('Test', test)]:
    u_diff = ds['u_vad'] - ds['u_hrrr']
    v_diff = ds['v_vad'] - ds['v_hrrr']
    print(f"\n{label}:")
    print(f"  u  MAD={np.mean(np.abs(u_diff)):.3f}  SD={np.std(u_diff):.3f}")
    print(f"  v  MAD={np.mean(np.abs(v_diff)):.3f}  SD={np.std(v_diff):.3f}")

print("\nDone.  Load the splits in your ML script with:")
print("  train = np.load('ml_train.npz', allow_pickle=True)")
print("  X_train = np.column_stack([train['u_hrrr'], train['v_hrrr'], train['height']])")
print("  y_train = np.column_stack([train['u_vad'],  train['v_vad']])")