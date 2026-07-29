import netCDF4 as nc
import numpy as np
import glob, os, warnings
warnings.filterwarnings('ignore')

vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_cnr33_witherror/'  # adjust to your path
BAD_THRESH = 60.0  # m/s - anything above this is considered garbage

print(f"{'file':>20} | {'count':>5} | {'max value':>10} | {'height range (m)':>20}")
print("-" * 65)

total_bad = 0
files_affected = 0

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    fname = os.path.basename(vad_file)

    vad = nc.Dataset(vad_file)
    ws = vad.variables['wind_speed'][:]
    height = vad.variables['height'][:]
    vad.close()

    # find values above the physical threshold (ignoring masked points)
    ws_filled = np.ma.filled(ws, 0.0)  # masked -> 0 so they don't trigger
    bad = ws_filled > BAD_THRESH

    if bad.any():
        files_affected += 1
        n_bad = bad.sum()
        total_bad += n_bad
        max_val = ws_filled[bad].max()

        # which heights are affected? bad has shape (time, height)
        bad_height_idx = np.where(bad.any(axis=0))[0]
        h_lo = height[bad_height_idx].min()
        h_hi = height[bad_height_idx].max()

        print(f"{fname:>20} | {n_bad:>5} | {max_val:>10.1f} | {h_lo:>8.0f} - {h_hi:<8.0f}")

print("-" * 65)
print(f"\nFiles affected: {files_affected}")
print(f"Total bad points (> {BAD_THRESH} m/s): {total_bad}")

# also report which height levels are most commonly affected, across all files
print("\nChecking height distribution of bad values across all files...")
height_bad_count = {}
for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    vad = nc.Dataset(vad_file)
    ws = vad.variables['wind_speed'][:]
    height = vad.variables['height'][:]
    vad.close()
    ws_filled = np.ma.filled(ws, 0.0)
    bad = ws_filled > BAD_THRESH
    for hi in np.where(bad.any(axis=0))[0]:
        h = float(height[hi])
        height_bad_count[h] = height_bad_count.get(h, 0) + int(bad[:, hi].sum())

if height_bad_count:
    print(f"{'height (m)':>12} | {'bad count':>10}")
    print("-" * 26)
    for h in sorted(height_bad_count):
        print(f"{h:>12.0f} | {height_bad_count[h]:>10}")
else:
    print("No bad values found above threshold.")