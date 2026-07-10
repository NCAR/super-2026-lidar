#!/usr/bin/env python
"""
LOTOS Marshall, CLEAN RESTART: direct scatter of profiler vs lidar winds.

Deliberately minimal to eliminate pipeline-bug classes from the last attempt:
  * uses each product's STORED variables (MAPR wspd/wdir; lidar
    wind_speed/wind_direction) -- no deriving speed from u/v
  * nearest-gate height matching within +-HEIGHT_TOL -- NO interpolation
    (np.interp across gappy per-scan VAD profiles can fabricate values)
  * nearest-scan time matching within +-TIME_TOL_S
  * prints every matching decision stat so nothing is silent

Output: two scatterplots (speed, direction) + bias/correlation stats.
MAPR winds.05 vs lidar vad_cnr31. Shows plots; saves nothing.
"""

import glob, os, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
MAPR_DIR  = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr31"
SITE_ALT_FALLBACK = 1742.0
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 300          # winds.05 vs per-scan VAD
HEIGHT_TOL = 25.0         # m; nearest gate must be within this -- NO interpolation
# -----------------------------------------------------------------------------

def to_nan(v, fill):
    a = v[:]
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    a = np.asarray(a, float); a[a == fill] = np.nan
    return a

def mapr_epoch(ds): return float(np.asarray(ds["base_time"][...])) + np.asarray(ds["time"][:], float)
def lidar_epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    if "time" in v:
        t = np.asarray(v["time"][:], float)
        if np.nanmax(t) > 1e8: return t
        if "base_time" in v: return float(np.asarray(v["base_time"][...])) + t
        return t
    return None
def lg(ds, names):
    for n in (names if isinstance(names, list) else [names]):
        if n in ds.variables: return to_nan(ds[n], VAD_FILL)
    return None

lid = {re.search(r"(\d{8})", os.path.basename(f)).group(1): f
       for f in glob.glob(os.path.join(LIDAR_DIR, "VAD_*.nc"))}
mapr_files = sorted(glob.glob(os.path.join(MAPR_DIR, "prof449.*.winds.05.nc")))
print(f"MAPR days: {len(mapr_files)}   lidar days: {len(lid)}")

MS, LS, MD, LD, H = [], [], [], [], []      # matched speeds, directions, heights
stat = dict(profiles=0, no_time=0, no_height=0, pairs=0)
vars_reported = False

for mf in mapr_files:
    date = re.search(r"prof449\.(\d{8})\.", os.path.basename(mf)).group(1)
    if date not in lid: continue
    m = nc.Dataset(mf)
    alt = float(np.asarray(m["alt"][...]))
    if not np.isfinite(alt): alt = SITE_ALT_FALLBACK
    m_t = mapr_epoch(m); m_h = to_nan(m["height"], MAPR_FILL)
    m_sp, m_di = to_nan(m["wspd"], MAPR_FILL), to_nan(m["wdir"], MAPR_FILL)
    m.close()
    m_h_agl = m_h - alt if np.nanmin(m_h) > alt - 200 else m_h

    d = nc.Dataset(lid[date])
    l_t = lidar_epoch(d)
    l_h = np.asarray(lg(d, "height"), float)
    l_sp = lg(d, ["wind_speed", "wspd", "speed"])
    l_di = lg(d, ["wind_direction", "wdir", "direction"])
    if not vars_reported:
        print(f"[vars] lidar speed: {'wind_speed-family FOUND' if l_sp is not None else 'MISSING'}"
              f"   direction: {'FOUND' if l_di is not None else 'MISSING'}"
              f"   height ndim: {l_h.ndim}")
        vars_reported = True
    d.close()
    if l_t is None or l_sp is None: continue
    l2d = (l_h.ndim == 2)

    for i in range(m_sp.shape[0]):
        stat["profiles"] += 1
        j = int(np.argmin(np.abs(l_t - m_t[i])))
        if abs(l_t[j] - m_t[i]) > TIME_TOL_S:
            stat["no_time"] += 1; continue
        lh = l_h[j] if l2d else l_h
        lsp, ldi = l_sp[j], (l_di[j] if l_di is not None else np.full_like(l_sp[j], np.nan))
        valid = np.isfinite(lh) & np.isfinite(lsp)
        if not valid.any():
            stat["no_height"] += 1; continue
        vh, vsp, vdi = lh[valid], lsp[valid], ldi[valid]
        for g in range(len(m_h_agl[i])):
            hm, sm, dm = m_h_agl[i][g], m_sp[i][g], m_di[i][g]
            if not (np.isfinite(hm) and np.isfinite(sm)): continue
            k = int(np.argmin(np.abs(vh - hm)))
            if abs(vh[k] - hm) > HEIGHT_TOL: continue     # nearest gate only, no interp
            MS.append(sm); LS.append(vsp[k])
            MD.append(dm); LD.append(vdi[k]); H.append(hm)
            stat["pairs"] += 1

MS, LS = np.array(MS), np.array(LS)
MD, LD, H = np.array(MD), np.array(LD), np.array(H)
print(f"[match] profiles {stat['profiles']}  no-time {stat['no_time']}  "
      f"no-valid-lidar {stat['no_height']}  matched pairs {stat['pairs']}")
if stat["pairs"] == 0:
    raise SystemExit("No pairs -- paste the [vars]/[match] lines back.")

dspd = MS - LS
dok = np.isfinite(MD) & np.isfinite(LD) & (MS >= 3) & (LS >= 3)
ddir = ((MD[dok] - LD[dok] + 180) % 360) - 180
r_spd = np.corrcoef(LS, MS)[0, 1]
print(f"\nspeed:     n={len(MS)}   r={r_spd:.3f}   median diff (MAPR-lidar) "
      f"{np.median(dspd):+.2f} m/s   MAD {np.median(np.abs(dspd - np.median(dspd))):.2f}")
print(f"direction: n={dok.sum()} (both >=3 m/s)   median diff {np.median(ddir):+.1f} deg   "
      f"MAD {np.median(np.abs(ddir - np.median(ddir))):.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 6))
mx = np.nanpercentile(np.concatenate([LS, MS]), 99.5)
ax[0].scatter(LS, MS, s=4, alpha=0.15, color="steelblue")
ax[0].plot([0, mx], [0, mx], "k--", lw=1, label="1:1")
ax[0].set_xlim(0, mx); ax[0].set_ylim(0, mx)
ax[0].set_xlabel("lidar wind speed (m/s)"); ax[0].set_ylabel("MAPR wind speed (m/s)")
ax[0].set_title(f"Speed (n={len(MS)}, r={r_spd:.2f}, "
                f"median Δ {np.median(dspd):+.2f} m/s)")
ax[0].legend()

ax[1].scatter(LD[dok], MD[dok], s=4, alpha=0.15, color="seagreen")
ax[1].plot([0, 360], [0, 360], "k--", lw=1, label="1:1")
ax[1].set_xlim(0, 360); ax[1].set_ylim(0, 360)
ax[1].set_xlabel("lidar wind direction (deg)"); ax[1].set_ylabel("MAPR wind direction (deg)")
ax[1].set_title(f"Direction, both ≥3 m/s (n={dok.sum()}, "
                f"median Δ {np.median(ddir):+.1f}°)")
ax[1].legend()

fig.suptitle("LOTOS Marshall: MAPR winds.05 vs lidar VAD cnr31 — stored variables, "
             "nearest gate, no interpolation", y=1.0)
plt.tight_layout(); plt.show()
print("done")

