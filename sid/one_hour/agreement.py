#!/usr/bin/env python
"""
ERA5 vs Windcube VAD lidar, M2HATS ISS1 — "agreement" version.

A point is GOOD when ERA5 wind speed is within +-GOOD_REL of the VAD speed
(relative band around 1:1), evaluated only where VAD speed >= SPD_FLOOR.
Agreement is then binned across the same VAD quality metrics used before.

Datum note: ERA5 altitude = z/9.80665 is geopotential height (MSL); VAD height
is AGL. Both are put on AGL (subtract SITE_ALT_M) before matching.

Displays (does not save):
  ERA5 vs VAD speed scatter, coloured by height AGL
  2x4 grid: in-band fraction across quality metrics

Run on mercury:  python compare_era5_vad_30min.py   (needs a display)
"""

import matplotlib.pyplot as plt            # ssh -X / Jupyter / VS Code remote
import netCDF4 as nc
import numpy as np
import glob, os, warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
ERA5_DIR = "/scr/isf_apg/models/m2hats/era5"
VAD_DIR  = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"

VAD_FILL    = -9999.0
SITE_ALT_M  = 1653.0     # Tonopah Airport elevation (m MSL) — verify against printout
ERA5_TOL_S  = 1800       # max |t_era5 - t_vad| for a time match (s); ERA5 is hourly
GOOD_REL    = 0.20       # |ERA5-VAD|/VAD within this = in-band
SPD_FLOOR   = 2.0        # m/s : skip relative band below this (0 = off)
G           = 9.80665    # geopotential -> geometric height
# -----------------------------------------------------------------------------

def to_nan(var, fill):
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    a = np.asarray(a, dtype=float)
    a[a == fill] = np.nan
    return a

def vad_epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    t = np.asarray(v["time"][:], dtype=float)
    if np.nanmax(t) > 1e8:
        return t
    if "base_time" in v:
        return float(np.asarray(v["base_time"][...])) + t
    return t

def vget(d, name, fill, like):
    if name in d.variables:
        return to_nan(d[name], fill)
    return np.full_like(like, np.nan)

# ----------------------------------------------------------------------------- gather
vad_files = sorted(glob.glob(os.path.join(VAD_DIR, "30min_winds_*.nc")))
if not vad_files:
    raise SystemExit("No VAD files found — check VAD_DIR")

E, S_v, H = [], [], []                       # era5 speed, vad speed, height AGL
SNR, RES, COR, W, UNP, VNP, WNP = [], [], [], [], [], [], []
n_days = 0
e5_agl_lo, e5_agl_hi = np.inf, -np.inf       # track ERA5 AGL coverage for the printout

for vf in vad_files:
    date = os.path.basename(vf).replace("30min_winds_", "").replace(".nc", "")
    efiles = sorted(glob.glob(os.path.join(ERA5_DIR, date, f"era5_pressure_{date}_*_ISS1.nc")))
    if not efiles:
        continue

    # --- ERA5: one (t, alt_agl, speed) profile per hour ---
    e5 = []
    for ef in efiles:
        ds = nc.Dataset(ef)
        u = np.asarray(ds["u"][0, :, 0, 0], dtype=float)
        v = np.asarray(ds["v"][0, :, 0, 0], dtype=float)
        z = np.asarray(ds["z"][0, :, 0, 0], dtype=float)
        t = float(np.asarray(ds["valid_time"][0]))
        ds.close()
        alt_agl = z / G - SITE_ALT_M
        spd = np.hypot(u, v)
        ok = np.isfinite(alt_agl) & np.isfinite(spd)
        if ok.sum() < 2:
            continue
        order = np.argsort(alt_agl[ok])
        e5.append((t, alt_agl[ok][order], spd[ok][order]))
        e5_agl_lo = min(e5_agl_lo, alt_agl[ok].min())
        e5_agl_hi = max(e5_agl_hi, alt_agl[ok].max())
    if not e5:
        continue
    e5_t = np.array([x[0] for x in e5])

    # --- VAD ---
    d = nc.Dataset(vf)
    v_t   = vad_epoch(d)
    v_h   = np.asarray(to_nan(d["height"], VAD_FILL), dtype=float).ravel()  # AGL
    v_sp  = to_nan(d["wind_speed"], VAD_FILL)
    v_snr = vget(d, "mean_snr",    VAD_FILL, v_sp)
    v_res = vget(d, "residual",    VAD_FILL, v_sp)
    v_cor = vget(d, "correlation", VAD_FILL, v_sp)
    v_w   = vget(d, "w",           VAD_FILL, v_sp)
    v_unp = vget(d, "u_npoints",   VAD_FILL, v_sp)
    v_vnp = vget(d, "v_npoints",   VAD_FILL, v_sp)
    v_wnp = vget(d, "w_npoints",   VAD_FILL, v_sp)
    d.close()
    n_days += 1

    for vi in range(len(v_t)):
        dt = np.abs(e5_t - v_t[vi])
        ei = int(np.argmin(dt))
        if dt[ei] > ERA5_TOL_S:
            continue
        _, e_alt, e_spd = e5[ei]
        lo, hi = e_alt.min(), e_alt.max()
        sel = np.isfinite(v_sp[vi]) & (v_h >= lo) & (v_h <= hi)
        if sel.sum() == 0:
            continue
        tgt = v_h[sel]
        e_at = np.interp(tgt, e_alt, e_spd)
        E.extend(e_at);            S_v.extend(v_sp[vi][sel]); H.extend(tgt)
        SNR.extend(v_snr[vi][sel]); RES.extend(v_res[vi][sel]); COR.extend(v_cor[vi][sel])
        W.extend(v_w[vi][sel])
        UNP.extend(v_unp[vi][sel]); VNP.extend(v_vnp[vi][sel]); WNP.extend(v_wnp[vi][sel])

E, S_v, H = np.array(E), np.array(S_v), np.array(H)
SNR, RES, COR, W = map(np.array, (SNR, RES, COR, W))
UNP, VNP, WNP = map(np.array, (UNP, VNP, WNP))
n = len(E)
if n == 0:
    raise SystemExit("No matched points — check date overlap / SITE_ALT_M / ERA5_TOL_S.")

print(f"[datum]  ERA5 AGL coverage (after -{SITE_ALT_M:.0f} m): {e5_agl_lo:.0f} .. {e5_agl_hi:.0f} m")
print(f"[datum]  VAD AGL range: {np.nanmin(H):.0f} .. {np.nanmax(H):.0f} m  (must sit inside ERA5 range)")
print(f"[match]  days with both: {n_days}   matched points: {n}")

# ----------------------------------------------------------------------------- good flag
evalm = S_v >= SPD_FLOOR
rel = np.abs(E - S_v) / np.where(S_v > 0, S_v, np.nan)
good = evalm & (rel <= GOOD_REL)
print(f"[in-band] {100*good[evalm].mean():.1f}% within +-{GOOD_REL*100:.0f}% "
      f"of {evalm.sum()} points (VAD speed >= {SPD_FLOOR} m/s)")
print(f"[speed]  bias (ERA5-VAD): {np.mean(E-S_v):+.2f}  RMSE {np.sqrt(np.mean((E-S_v)**2)):.2f}  "
      f"r {np.corrcoef(S_v, E)[0,1]:.3f}  m/s")

# ----------------------------------------------------------------------------- plot 1
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(S_v, E, c=H, s=6, alpha=0.4, cmap="viridis")
mx = np.nanmax([S_v.max(), E.max()]) * 1.05
ax.plot([0, mx], [0, mx], "k--", lw=1)
ax.fill_between([0, mx], [0, mx*(1-GOOD_REL)], [0, mx*(1+GOOD_REL)], color="seagreen", alpha=0.12)
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel("VAD lidar speed (m/s)"); ax.set_ylabel("ERA5 speed (m/s)")
ax.set_title(f"ERA5 vs VAD speed (n={n})\nshaded = +-{GOOD_REL*100:.0f}% band")
fig.colorbar(sc, ax=ax, label="height AGL (m)")
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- plot 2
params = [
    (H,          "Height AGL (m)"),
    (SNR,        "VAD mean SNR"),
    (RES,        "VAD fit residual (m/s)"),
    (COR,        "VAD correlation"),
    (np.abs(W),  "VAD |vertical wind| (m/s)"),
    (UNP,        "u consensus points"),
    (VNP,        "v consensus points"),
    (WNP,        "w consensus points"),
]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for axp, (arr, lab) in zip(axes.ravel(), params):
    ok = np.isfinite(arr) & evalm
    a, g = arr[ok], good[ok]
    if a.size == 0:
        axp.set_title(f"{lab}\n(no data)"); axp.set_axis_off(); continue
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if lo == hi:
        lo, hi = a.min(), a.max() + 1e-9
    edges = np.linspace(lo, hi, 21)
    axp.hist(a, bins=edges, color="lightgray", label="evaluated")
    axp.hist(a[g], bins=edges, color="seagreen", alpha=0.85, label=f"within +-{GOOD_REL*100:.0f}%")
    axp.set_xlabel(lab); axp.set_ylabel("count")
    cen = 0.5 * (edges[:-1] + edges[1:])
    tot, _ = np.histogram(a, bins=edges)
    inb, _ = np.histogram(a[g], bins=edges)
    frac = np.divide(inb, tot, out=np.zeros(len(tot)), where=tot > 0)
    axt = axp.twinx()
    axt.plot(cen, 100 * frac, "k.-", ms=3, lw=0.8)
    axt.set_ylabel("in-band %", fontsize=8); axt.set_ylim(0, 100)
axes.ravel()[0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"ERA5-VAD agreement: speed within +-{GOOD_REL*100:.0f}%  "
             f"({100*good[evalm].mean():.1f}% of {evalm.sum()} pts)", y=1.0)
plt.tight_layout(); plt.show()

print("\ndone")

