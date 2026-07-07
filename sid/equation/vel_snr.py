#!/usr/bin/env python
"""
Wide screening matrix for the MAPR-lidar VECTOR disagreement, M2HATS ISS1.

dV = |V_MAPR - V_lidar| (vector, m/s) at each matched gate  <-- the target.

Correlates dV against a broad candidate set so nothing obvious is missed:
  MAPR quality : snrw, specWid, u/v/w_dispersion, cons_npoints, abs_wvert
  lidar quality: mean_snr, residual, correlation, npoints, abs_w   (interp to MAPR gates)
  physics/derived: height, wind-speed regime, vertical shear, hour_local

SCREENING ONLY. This is NOT the equation. Use it to:
  - EXCLUDE variables whose dV-row correlation is ~0
  - spot correlated CLUSTERS (red/blue blocks) and keep only ONE per cluster
Then confirm survivors with a plot + held-out validation. A wide matrix is free;
a wide equation overfits -- keep the equation lean.

MAPR winds.30 vs lidar vad_consensus. Shows plots; saves nothing.
Needs numpy, pandas, matplotlib, netCDF4, scipy.
"""

import glob, os, re, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
WINDS = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 900
LOCAL_OFFSET = -7
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
    return np.asarray(v["time"][:], float)
def mg(ds, n): return to_nan(ds[n], MAPR_FILL) if n in ds.variables else None
def lg(ds, names):
    for n in (names if isinstance(names, list) else [names]):
        if n in ds.variables: return to_nan(ds[n], VAD_FILL)
    return None

lid = {re.search(r"(\d{8})", os.path.basename(f)).group(1): f
       for f in glob.glob(os.path.join(LIDAR_DIR, "*.nc"))}

rows = []
for mf in sorted(glob.glob(os.path.join(WINDS, "prof449.*.winds.30.nc"))):
    date = re.search(r"prof449\.(\d{8})\.", os.path.basename(mf)).group(1)
    if date not in lid: continue
    m = nc.Dataset(mf)
    alt = float(np.asarray(m["alt"][...])); m_t = mapr_epoch(m)
    m_h = to_nan(m["height"], MAPR_FILL)
    m_u, m_v = mg(m, "u"), mg(m, "v")
    snrw, sw = mg(m, "snrw"), mg(m, "specWid")
    ud, vd, wd = mg(m, "u_dispersion"), mg(m, "v_dispersion"), mg(m, "w_dispersion")
    cn, wv = mg(m, "cons_npoints"), mg(m, "wvert")
    m.close()
    m_h_agl = m_h - alt if np.nanmin(m_h) > alt - 200 else m_h

    d = nc.Dataset(lid[date])
    l_t = lidar_epoch(d); l_h = np.asarray(lg(d, "height"), float)
    l_u, l_v, l_w = lg(d, "u"), lg(d, "v"), lg(d, "w")
    l_snr, l_res, l_cor = lg(d, "mean_snr"), lg(d, "residual"), lg(d, "correlation")
    l_np = lg(d, ["w_npoints", "u_npoints", "npoints"])
    d.close()
    l2d = (l_h.ndim == 2)

    for i in range(m_u.shape[0]):
        j = int(np.argmin(np.abs(l_t - m_t[i])))
        if abs(l_t[j] - m_t[i]) > TIME_TOL_S: continue
        lh = (l_h[j] if l2d else l_h)
        wok = np.isfinite(lh) & np.isfinite(l_u[j]) & np.isfinite(l_v[j])
        if wok.sum() < 2: continue
        o = np.argsort(lh[wok]); sh = lh[wok][o]
        def ip(arr, tgt): return np.interp(tgt, sh, arr[j][wok][o])

        hcol = m_h_agl[i]
        ok = (np.isfinite(m_u[i]) & np.isfinite(m_v[i]) & np.isfinite(hcol)
              & (hcol >= sh.min()) & (hcol <= sh.max()))
        if not ok.any(): continue
        tgt = hcol[ok]
        ru, rv = ip(l_u, tgt), ip(l_v, tgt)
        rw = ip(l_w, tgt) if l_w is not None else np.full(len(tgt), np.nan)
        # vertical shear of MAPR wind at these gates
        du_dz, dv_dz = np.gradient(m_u[i], hcol), np.gradient(m_v[i], hcol)
        shear = np.hypot(du_dz, dv_dz)[ok]
        hr = (datetime.fromtimestamp(m_t[i], tz=timezone.utc).hour + LOCAL_OFFSET) % 24

        rec = pd.DataFrame({
            "dV": np.hypot(m_u[i][ok] - ru, m_v[i][ok] - rv),
            "height": tgt,
            "wind_speed": 0.5 * (np.hypot(m_u[i][ok], m_v[i][ok]) + np.hypot(ru, rv)),
            "shear": shear,
            "hour_local": hr,
            "snrw": snrw[i][ok] if snrw is not None else np.nan,
            "specWid": sw[i][ok] if sw is not None else np.nan,
            "u_dispersion": ud[i][ok] if ud is not None else np.nan,
            "v_dispersion": vd[i][ok] if vd is not None else np.nan,
            "w_dispersion": wd[i][ok] if wd is not None else np.nan,
            "cons_npoints": cn[i][ok] if cn is not None else np.nan,
            "abs_wvert": np.abs(wv[i][ok]) if wv is not None else np.nan,
            "lid_mean_snr": ip(l_snr, tgt) if l_snr is not None else np.nan,
            "lid_residual": ip(l_res, tgt) if l_res is not None else np.nan,
            "lid_correlation": ip(l_cor, tgt) if l_cor is not None else np.nan,
            "lid_npoints": ip(l_np, tgt) if l_np is not None else np.nan,
            "lid_abs_w": np.abs(rw),
        })
        rows.append(rec)

df = pd.concat(rows, ignore_index=True)
cols = [c for c in df.columns if df[c].notna().any()]
df = df[cols]
print(f"matched gates: {len(df)}   candidates: {len(cols)-1}\n")

sp = df.corr(method="spearman")
dv_row = sp["dV"].drop("dV").reindex(sp["dV"].drop("dV").abs().sort_values(ascending=False).index)
print("Spearman vs dV (sorted by |r|) -- screen with this:")
for k, val in dv_row.items():
    print(f"   {k:16s} {val:+.3f}")
print("\n(near 0 -> drop; strong -> candidate, but prune correlated twins below;")
print(" hour_local is CYCLIC -> Spearman unreliable, use diurnal stratification instead)\n")

# heatmap, dV first
order_cols = ["dV"] + list(dv_row.index)
M = sp.loc[order_cols, order_cols]
k = len(order_cols)
fig, ax = plt.subplots(figsize=(1.05*k + 2, 0.95*k + 1))
im = ax.imshow(M.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(k)); ax.set_xticklabels(order_cols, rotation=45, ha="right")
ax.set_yticks(range(k)); ax.set_yticklabels(order_cols)
for a in range(k):
    for b in range(k):
        ax.text(b, a, f"{M.values[a,b]:.2f}", ha="center", va="center", fontsize=7)
ax.axhline(0.5, color="k", lw=1.5); ax.axvline(0.5, color="k", lw=1.5)  # set off the dV row/col
fig.colorbar(im, label="Spearman r")
ax.set_title("MAPR-lidar |ΔV| vs candidates (top row/col = target)")
plt.tight_layout(); plt.show()
print("done")

