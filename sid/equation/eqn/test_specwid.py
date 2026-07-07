#!/usr/bin/env python
"""
Shape of the |dV| (MAPR-lidar vector disagreement) vs specWid relationship.

Look-first, fit-later. Four views:
  1. hexbin density
  2. binned median + IQR (the clean trend)
  3. the SAME medians on linear / semi-log / log-log axes -- whichever is
     straightest names the functional form:
        linear   straight -> dV = a*sW + b
        semilog  straight -> dV = A * exp(k*sW)        (+ maybe floor)
        loglog   straight -> dV = A * sW**k            (power law)
  4. binned medians by height band (confound check)

Prints the binned (specWid, median dV) table so the numbers can be read
directly. MAPR winds.30 vs lidar vad_consensus. Shows plots; saves nothing.
"""

import glob, os, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

WINDS = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 900

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

lid = {re.search(r"(\d{8})", os.path.basename(f)).group(1): f
       for f in glob.glob(os.path.join(LIDAR_DIR, "*.nc"))}

DV, SW, H = [], [], []
for mf in sorted(glob.glob(os.path.join(WINDS, "prof449.*.winds.30.nc"))):
    date = re.search(r"prof449\.(\d{8})\.", os.path.basename(mf)).group(1)
    if date not in lid: continue
    m = nc.Dataset(mf)
    alt = float(np.asarray(m["alt"][...])); m_t = mapr_epoch(m)
    m_h = to_nan(m["height"], MAPR_FILL)
    m_u, m_v = to_nan(m["u"], MAPR_FILL), to_nan(m["v"], MAPR_FILL)
    sw = to_nan(m["specWid"], MAPR_FILL)
    m.close()
    m_h_agl = m_h - alt if np.nanmin(m_h) > alt - 200 else m_h

    d = nc.Dataset(lid[date])
    l_t = lidar_epoch(d); l_h = np.asarray(to_nan(d["height"], VAD_FILL), float)
    l_u, l_v = to_nan(d["u"], VAD_FILL), to_nan(d["v"], VAD_FILL)
    d.close()
    l2d = (l_h.ndim == 2)

    for i in range(m_u.shape[0]):
        j = int(np.argmin(np.abs(l_t - m_t[i])))
        if abs(l_t[j] - m_t[i]) > TIME_TOL_S: continue
        lh = l_h[j] if l2d else l_h
        wok = np.isfinite(lh) & np.isfinite(l_u[j]) & np.isfinite(l_v[j])
        if wok.sum() < 2: continue
        o = np.argsort(lh[wok]); sh = lh[wok][o]
        su, sv = l_u[j][wok][o], l_v[j][wok][o]
        hcol = m_h_agl[i]
        ok = (np.isfinite(m_u[i]) & np.isfinite(m_v[i]) & np.isfinite(sw[i])
              & (hcol >= sh.min()) & (hcol <= sh.max()))
        if not ok.any(): continue
        ru, rv = np.interp(hcol[ok], sh, su), np.interp(hcol[ok], sh, sv)
        DV.extend(np.hypot(m_u[i][ok] - ru, m_v[i][ok] - rv))
        SW.extend(sw[i][ok]); H.extend(hcol[ok])

DV, SW, H = np.array(DV), np.array(SW), np.array(H)
n = len(DV)
rho, _ = spearmanr(SW, DV)
print(f"pairs: {n}   Spearman(specWid, |dV|) = {rho:+.3f}\n")

def binned(x, y, edges):
    cen = 0.5*(edges[:-1]+edges[1:]); med, q1, q3 = [], [], []
    for i in range(len(edges)-1):
        s = (x >= edges[i]) & (x < edges[i+1])
        if s.sum() < 30: med.append(np.nan); q1.append(np.nan); q3.append(np.nan); continue
        med.append(np.median(y[s])); q1.append(np.percentile(y[s],25)); q3.append(np.percentile(y[s],75))
    return cen, np.array(med), np.array(q1), np.array(q3)

edges = np.linspace(np.percentile(SW, 0.5), np.percentile(SW, 99.5), 26)
cen, med, q1, q3 = binned(SW, DV, edges)

print("binned table (specWid, median |dV|):")
for c, mv in zip(cen, med):
    if np.isfinite(mv): print(f"   {c:6.3f}   {mv:5.2f}")

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
hb = ax[0,0].hexbin(SW, DV, gridsize=45, bins="log", cmap="viridis",
                    extent=(edges[0], edges[-1], 0, np.percentile(DV, 99)))
ax[0,0].set_xlabel("specWid (m/s)"); ax[0,0].set_ylabel("|ΔV| (m/s)")
ax[0,0].set_title(f"Density (n={n}, ρ={rho:+.2f})"); fig.colorbar(hb, ax=ax[0,0], label="log count")

ax[0,1].fill_between(cen, q1, q3, alpha=0.25, color="steelblue", label="IQR")
ax[0,1].plot(cen, med, "o-", color="steelblue", ms=4, label="median")
ax[0,1].set_xlabel("specWid (m/s)"); ax[0,1].set_ylabel("|ΔV| (m/s)")
ax[0,1].set_title("Binned median (the trend to fit)"); ax[0,1].legend()

for lo, hi in [(0,500),(500,1000),(1000,2000)]:
    mm = (H >= lo) & (H < hi)
    if mm.sum() < 200: continue
    c, md, *_ = binned(SW[mm], DV[mm], edges)
    ax[0,2].plot(c, md, "o-", ms=3, label=f"{lo}-{hi} m")
ax[0,2].set_xlabel("specWid (m/s)"); ax[0,2].set_ylabel("median |ΔV| (m/s)")
ax[0,2].set_title("By height band"); ax[0,2].legend()

# form tests: same medians, three axes
ax[1,0].plot(cen, med, "o-", color="k", ms=4)
ax[1,0].set_xlabel("specWid"); ax[1,0].set_ylabel("median |ΔV|")
ax[1,0].set_title("LINEAR axes: straight -> dV = a*sW + b")

ax[1,1].semilogy(cen, med, "o-", color="crimson", ms=4)
ax[1,1].set_xlabel("specWid"); ax[1,1].set_ylabel("median |ΔV| (log)")
ax[1,1].set_title("SEMI-LOG: straight -> exponential")

pos = np.isfinite(med) & (cen > 0) & (med > 0)
ax[1,2].loglog(cen[pos], med[pos], "o-", color="darkorange", ms=4)
ax[1,2].set_xlabel("specWid (log)"); ax[1,2].set_ylabel("median |ΔV| (log)")
ax[1,2].set_title("LOG-LOG: straight -> power law dV = A*sW^k")

plt.tight_layout(); plt.show()
print("done")

