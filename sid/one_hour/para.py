#!/usr/bin/env python
"""
Three-way wind comparison on a common reference, M2HATS ISS1.

VAD Windcube lidar is the shared truth. Both the MAPR 449 MHz radar profiler
and ERA5 reanalysis are matched to the SAME VAD times and interpolated to the
SAME VAD heights, so MAPR and ERA5 are evaluated on identical points and can be
compared head-to-head as well as each against the lidar.

Everything is on height-AGL (MAPR datum auto-detected; ERA5 geopotential
z/9.80665 is MSL, converted with the site altitude read from the MAPR file).

Prints a data-driven findings block, then shows one 2x3 summary figure.
Run on mercury (needs a display): python compare_three_way.py
"""

import matplotlib.pyplot as plt            # ssh -X / Jupyter / VS Code remote
import netCDF4 as nc
import numpy as np
import glob, os, re, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
MAPR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
ERA5_DIR = "/scr/isf_apg/models/m2hats/era5"
VAD_DIR  = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"

MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 900         # VAD<->MAPR match window (s)
ERA5_TOL_S = 1800        # VAD<->ERA5 match window (s); ERA5 hourly
GOOD_REL   = 0.20        # within +-20% of VAD speed = "in-band"
SPD_FLOOR  = 2.0         # m/s : skip relative band below this
LOCAL_OFFSET = -7        # ISS1 local time = UTC-7 (PDT, summer)
G = 9.80665
# -----------------------------------------------------------------------------

def to_nan(var, fill):
    a = var[:]
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    a = np.asarray(a, dtype=float); a[a == fill] = np.nan
    return a

def vad_epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    t = np.asarray(v["time"][:], float)
    if np.nanmax(t) > 1e8: return t
    if "base_time" in v: return float(np.asarray(v["base_time"][...])) + t
    return t

def vget(d, name, fill, like):
    return to_nan(d[name], fill) if name in d.variables else np.full_like(like, np.nan)

def met_dir(u, v):
    return (np.degrees(np.arctan2(-u, -v))) % 360.0

def prep(h, u, v):
    """Return (h,u,v) finite & sorted ascending in height, or None."""
    ok = np.isfinite(h) & np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 2: return None
    o = np.argsort(h[ok])
    return h[ok][o], u[ok][o], v[ok][o]

# ----------------------------------------------------------------------------- gather
vad_files = sorted(glob.glob(os.path.join(VAD_DIR, "30min_winds_*.nc")))
if not vad_files:
    raise SystemExit("No VAD files — check VAD_DIR")

cols = {k: [] for k in ["h", "hr", "vs", "vd", "ms", "md", "es", "ed",
                         "snr", "res", "cor", "w", "unp", "vnp", "wnp"]}
n_days = 0

for vf in vad_files:
    date = os.path.basename(vf).replace("30min_winds_", "").replace(".nc", "")
    mf = os.path.join(MAPR_DIR, f"prof449.{date}.winds.30.nc")
    efiles = sorted(glob.glob(os.path.join(ERA5_DIR, date, f"era5_pressure_{date}_*_ISS1.nc")))
    if not os.path.exists(mf) or not efiles:
        continue

    # MAPR
    m = nc.Dataset(mf)
    site_alt = float(np.asarray(m["alt"][...]))
    m_t = float(np.asarray(m["base_time"][...])) + np.asarray(m["time"][:], float)
    m_h = to_nan(m["height"], MAPR_FILL)
    m_u, m_v = to_nan(m["u"], MAPR_FILL), to_nan(m["v"], MAPR_FILL)
    m.close()
    m_h_agl = m_h - site_alt if np.nanmin(m_h) > site_alt - 200 else m_h

    # ERA5: per-hour (t, h_agl, u, v) sorted
    e5 = []
    for ef in efiles:
        ds = nc.Dataset(ef)
        u = np.asarray(ds["u"][0, :, 0, 0], float)
        v = np.asarray(ds["v"][0, :, 0, 0], float)
        z = np.asarray(ds["z"][0, :, 0, 0], float)
        t = float(np.asarray(ds["valid_time"][0]))
        ds.close()
        p = prep(z / G - site_alt, u, v)
        if p: e5.append((t, *p))
    if not e5:
        continue
    e5_t = np.array([x[0] for x in e5])

    # VAD
    d = nc.Dataset(vf)
    v_t = vad_epoch(d)
    v_h = np.asarray(to_nan(d["height"], VAD_FILL), float).ravel()
    v_sp, v_di = to_nan(d["wind_speed"], VAD_FILL), to_nan(d["wind_direction"], VAD_FILL)
    q_snr = vget(d, "mean_snr", VAD_FILL, v_sp);   q_res = vget(d, "residual", VAD_FILL, v_sp)
    q_cor = vget(d, "correlation", VAD_FILL, v_sp); q_w = vget(d, "w", VAD_FILL, v_sp)
    q_unp = vget(d, "u_npoints", VAD_FILL, v_sp);  q_vnp = vget(d, "v_npoints", VAD_FILL, v_sp)
    q_wnp = vget(d, "w_npoints", VAD_FILL, v_sp)
    d.close()
    n_days += 1

    for vi in range(len(v_t)):
        mi = int(np.argmin(np.abs(m_t - v_t[vi])))
        ei = int(np.argmin(np.abs(e5_t - v_t[vi])))
        if abs(m_t[mi] - v_t[vi]) > TIME_TOL_S or abs(e5_t[ei] - v_t[vi]) > ERA5_TOL_S:
            continue
        pm = prep(m_h_agl[mi], m_u[mi], m_v[mi])
        if pm is None:
            continue
        mh, mu, mv = pm
        _, eh, eu, ev = e5[ei]

        lo = max(mh.min(), eh.min())
        hi = min(mh.max(), eh.max())
        sel = np.isfinite(v_sp[vi]) & np.isfinite(v_di[vi]) & (v_h >= lo) & (v_h <= hi)
        if sel.sum() == 0:
            continue
        tgt = v_h[sel]
        mui, mvi = np.interp(tgt, mh, mu), np.interp(tgt, mh, mv)
        eui, evi = np.interp(tgt, eh, eu), np.interp(tgt, eh, ev)
        hr_local = (datetime.fromtimestamp(v_t[vi], tz=timezone.utc).hour + LOCAL_OFFSET) % 24

        cols["h"].extend(tgt);             cols["hr"].extend([hr_local] * sel.sum())
        cols["vs"].extend(v_sp[vi][sel]);  cols["vd"].extend(v_di[vi][sel])
        cols["ms"].extend(np.hypot(mui, mvi)); cols["md"].extend(met_dir(mui, mvi))
        cols["es"].extend(np.hypot(eui, evi)); cols["ed"].extend(met_dir(eui, evi))
        cols["snr"].extend(q_snr[vi][sel]); cols["res"].extend(q_res[vi][sel])
        cols["cor"].extend(q_cor[vi][sel]); cols["w"].extend(q_w[vi][sel])
        cols["unp"].extend(q_unp[vi][sel]); cols["vnp"].extend(q_vnp[vi][sel])
        cols["wnp"].extend(q_wnp[vi][sel])

C = {k: np.array(v) for k, v in cols.items()}
n = len(C["h"])
if n == 0:
    raise SystemExit("No common 3-way matched points — check overlaps / tolerances.")

# ----------------------------------------------------------------------------- derived
def circ(a, b): return np.abs(((a - b + 180) % 360) - 180)

vs = C["vs"]
m_bias, e_bias = C["ms"] - vs, C["es"] - vs
m_ddiff, e_ddiff = circ(C["md"], C["vd"]), circ(C["ed"], C["vd"])
evalm = vs >= SPD_FLOOR
m_good = evalm & (np.abs(C["ms"] - vs) / np.where(vs > 0, vs, np.nan) <= GOOD_REL)
e_good = evalm & (np.abs(C["es"] - vs) / np.where(vs > 0, vs, np.nan) <= GOOD_REL)

def rmse(x): x = x[np.isfinite(x)]; return np.sqrt(np.mean(x ** 2)) if x.size else np.nan
def rate(mask, sub): s = sub & np.isfinite(mask.astype(float)); return 100 * mask[sub].mean() if sub.sum() else np.nan

# ----------------------------------------------------------------------------- findings
L = []
L.append(f"common 3-way matched points: {n}   (days with all three: {n_days})")
L.append("")
L.append(f"{'':14s}{'bias':>8s}{'RMSE':>8s}{'r':>8s}{'in±20%':>9s}{'dirMAD':>9s}")
for tag, b, dd, good, sp in [("MAPR radar", m_bias, m_ddiff, m_good, C["ms"]),
                             ("ERA5", e_bias, e_ddiff, e_good, C["es"])]:
    r = np.corrcoef(vs[np.isfinite(sp)], sp[np.isfinite(sp)])[0, 1]
    L.append(f"{tag:14s}{np.nanmean(b):+8.2f}{rmse(b):8.2f}{r:8.3f}"
             f"{100*good[evalm].mean():8.1f}%{np.nanmedian(dd):8.1f}°")
better = "MAPR radar" if rmse(m_bias) < rmse(e_bias) else "ERA5"
L.append(f"-> closer to the lidar overall (lower speed RMSE): {better}")
L.append("")

# RMSE vs height -> find the crossover
L.append("speed RMSE by height band (winner = lower):")
bands = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 1e9)]
prev_winner = None
for lo, hi in bands:
    s = (C["h"] >= lo) & (C["h"] < hi)
    if s.sum() < 20:
        continue
    rm, re_ = rmse(m_bias[s]), rmse(e_bias[s])
    w = "MAPR" if rm < re_ else "ERA5"
    flip = "   <-- crossover" if prev_winner and w != prev_winner else ""
    hi_s = "+" if hi > 1e8 else f"{int(hi)}"
    L.append(f"  {int(lo):>4}-{hi_s:<5} m  n={s.sum():>5}  MAPR {rm:4.2f}  ERA5 {re_:4.2f}  -> {w}{flip}")
    prev_winner = w
L.append("")

# diurnal
day = (C["hr"] >= 10) & (C["hr"] < 18)
night = (C["hr"] >= 22) | (C["hr"] < 6)
L.append("in-band % by time of day (local):")
for lab, sub in [("day 10-18", day & evalm), ("night 22-06", night & evalm)]:
    L.append(f"  {lab:11s} n={sub.sum():>5}  MAPR {100*m_good[sub].mean():4.1f}%  ERA5 {100*e_good[sub].mean():4.1f}%")
L.append("")

# convection: |w| quartiles
aw = np.abs(C["w"])
finw = np.isfinite(aw) & evalm
if finw.sum() > 40:
    q1, q3 = np.percentile(aw[finw], [25, 75])
    calm = finw & (aw <= q1)
    conv = finw & (aw >= q3)
    L.append(f"convection test (VAD |vertical wind| quartiles, calm<= {q1:.2f}, active>= {q3:.2f} m/s):")
    L.append(f"  calm   n={calm.sum():>5}  MAPR {100*m_good[calm].mean():4.1f}%  ERA5 {100*e_good[calm].mean():4.1f}%")
    L.append(f"  active n={conv.sum():>5}  MAPR {100*m_good[conv].mean():4.1f}%  ERA5 {100*e_good[conv].mean():4.1f}%")
    L.append(f"  -> drop calm->active:  MAPR {100*(m_good[calm].mean()-m_good[conv].mean()):+.1f} pts, "
             f"ERA5 {100*(e_good[calm].mean()-e_good[conv].mean()):+.1f} pts")
    L.append("")

# radar vs reanalysis head-to-head
me = C["ms"] - C["es"]
L.append(f"MAPR vs ERA5 directly: RMSE {rmse(me):.2f} m/s, r {np.corrcoef(C['ms'],C['es'])[0,1]:.3f}")

print("\n" + "=" * 64)
print("FINDINGS  (MAPR radar / ERA5, both vs VAD lidar, common points)")
print("=" * 64)
print("\n".join(L))
print("=" * 64 + "\n")

# ----------------------------------------------------------------------------- figure
def binline(x, mask_good, edges):
    cen = 0.5 * (edges[:-1] + edges[1:]); frac = []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1]) & evalm
        frac.append(100 * mask_good[s].mean() if s.sum() >= 10 else np.nan)
    return cen, np.array(frac)

def binrmse(x, bias, edges):
    cen = 0.5 * (edges[:-1] + edges[1:]); out = []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        out.append(rmse(bias[s]) if s.sum() >= 20 else np.nan)
    return cen, np.array(out)

hedge = np.arange(0, np.nanmax(C["h"]) + 200, 200.0)
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# 1. scatter
mx = np.nanmax([vs.max(), C["ms"].max(), C["es"].max()]) * 1.05
ax[0,0].scatter(vs, C["ms"], s=5, alpha=0.25, color="steelblue", label="MAPR")
ax[0,0].scatter(vs, C["es"], s=5, alpha=0.25, color="darkorange", label="ERA5")
ax[0,0].plot([0,mx],[0,mx],"k--",lw=1)
ax[0,0].plot([0,mx],[0,mx*(1+GOOD_REL)],"g:",lw=0.8); ax[0,0].plot([0,mx],[0,mx*(1-GOOD_REL)],"g:",lw=0.8)
ax[0,0].set_xlim(0,mx); ax[0,0].set_ylim(0,mx)
ax[0,0].set_xlabel("VAD speed (m/s)"); ax[0,0].set_ylabel("model/radar speed (m/s)")
ax[0,0].set_title("Speed vs lidar (±20% dotted)"); ax[0,0].legend()

# 2. RMSE vs height  (the crossover)
cm, rm = binrmse(C["h"], m_bias, hedge); ce, re_ = binrmse(C["h"], e_bias, hedge)
ax[0,1].plot(rm, cm, "o-", color="steelblue", ms=4, label="MAPR")
ax[0,1].plot(re_, ce, "o-", color="darkorange", ms=4, label="ERA5")
ax[0,1].set_xlabel("speed RMSE vs VAD (m/s)"); ax[0,1].set_ylabel("height AGL (m)")
ax[0,1].set_title("Error vs height (look for crossover)"); ax[0,1].legend()

# 3. in-band vs local hour
hh = np.arange(0, 25, 2.0)
chm, fhm = binline(C["hr"], m_good, hh); che, fhe = binline(C["hr"], e_good, hh)
ax[0,2].plot(chm, fhm, "o-", color="steelblue", ms=4, label="MAPR")
ax[0,2].plot(che, fhe, "o-", color="darkorange", ms=4, label="ERA5")
ax[0,2].set_xlabel("local hour (UTC-7)"); ax[0,2].set_ylabel("in-band %")
ax[0,2].set_ylim(0,100); ax[0,2].set_title("Agreement diurnal cycle"); ax[0,2].legend()

# 4. in-band vs |vertical wind|
we = np.linspace(0, np.nanpercentile(np.abs(C["w"]), 95), 11)
cwm, fwm = binline(np.abs(C["w"]), m_good, we); cwe, fwe = binline(np.abs(C["w"]), e_good, we)
ax[1,0].plot(cwm, fwm, "o-", color="steelblue", ms=4, label="MAPR")
ax[1,0].plot(cwe, fwe, "o-", color="darkorange", ms=4, label="ERA5")
ax[1,0].set_xlabel("VAD |vertical wind| (m/s)"); ax[1,0].set_ylabel("in-band %")
ax[1,0].set_ylim(0,100); ax[1,0].set_title("Agreement vs convection"); ax[1,0].legend()

# 5. direction MAD vs height
def binmad(x, dd, edges):
    cen = 0.5*(edges[:-1]+edges[1:]); out=[]
    for i in range(len(edges)-1):
        s=(x>=edges[i])&(x<edges[i+1])
        out.append(np.nanmedian(dd[s]) if s.sum()>=20 else np.nan)
    return cen, np.array(out)
cdm, dm = binmad(C["h"], m_ddiff, hedge); cde, de = binmad(C["h"], e_ddiff, hedge)
ax[1,1].plot(dm, cdm, "o-", color="steelblue", ms=4, label="MAPR")
ax[1,1].plot(de, cde, "o-", color="darkorange", ms=4, label="ERA5")
ax[1,1].set_xlabel("direction |diff| median (deg)"); ax[1,1].set_ylabel("height AGL (m)")
ax[1,1].set_title("Direction error vs height"); ax[1,1].legend()

# 6. sampling
cnt = [((C["h"]>=hedge[i])&(C["h"]<hedge[i+1])).sum() for i in range(len(hedge)-1)]
ax[1,2].plot(cnt, 0.5*(hedge[:-1]+hedge[1:]), "k.-", ms=4)
ax[1,2].set_xlabel("matched points"); ax[1,2].set_ylabel("height AGL (m)")
ax[1,2].set_title("Sampling per level")

fig.suptitle(f"MAPR radar vs ERA5, both vs VAD lidar — {n} common points, {n_days} days", y=1.0)
plt.tight_layout(); plt.show()
print("done")

