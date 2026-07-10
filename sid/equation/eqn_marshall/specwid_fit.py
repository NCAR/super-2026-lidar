#!/usr/bin/env python
"""
LOTOS Marshall (refined/clean matcher): specWid vs |dV| shape + fit.

CLEAN matcher only: stored wspd/wind_speed + wdir/wind_direction, nearest gate
<= 25 m, NO interpolation (same as the corrected scatter). |dV| from stored
speed+direction components.

Does two things:
  (1) SHAPE: binned median |dV| vs specWid on linear/semilog/loglog + height bands
  (2) FIT:  dV = a*specWid + b, OLS + median regression, day-bootstrap 95% CIs,
            held-out (by-day) MAE vs constant baseline

Prints M2HATS coefficients for the cross-campaign comparison. Expect a WEAKER
relation here (screening rho ~0.22 vs 0.45) -- report r^2 honestly.
Shows plots; saves nothing. Needs numpy, pandas, matplotlib, netCDF4, scikit-learn, statsmodels.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

MAPR_DIR  = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr31"
SITE_ALT_FALLBACK = 1742.0
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 300
HEIGHT_TOL = 25.0
TEST_FRAC = 0.2
N_BOOT = 300

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
def mg(ds, n): return to_nan(ds[n], MAPR_FILL) if n in ds.variables else None
def lg(ds, names):
    for n in (names if isinstance(names, list) else [names]):
        if n in ds.variables: return to_nan(ds[n], VAD_FILL)
    return None
def uv(spd, di):
    r = np.radians(di); return -spd*np.sin(r), -spd*np.cos(r)

lid = {re.search(r"(\d{8})", os.path.basename(f)).group(1): f
       for f in glob.glob(os.path.join(LIDAR_DIR, "VAD_*.nc"))}
rows = []
for mf in sorted(glob.glob(os.path.join(MAPR_DIR, "prof449.*.winds.05.nc"))):
    date = re.search(r"prof449\.(\d{8})\.", os.path.basename(mf)).group(1)
    if date not in lid: continue
    m = nc.Dataset(mf)
    alt = float(np.asarray(m["alt"][...]))
    if not np.isfinite(alt): alt = SITE_ALT_FALLBACK
    m_t = mapr_epoch(m); m_h = to_nan(m["height"], MAPR_FILL)
    m_sp, m_di = mg(m, "wspd"), mg(m, "wdir")
    sw = mg(m, "specWid")
    m.close()
    m_h_agl = m_h - alt if np.nanmin(m_h) > alt - 200 else m_h

    d = nc.Dataset(lid[date])
    l_t = lidar_epoch(d); l_h = np.asarray(lg(d, "height"), float)
    l_sp, l_di = lg(d, ["wind_speed", "wspd"]), lg(d, ["wind_direction", "wdir"])
    d.close()
    if l_t is None or l_sp is None: continue
    l2d = (l_h.ndim == 2)

    for i in range(m_sp.shape[0]):
        j = int(np.argmin(np.abs(l_t - m_t[i])))
        if abs(l_t[j] - m_t[i]) > TIME_TOL_S: continue
        lh = l_h[j] if l2d else l_h
        lsp, ldi = l_sp[j], (l_di[j] if l_di is not None else np.full_like(l_sp[j], np.nan))
        valid = np.isfinite(lh) & np.isfinite(lsp) & np.isfinite(ldi)
        if not valid.any(): continue
        vh, vsp, vdi = lh[valid], lsp[valid], ldi[valid]
        for g in range(len(m_h_agl[i])):
            hm, sm, dm = m_h_agl[i][g], m_sp[i][g], m_di[i][g]
            if not (np.isfinite(hm) and np.isfinite(sm) and np.isfinite(dm) and np.isfinite(sw[i][g])):
                continue
            k = int(np.argmin(np.abs(vh - hm)))
            if abs(vh[k] - hm) > HEIGHT_TOL: continue
            mu, mvv = uv(sm, dm); lu, lvv = uv(vsp[k], vdi[k])
            rows.append(dict(day=date, dV=np.hypot(mu - lu, mvv - lvv),
                             specWid=sw[i][g], height=hm,
                             wind_speed=0.5*(sm + vsp[k])))

df = pd.DataFrame(rows).dropna().reset_index(drop=True)
rho = spearmanr(df["specWid"], df["dV"])[0]
print(f"gates: {len(df)}   days: {df['day'].nunique()}   Spearman(specWid,|dV|)={rho:+.3f}\n")

def binned(x, y, edges):
    cen = 0.5*(edges[:-1]+edges[1:]); med, q1, q3 = [], [], []
    for i in range(len(edges)-1):
        s = (x >= edges[i]) & (x < edges[i+1])
        if s.sum() < 30: med.append(np.nan); q1.append(np.nan); q3.append(np.nan); continue
        med.append(np.median(y[s])); q1.append(np.percentile(y[s],25)); q3.append(np.percentile(y[s],75))
    return cen, np.array(med), np.array(q1), np.array(q3)

edges = np.linspace(df["specWid"].quantile(0.005), df["specWid"].quantile(0.995), 26)
cen, med, q1, q3 = binned(df["specWid"].values, df["dV"].values, edges)
print("binned table (specWid, median |dV|):")
for c, mv in zip(cen, med):
    if np.isfinite(mv): print(f"   {c:6.3f}   {mv:5.2f}")

# fit dV = a*specWid + b  (OLS + median), held out by day
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
tr, te = next(gss.split(df, groups=df["day"]))
train, test = df.iloc[tr], df.iloc[te]
mae = lambda a, b: np.mean(np.abs(a - b))
base = mae(test["dV"], np.full(len(test), train["dV"].mean()))
m1 = LinearRegression().fit(train[["specWid"]], train["dV"])
p1 = m1.predict(test[["specWid"]])
print(f"\nheld-out MAE: constant {base:.3f} -> specWid {mae(test['dV'], p1):.3f} "
      f"({100*(base-mae(test['dV'],p1))/base:+.1f}%)   test R^2 {r2_score(test['dV'], p1):.3f}")

ols = LinearRegression().fit(df[["specWid"]], df["dV"])
qr = smf.quantreg("dV ~ specWid", df).fit(q=0.5)
days = df["day"].unique(); grouped = {d: g for d, g in df.groupby("day")}
rng = np.random.default_rng(0); boots = []
for _ in range(N_BOOT):
    pick = rng.choice(days, size=len(days), replace=True)
    bs = pd.concat([grouped[p] for p in pick], ignore_index=True)
    r = LinearRegression().fit(bs[["specWid"]], bs["dV"])
    boots.append([r.coef_[0], r.intercept_])
boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
print(f"\nLOTOS FIT (clean matcher):")
print(f"  OLS:    dV = {ols.coef_[0]:.3f}*specWid + {ols.intercept_:.2f}"
      f"   [a CI {lo[0]:+.2f},{hi[0]:+.2f}  b CI {lo[1]:+.2f},{hi[1]:+.2f}]")
print(f"  Median: dV = {qr.params['specWid']:.3f}*specWid + {qr.params['Intercept']:.2f}")
print(f"  M2HATS median (2-term ref): dV = 3.58*specWid + 0.073*wind + 0.10")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].fill_between(cen, q1, q3, alpha=0.25, color="steelblue", label="IQR")
ax[0].plot(cen, med, "o-", color="steelblue", ms=4, label="median")
xs = np.linspace(edges[0], edges[-1], 50)
ax[0].plot(xs, ols.coef_[0]*xs + ols.intercept_, "k--", label="OLS fit")
ax[0].set_xlabel("specWid (m/s)"); ax[0].set_ylabel("|ΔV| (m/s)")
ax[0].set_title(f"Binned median + fit (ρ={rho:+.2f})"); ax[0].legend()

ax[1].semilogy(cen, med, "o-", color="crimson", ms=4)
ax[1].set_xlabel("specWid"); ax[1].set_ylabel("median |ΔV| (log)")
ax[1].set_title("semilog: straight->exp")
pos = np.isfinite(med) & (cen > 0) & (med > 0)
ax[2].loglog(cen[pos], med[pos], "o-", color="darkorange", ms=4)
ax[2].set_xlabel("specWid (log)"); ax[2].set_ylabel("median |ΔV| (log)")
ax[2].set_title("loglog: straight->power")
plt.tight_layout(); plt.show()
print("done")

