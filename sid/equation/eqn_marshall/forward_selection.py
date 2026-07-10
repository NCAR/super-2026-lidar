#!/usr/bin/env python
"""
LOTOS Marshall (clean matcher): forward selection for the |dV| equation.

CLEAN matcher: stored wspd/wind_speed + wdir/wind_direction, nearest gate
<= 25 m, NO interpolation. |dV| from stored speed+direction components.

Stages candidates in screening order:
  specWid -> wind_speed -> w_dispersion -> u_dispersion -> v_dispersion -> cons_npoints
Refits all coefficients each stage on TRAIN days; scores held-out TEST days
(split BY DAY). A term stays only if test MAE drops > KEEP_BAR (1%).
Prints per-stage MAE, %-gain, coefficients (watch dispersion coeffs thrash =
collinear with specWid), and the held-out R^2 ceiling.

Needs numpy, pandas, matplotlib, netCDF4, scikit-learn.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

MAPR_DIR  = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr31"
SITE_ALT_FALLBACK = 1742.0
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 300
HEIGHT_TOL = 25.0
TEST_FRAC = 0.2
KEEP_BAR = 0.01
ORDER = ["specWid", "wind_speed", "w_dispersion", "u_dispersion", "v_dispersion", "cons_npoints"]

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
    ud, vd, wd = mg(m, "u_dispersion"), mg(m, "v_dispersion"), mg(m, "w_dispersion")
    cn = mg(m, "cons_npoints")
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
            if not (np.isfinite(hm) and np.isfinite(sm) and np.isfinite(dm)): continue
            k = int(np.argmin(np.abs(vh - hm)))
            if abs(vh[k] - hm) > HEIGHT_TOL: continue
            mu, mvv = uv(sm, dm); lu, lvv = uv(vsp[k], vdi[k])
            rows.append(dict(day=date, dV=np.hypot(mu - lu, mvv - lvv),
                             specWid=sw[i][g] if sw is not None else np.nan,
                             wind_speed=0.5*(sm + vsp[k]),
                             w_dispersion=wd[i][g] if wd is not None else np.nan,
                             u_dispersion=ud[i][g] if ud is not None else np.nan,
                             v_dispersion=vd[i][g] if vd is not None else np.nan,
                             cons_npoints=cn[i][g] if cn is not None else np.nan))

df = pd.DataFrame(rows).dropna(subset=["dV"] + ORDER).reset_index(drop=True)
print(f"gates: {len(df)}   days: {df['day'].nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
tr, te = next(gss.split(df, groups=df["day"]))
train, test = df.iloc[tr], df.iloc[te]
mae = lambda a, b: np.mean(np.abs(a - b))
y_te = test["dV"].values

kept, kept_mae = [], mae(y_te, np.full(len(test), train["dV"].mean()))
print(f"\nstage 0 (constant)          test MAE {kept_mae:.3f}")
maes = [kept_mae]; labels = ["const"]
for s in ORDER:
    feats = kept + [s]
    reg = LinearRegression().fit(train[feats], train["dV"])
    pred = reg.predict(test[feats])
    m_ = mae(y_te, pred); gain = (kept_mae - m_) / kept_mae
    r2 = r2_score(y_te, pred)
    verdict = "KEEP" if gain > KEEP_BAR else "drop"
    cstr = "  ".join(f"{f}={c:+.2f}" for f, c in zip(feats, reg.coef_))
    print(f"+{s:14s} test MAE {m_:.3f}  R^2 {r2:+.3f}  ({100*gain:+.1f}%)  -> {verdict}"
          f"\n    [{cstr}  b={reg.intercept_:+.2f}]")
    maes.append(m_); labels.append("+" + s)
    if gain > KEEP_BAR:
        kept, kept_mae = feats, m_

print(f"\nfinal kept terms: {kept}   test MAE {kept_mae:.3f}")
print("(dispersion coeffs swinging across stages = collinear with specWid = redundant)")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(len(maes)), maes, "o-", color="steelblue")
for i, (l, mv) in enumerate(zip(labels, maes)):
    ax.annotate(l, (i, mv), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, rotation=20)
ax.set_xlabel("terms added"); ax.set_ylabel("held-out test MAE (m/s)")
ax.set_title("LOTOS forward selection (clean matcher)")
plt.tight_layout(); plt.show()
print("done")

