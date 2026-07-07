#!/usr/bin/env python
"""
Forward selection for the |dV| equation, judged on HELD-OUT DAYS.

Terms added in screening order:
  1 specWid   2 v_dispersion   3 u_dispersion   4 w_dispersion
  5 wind_speed   6 cons_npoints

At each stage a linear model dV = b + sum(a_i * x_i) is refit on TRAIN days and
scored (MAE, RMSE) on TEST days. A term earns its place only if test error
drops meaningfully (> ~1-2%); raw R^2 on train data is NOT the judge.
Watch the coefficient table: wild swings in the dispersion coefficients
between stages = collinearity = redundant with specWid.

MAPR winds.30 vs lidar vad_consensus. Shows one summary plot; saves nothing.
Needs numpy, pandas, matplotlib, netCDF4, scikit-learn.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")

WINDS = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 900
ORDER = ["specWid", "v_dispersion", "u_dispersion", "w_dispersion",
         "wind_speed", "cons_npoints"]
TEST_FRAC = 0.2

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
    sw, ud, vd, wd = mg(m, "specWid"), mg(m, "u_dispersion"), mg(m, "v_dispersion"), mg(m, "w_dispersion")
    cn = mg(m, "cons_npoints")
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
        o = np.argsort(lh[wok]); shh = lh[wok][o]
        su, sv = l_u[j][wok][o], l_v[j][wok][o]
        hcol = m_h_agl[i]
        ok = (np.isfinite(m_u[i]) & np.isfinite(m_v[i]) & np.isfinite(hcol)
              & (hcol >= shh.min()) & (hcol <= shh.max()))
        if not ok.any(): continue
        ru, rv = np.interp(hcol[ok], shh, su), np.interp(hcol[ok], shh, sv)
        rec = pd.DataFrame({
            "day": date,
            "dV": np.hypot(m_u[i][ok] - ru, m_v[i][ok] - rv),
            "specWid": sw[i][ok] if sw is not None else np.nan,
            "u_dispersion": ud[i][ok] if ud is not None else np.nan,
            "v_dispersion": vd[i][ok] if vd is not None else np.nan,
            "w_dispersion": wd[i][ok] if wd is not None else np.nan,
            "cons_npoints": cn[i][ok] if cn is not None else np.nan,
            "wind_speed": 0.5*(np.hypot(m_u[i][ok], m_v[i][ok]) + np.hypot(ru, rv)),
        })
        rows.append(rec)

df = pd.concat(rows, ignore_index=True).dropna(subset=["dV"] + ORDER).reset_index(drop=True)
print(f"gates: {len(df)}   days: {df['day'].nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
tr, te = next(gss.split(df, groups=df["day"]))
train, test = df.iloc[tr], df.iloc[te]
print(f"train {len(train)} gates ({train['day'].nunique()}d) / "
      f"test {len(test)} gates ({test['day'].nunique()}d)\n")

y_tr, y_te = train["dV"].values, test["dV"].values

def mae(a, b): return np.mean(np.abs(a - b))
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))

results = []
# stage 0: constant
pred = np.full_like(y_te, y_tr.mean())
results.append(("(constant)", mae(y_te, pred), rmse(y_te, pred), {}, y_tr.mean()))
print(f"stage 0  (constant only)          test MAE {results[-1][1]:.3f}  RMSE {results[-1][2]:.3f}")

for k in range(1, len(ORDER) + 1):
    feats = ORDER[:k]
    reg = LinearRegression().fit(train[feats], y_tr)
    pred = reg.predict(test[feats])
    coefs = dict(zip(feats, reg.coef_))
    m_, r_ = mae(y_te, pred), rmse(y_te, pred)
    prev = results[-1][1]
    gain = 100 * (prev - m_) / prev
    results.append((f"+{feats[-1]}", m_, r_, coefs, reg.intercept_))
    cstr = "  ".join(f"{f}={c:+.2f}" for f, c in coefs.items())
    print(f"stage {k}  {('+'+feats[-1]):18s} test MAE {m_:.3f}  RMSE {r_:.3f}  "
          f"(ΔMAE {gain:+.1f}%)   [{cstr}  b={reg.intercept_:+.2f}]")

print("\nRead: keep a stage only if its ΔMAE is a meaningful drop (>~1-2%).")
print("Unstable dispersion coefficients across stages = collinear with specWid.")

# summary plot
labels = [r[0] for r in results]
maes = [r[1] for r in results]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(len(maes)), maes, "o-", color="steelblue")
for i, (l, m_) in enumerate(zip(labels, maes)):
    ax.annotate(l, (i, m_), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, rotation=20)
ax.set_xlabel("terms added (cumulative)"); ax.set_ylabel("held-out test MAE (m/s)")
ax.set_title("Forward selection: where the curve flattens, the equation stops")
plt.tight_layout(); plt.show()
print("done")

