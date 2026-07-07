#!/usr/bin/env python
"""
FINAL fit of the |dV| equation, now testing two extended candidates against the
2-term base (specWid + wind_speed):

  +specWid^2      curvature term  (the residuals showed ~+-0.3 m/s structure)
  +specWid*wind   interaction     (does broadening hurt more in strong wind?)

All forms fight on the SAME held-out day split; an extended term is adopted only
if it beats the base test MAE by >1%. Whichever form wins gets:
  * OLS coefficients with bootstrap-BY-DAY 95% CIs (flags any CI crossing 0)
  * median (quantile) regression -- the "typical disagreement" QC line
  * residuals vs specWid for leftover structure

Shows one figure; saves nothing.
Needs numpy, pandas, matplotlib, netCDF4, scikit-learn, statsmodels.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

WINDS = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 900
TEST_FRAC = 0.2
N_BOOT = 500

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

rows = []
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
        o = np.argsort(lh[wok]); shh = lh[wok][o]
        su, sv = l_u[j][wok][o], l_v[j][wok][o]
        hcol = m_h_agl[i]
        ok = (np.isfinite(m_u[i]) & np.isfinite(m_v[i]) & np.isfinite(sw[i])
              & (hcol >= shh.min()) & (hcol <= shh.max()))
        if not ok.any(): continue
        ru, rv = np.interp(hcol[ok], shh, su), np.interp(hcol[ok], shh, sv)
        rows.append(pd.DataFrame({
            "day": date,
            "dV": np.hypot(m_u[i][ok] - ru, m_v[i][ok] - rv),
            "specWid": sw[i][ok],
            "wind_speed": 0.5*(np.hypot(m_u[i][ok], m_v[i][ok]) + np.hypot(ru, rv)),
        }))

df = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)
df["specWid2"] = df["specWid"] ** 2                      # curvature candidate
df["sWxWS"] = df["specWid"] * df["wind_speed"]           # interaction candidate
print(f"gates: {len(df)}   days: {df['day'].nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
tr, te = next(gss.split(df, groups=df["day"]))
train, test = df.iloc[tr], df.iloc[te]
mae = lambda a, b: np.mean(np.abs(a - b))

# held-out shootout: the new terms must EARN their place (>~1-2% MAE drop)
MODELS = {
    "constant":        [],
    "specWid-only":    ["specWid"],
    "base (2-term)":   ["specWid", "wind_speed"],
    "+specWid^2":      ["specWid", "wind_speed", "specWid2"],
    "+interaction":    ["specWid", "wind_speed", "sWxWS"],
    "+both":           ["specWid", "wind_speed", "specWid2", "sWxWS"],
}
scores = {}
print("\nheld-out test MAE:")
base_mae = None
for name, feats in MODELS.items():
    if not feats:
        pred = np.full(len(test), train["dV"].mean())
    else:
        reg = LinearRegression().fit(train[feats], train["dV"])
        pred = reg.predict(test[feats])
    scores[name] = mae(test["dV"], pred)
    note = ""
    if name == "base (2-term)":
        base_mae = scores[name]
    elif base_mae is not None:
        note = f"   (vs base {100*(base_mae - scores[name])/base_mae:+.1f}%)"
    print(f"  {name:15s} {scores[name]:.3f}{note}")

# pick the winner among the extended candidates, but only if it beats base by >1%
cands = {k: v for k, v in scores.items() if k in ("+specWid^2", "+interaction", "+both")}
best_name = min(cands, key=cands.get)
if (base_mae - cands[best_name]) / base_mae > 0.01:
    FINAL = MODELS[best_name]
    print(f"\n-> '{best_name}' clears the >1% bar; fitting it as the final form.")
else:
    FINAL = MODELS["base (2-term)"]
    best_name = "base (2-term)"
    print(f"\n-> no extended term clears the >1% bar; the 2-term equation stands.")

# final coefficients on ALL days for the WINNING form
ols = LinearRegression().fit(df[FINAL], df["dV"])
qr = smf.quantreg("dV ~ " + " + ".join(FINAL), df).fit(q=0.5)

# bootstrap BY DAY for the OLS coefficients
days = df["day"].unique()
boots = []
rng = np.random.default_rng(0)
for _ in range(N_BOOT):
    pick = rng.choice(days, size=len(days), replace=True)
    bs = pd.concat([df[df["day"] == p] for p in pick], ignore_index=True)
    r = LinearRegression().fit(bs[FINAL], bs["dV"])
    boots.append(list(r.coef_) + [r.intercept_])
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

def eqstr(coefs, intercept):
    return "dV = " + " + ".join(f"{c:.3f}*{f}" for f, c in zip(FINAL, coefs)) + f" + {intercept:.2f}"

print(f"\nFINAL EQUATION ({best_name}, all days):")
print(f"  OLS (mean):     {eqstr(ols.coef_, ols.intercept_)}")
for k, f in enumerate(FINAL):
    print(f"    {f:12s} 95% CI [{lo[k]:+.3f}, {hi[k]:+.3f}]"
          + ("   <-- CI crosses 0: term not robust" if lo[k] < 0 < hi[k] else ""))
print(f"    intercept    95% CI [{lo[-1]:+.3f}, {hi[-1]:+.3f}]")
print(f"  Median (q=0.5): {eqstr([qr.params[f] for f in FINAL], qr.params['Intercept'])}")
print("  (median line = 'typical disagreement' QC relation; OLS = mean relation)")

# residual check vs specWid (any leftover structure = form still wrong)
res = df["dV"].values - ols.predict(df[FINAL])
edges = np.linspace(df["specWid"].quantile(0.005), df["specWid"].quantile(0.995), 26)
cen = 0.5*(edges[:-1]+edges[1:])
rmed = [np.median(res[(df["specWid"] >= edges[i]) & (df["specWid"] < edges[i+1])])
        if ((df["specWid"] >= edges[i]) & (df["specWid"] < edges[i+1])).sum() >= 30 else np.nan
        for i in range(len(cen))]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sub = df.sample(min(6000, len(df)), random_state=0)
ax[0].scatter(sub["specWid"], sub["dV"], s=4, alpha=0.15, color="gray")
xs = np.linspace(edges[0], edges[-1], 50)
wbar = df["wind_speed"].median()
grid = pd.DataFrame({"specWid": xs, "wind_speed": wbar,
                     "specWid2": xs**2, "sWxWS": xs*wbar})[FINAL]
ax[0].plot(xs, ols.predict(grid), "-", color="steelblue", lw=2,
           label=f"OLS {best_name} (at median wind)")
ax[0].plot(xs, qr.params["Intercept"] + sum(qr.params[f]*grid[f].values for f in FINAL),
           "--", color="crimson", lw=2, label="median regression")
ax[0].set_xlabel("specWid (m/s)"); ax[0].set_ylabel("|ΔV| (m/s)")
ax[0].set_title("Final fits over the data"); ax[0].legend()

ax[1].axhline(0, color="k", lw=0.8)
ax[1].plot(cen, rmed, "o-", ms=4, color="darkorange")
ax[1].set_xlabel("specWid (m/s)"); ax[1].set_ylabel("median residual (m/s)")
ax[1].set_title("Residuals vs specWid (structure = wrong form)")
plt.tight_layout(); plt.show()
print("done")

