#!/usr/bin/env python
"""
MAPR 449 MHz wind profiler vs Windcube lidar VAD -- full analysis pipeline.

MAPR is the primary dataset: each MAPR range gate is a datapoint described by the
MAPR's OWN quality metrics (snrw, specWid, u/v/w dispersion, cons_npoints,
|vertical wind|, height). The lidar VAD is the reference truth -- its speed and
direction are interpolated onto each MAPR gate. Everything is on height AGL
(MAPR datum auto-detected; lidar is AGL).

Repeats every plot family from the lidar work:
  1. Spearman feature correlation  (MAPR-only, and with the lidar reference)
  2. Outlier histograms across each MAPR parameter
  3. In-band (+-20%) histograms across each MAPR parameter
  4. Feature-importance models (80/20 train/test split by DAY, random forest,
     permutation importance) for targets a/b/c

MODE switch:
  "30" -> MAPR winds.30  vs  lidar vad_consensus (30-min)   [start here]
  "05" -> MAPR winds.05  vs  lidar vad_cnr33                [switch after]

RUN_ALL prints all rankings (headless-safe) then shows plots (need a display).
Shows plots; saves nothing. Needs scikit-learn, pandas, numpy, matplotlib, netCDF4.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, roc_curve
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
MODE = "30"                                  # "30" or "05"
WINDS = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
WC    = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube"
if MODE == "30":
    MAPR_GLOB = os.path.join(WINDS, "prof449.*.winds.30.nc")
    LIDAR_DIR = os.path.join(WC, "vad_consensus")
    TIME_TOL_S = 900
elif MODE == "05":
    MAPR_GLOB = os.path.join(WINDS, "prof449.*.winds.05.nc")
    LIDAR_DIR = os.path.join(WC, "vad_cnr33")
    TIME_TOL_S = 300
else:
    raise SystemExit("MODE must be '30' or '05'")

MAPR_FILL, VAD_FILL = -999.0, -9999.0
SPD_TOL, DIR_TOL = 2.0, 30.0                 # outlier: |dspeed|>SPD_TOL or |ddir|>DIR_TOL
GOOD_REL, SPD_FLOOR = 0.20, 2.0              # in-band: within +-20% of lidar, ref>=floor
TEST_FRAC = 0.2
SUBSAMPLE_N = 200_000
RF_TREES = 300

PARAMS = ["height", "snrw", "specWid", "u_dispersion", "v_dispersion",
          "w_dispersion", "cons_npoints", "abs_wvert"]
LABEL = {"a_error": "(a) |MAPR-lidar| error", "b_inband": "(b) in-band flag",
         "c_speed": "(c) MAPR wind speed"}
# -----------------------------------------------------------------------------

def to_nan(v, fill):
    a = v[:]
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    a = np.asarray(a, float); a[a == fill] = np.nan
    return a

def mapr_epoch(ds):
    return float(np.asarray(ds["base_time"][...])) + np.asarray(ds["time"][:], float)

def lidar_epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    if "time" in v:
        t = np.asarray(v["time"][:], float)
        return t if np.nanmax(t) > 1e8 else (float(np.asarray(v["base_time"][...])) + t
                                             if "base_time" in v else t)
    return None

def mget(ds, name):
    return to_nan(ds[name], MAPR_FILL) if name in ds.variables else None

def lget(ds, names):
    for nm in (names if isinstance(names, list) else [names]):
        if nm in ds.variables: return to_nan(ds[nm], VAD_FILL)
    return None

def met_dir(u, v): return (np.degrees(np.arctan2(-u, -v))) % 360.0
def circ(a, b): return np.abs(((a - b + 180) % 360) - 180)

def prep(h, *fields):
    ok = np.isfinite(h)
    for f in fields: ok &= np.isfinite(f)
    if ok.sum() < 2: return None
    o = np.argsort(h[ok])
    return (h[ok][o],) + tuple(f[ok][o] for f in fields)

def datemap(d, pat=r"(\d{8})"):
    out = {}
    for f in glob.glob(os.path.join(d, "*.nc")):
        m = re.search(pat, os.path.basename(f))
        if m: out[m.group(1)] = f
    return out

# ----------------------------------------------------------------------------- load
def load_matched():
    mapr_files = sorted(glob.glob(MAPR_GLOB))
    lid = datemap(LIDAR_DIR)
    if not mapr_files: raise SystemExit(f"No MAPR files: {MAPR_GLOB}")
    if not lid: raise SystemExit(f"No lidar files in {LIDAR_DIR}")

    cols = {k: [] for k in ["day", "scan_t", "height", "mapr_speed", "mapr_dir",
                            "snrw", "specWid", "u_dispersion", "v_dispersion",
                            "w_dispersion", "cons_npoints", "abs_wvert",
                            "ref_speed", "ref_dir"]}
    n_days = 0
    for mf in mapr_files:
        date = re.search(r"prof449\.(\d{8})\.", os.path.basename(mf)).group(1)
        if date not in lid: continue
        m = nc.Dataset(mf)
        site_alt = float(np.asarray(m["alt"][...]))
        m_t = mapr_epoch(m)
        m_h = to_nan(m["height"], MAPR_FILL)
        sp, di = mget(m, "wspd"), mget(m, "wdir")
        snrw, sw = mget(m, "snrw"), mget(m, "specWid")
        ud, vd, wd = mget(m, "u_dispersion"), mget(m, "v_dispersion"), mget(m, "w_dispersion")
        cn, wv = mget(m, "cons_npoints"), mget(m, "wvert")
        m.close()
        if sp is None or m_h is None: continue
        m_h_agl = m_h - site_alt if np.nanmin(m_h) > site_alt - 200 else m_h

        d = nc.Dataset(lid[date])
        l_t = lidar_epoch(d)
        l_h = np.asarray(lget(d, "height"), float)
        l_sp = lget(d, "wind_speed"); l_di = lget(d, "wind_direction")
        l_u, l_v = lget(d, "u"), lget(d, "v")
        d.close()
        if l_sp is None or l_h is None or l_t is None: continue
        l2d = (l_h.ndim == 2)
        n_days += 1

        for i in range(sp.shape[0]):
            j = int(np.argmin(np.abs(l_t - m_t[i])))
            if abs(l_t[j] - m_t[i]) > TIME_TOL_S: continue
            lh = l_h[j] if l2d else l_h
            ps = prep(lh, l_sp[j])
            if ps is None: continue
            sh, ssp = ps
            hcol = m_h_agl[i]
            v = np.isfinite(sp[i]) & (hcol >= sh.min()) & (hcol <= sh.max())
            if not v.any(): continue
            ref_sp = np.interp(hcol[v], sh, ssp)
            if l_u is not None and l_v is not None:
                pu = prep(lh, l_u[j], l_v[j])
                rdir = met_dir(np.interp(hcol[v], pu[0], pu[1]),
                               np.interp(hcol[v], pu[0], pu[2])) if pu else np.full(v.sum(), np.nan)
            else:
                rdir = np.full(v.sum(), np.nan)
            nN = int(v.sum())
            cols["day"] += [date]*nN; cols["scan_t"] += [m_t[i]]*nN
            cols["height"].extend(hcol[v]); cols["mapr_speed"].extend(sp[i][v])
            cols["mapr_dir"].extend(di[i][v] if di is not None else [np.nan]*nN)
            cols["snrw"].extend(snrw[i][v] if snrw is not None else [np.nan]*nN)
            cols["specWid"].extend(sw[i][v] if sw is not None else [np.nan]*nN)
            cols["u_dispersion"].extend(ud[i][v] if ud is not None else [np.nan]*nN)
            cols["v_dispersion"].extend(vd[i][v] if vd is not None else [np.nan]*nN)
            cols["w_dispersion"].extend(wd[i][v] if wd is not None else [np.nan]*nN)
            cols["cons_npoints"].extend(cn[i][v] if cn is not None else [np.nan]*nN)
            cols["abs_wvert"].extend(np.abs(wv[i][v]) if wv is not None else [np.nan]*nN)
            cols["ref_speed"].extend(ref_sp); cols["ref_dir"].extend(rdir)

    df = pd.DataFrame(cols)
    feats = [p for p in PARAMS if df[p].notna().any()]
    print(f"matched MAPR gates: {len(df)}   days: {df['day'].nunique()}")
    print(f"MAPR features present: {feats}")
    if SUBSAMPLE_N and len(df) > SUBSAMPLE_N:
        df = df.sample(SUBSAMPLE_N, random_state=0).reset_index(drop=True)
    return df, feats

# ----------------------------------------------------------------------------- derived
def derive(df):
    df["dspd"] = df["mapr_speed"] - df["ref_speed"]
    df["ddir"] = circ(df["mapr_dir"], df["ref_dir"])
    df["ref_err"] = np.abs(df["dspd"])
    df["matched"] = np.isfinite(df["ref_speed"])
    df["outlier"] = ((df["dspd"].abs() > SPD_TOL) |
                     (df["ddir"].fillna(0) > DIR_TOL)) & df["matched"]
    df["eval_ib"] = df["matched"] & (df["ref_speed"] >= SPD_FLOOR)
    df["inband"] = df["eval_ib"] & ((df["dspd"].abs() / df["ref_speed"]) <= GOOD_REL)
    return df

# ----------------------------------------------------------------------------- plots: correlation
def plot_correlation(df, cols, title):
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    sp = sub.corr(method="spearman")
    print(f"\nSpearman [{title}] n={len(sub)} (|r|>0.8 redundant):\n{sp.round(2).to_string()}")
    k = len(cols)
    fig, ax = plt.subplots(figsize=(0.95*k + 2, 0.85*k + 1.5))
    im = ax.imshow(sp.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(k)); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(cols)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sp.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Spearman r"); ax.set_title(f"Feature correlation — {title}")
    plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- plots: histogram grid
def hist_grid(df, params, pop, flag, color, flabel, suptitle):
    rows, cols = 2, int(np.ceil(len(params)/2))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4*cols, 9))
    for axp, p in zip(np.atleast_1d(axes).ravel(), params):
        a = df.loc[pop, p].values; g = df.loc[pop, flag].values.astype(bool)
        a = a.astype(float); ok = np.isfinite(a); a, g = a[ok], g[ok]
        if a.size == 0: axp.set_title(f"{p}\n(no data)"); axp.axis("off"); continue
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        if lo == hi: lo, hi = a.min(), a.max() + 1e-9
        edges = np.linspace(lo, hi, 21)
        axp.hist(a, bins=edges, color="lightgray", label="all")
        axp.hist(a[g], bins=edges, color=color, alpha=0.85, label=flabel)
        axp.set_xlabel(p); axp.set_ylabel("count")
        cen = 0.5*(edges[:-1]+edges[1:])
        tot, _ = np.histogram(a, bins=edges); sub, _ = np.histogram(a[g], bins=edges)
        frac = np.divide(sub, tot, out=np.zeros(len(tot)), where=tot > 0)
        axt = axp.twinx(); axt.plot(cen, 100*frac, "k.-", ms=3, lw=0.8)
        axt.set_ylabel(f"{flabel} %", fontsize=8); axt.set_ylim(0, 100)
    np.atleast_1d(axes).ravel()[0].legend(fontsize=8)
    fig.suptitle(suptitle, y=1.0); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- ML
def build_target(df, which, feats):
    if which == "c_speed":
        return df["mapr_speed"].values, "regression", np.ones(len(df), bool)
    ref = df["ref_speed"].values
    if which == "a_error":
        return np.abs(df["mapr_speed"].values - ref), "regression", np.isfinite(ref)
    rel = np.abs(df["mapr_speed"].values - ref) / df["mapr_speed"].values
    return (df["inband"].values.astype(int)), "classification", df["eval_ib"].values

def run_target(df, feats, which):
    y_all, task, mask = build_target(df, which, feats)
    sub = df.loc[mask].reset_index(drop=True)
    X, y, g = sub[feats], y_all[mask], sub["day"].values
    fin = X.notna().all(axis=1).values & np.isfinite(y)
    X, y, g = X[fin].reset_index(drop=True), y[fin], g[fin]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
    tr, te = next(gss.split(X, y, g))
    Model = RandomForestRegressor if task == "regression" else RandomForestClassifier
    scoring = "r2" if task == "regression" else "roc_auc"
    mdl = Model(n_estimators=RF_TREES, n_jobs=-1, random_state=0).fit(X.iloc[tr], y[tr])
    pred = (mdl.predict(X.iloc[te]) if task == "regression"
            else mdl.predict_proba(X.iloc[te])[:, 1])
    pi = permutation_importance(mdl, X.iloc[te], y[te], scoring=scoring,
                                n_repeats=10, random_state=0, n_jobs=-1)
    return dict(which=which, task=task, scoring=scoring, features=feats, X_test=X.iloc[te],
                y_test=y[te], pred=pred, final=mdl, imp_mean=pi.importances_mean,
                imp_std=pi.importances_std, n_train=len(tr), n_test=len(te),
                days_train=len(set(g[tr])), days_test=len(set(g[te])))

def report(r):
    score = (r2_score(r["y_test"], r["pred"]) if r["task"] == "regression"
             else roc_auc_score(r["y_test"], r["pred"]))
    extra = (f"  MAE {mean_absolute_error(r['y_test'], r['pred']):.3f}" if r["task"] == "regression"
             else f"  (test positives {100*np.mean(r['y_test']):.1f}%)")
    print(f"\n=== {LABEL[r['which']]} | {r['task']} | train {r['n_train']} ({r['days_train']}d)"
          f" / test {r['n_test']} ({r['days_test']}d) | test {r['scoring']} {score:.3f}{extra} ===")
    for i in np.argsort(r["imp_mean"])[::-1]:
        print(f"   {r['features'][i]:14s} {r['imp_mean'][i]:+.4f} ± {r['imp_std'][i]:.4f}")

def plot_target(r):
    order = np.argsort(r["imp_mean"])[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].barh([r["features"][i] for i in order][::-1], r["imp_mean"][order][::-1],
               xerr=r["imp_std"][order][::-1], color="steelblue")
    ax[0].axvline(0, color="k", lw=0.8); ax[0].set_xlabel(f"perm. importance ({r['scoring']})")
    ax[0].set_title(LABEL[r["which"]])
    if r["task"] == "regression":
        ax[1].scatter(r["y_test"], r["pred"], s=4, alpha=0.2)
        lo, hi = r["y_test"].min(), r["y_test"].max(); ax[1].plot([lo, hi], [lo, hi], "k--")
        ax[1].set_xlabel("actual (test)"); ax[1].set_ylabel("predicted"); ax[1].set_title("Test fit")
    else:
        fpr, tpr, _ = roc_curve(r["y_test"], r["pred"]); ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--")
        ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("Test ROC")
    top = r["features"][order[0]]
    PartialDependenceDisplay.from_estimator(r["final"], r["X_test"], [top], ax=ax[2])
    ax[2].set_title(f"partial dependence: {top}")
    plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- main
def main():
    print(f"MODE={MODE}  (MAPR winds.{MODE} vs {os.path.basename(LIDAR_DIR)})")
    df, feats = load_matched()
    if len(df) == 0: raise SystemExit("No matched points -- check paths / TIME_TOL_S / datum.")
    df = derive(df)
    print(f"\noutliers: {100*df['outlier'][df['matched']].mean():.1f}% of matched | "
          f"in-band: {100*df['inband'][df['eval_ib']].mean():.1f}% of evaluated")

    targets = ["a_error", "b_inband", "c_speed"]
    for w in targets:
        report(run_target(df, feats, w))

    # 1. Spearman (MAPR features, and with lidar reference)
    plot_correlation(df, feats, "MAPR quality features")
    plot_correlation(df, feats + ["mapr_speed", "ref_speed", "ref_err"], "with lidar reference")
    # 2. outlier histograms
    hist_grid(df, feats, df["matched"].values, "outlier", "crimson", "outliers",
              f"MAPR-lidar outliers (Δspeed>{SPD_TOL} or Δdir>{DIR_TOL}°), MODE {MODE}")
    # 3. in-band histograms
    hist_grid(df, feats, df["eval_ib"].values, "inband", "seagreen", "in-band",
              f"MAPR-lidar agreement within +-{int(GOOD_REL*100)}%, MODE {MODE}")
    # 4. feature-importance model plots
    for w in targets:
        plot_target(run_target(df, feats, w))
    print("\ndone")


if __name__ == "__main__":
    main()


