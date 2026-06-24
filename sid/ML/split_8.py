#!/usr/bin/env python
"""
Feature importance for M2HATS Windcube VAD datapoints -- FINAL, real data.

Reads the vad_cnr_40 product (the looser cutoff: it keeps the bad gates, so the
quality features span their full range -- the consensus and -33 products are
censored and would hide the very boundary we want to learn).

Confirmed stored features: height, residual, correlation, mean_snr
(+ abs_w, npoints auto-added if present). CNR is NOT stored, so it is not a
feature despite the directory name.

Targets:
  (c) c_speed         regression on VAD wind speed              [always available]
  (d) d_passes_strict classification: does the gate also pass the strict -33
                      cutoff? label from vad_cnr33; non-circular because CNR is
                      not a feature                              [needs STRICT_DIR]
  (a) a_error         regression on |VAD speed - reference|      [needs REF]
  (b) b_inband        classification: VAD within +-GOOD_REL of reference [needs REF]

Rigor: ONE train/test split held out BY DAY (no day in both sides -> no
autocorrelation leakage); PERMUTATION importance on the held-out test set (not
impurity); Spearman redundancy check; partial dependence for the top feature.

RUN_ALL=True computes every available target and prints rankings (works on a
headless node); it then tries the plots, which need a display (ssh -X/Jupyter).
RUN_ALL=False gives the interactive menu. Shows plots; saves nothing.
Needs scikit-learn, pandas, numpy, matplotlib, netCDF4.
"""

import sys, glob, os, re, warnings
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
BASE = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube"
PRODUCT_DIR = os.path.join(BASE, "vad_cnr_40")     # the product to model
STRICT_DIR  = os.path.join(BASE, "vad_cnr33")      # for target d; None to disable

REF      = "none"                                   # "none" | "era5" | "mapr"  (targets a/b)
ERA5_DIR = "/scr/isf_apg/models/m2hats/era5"
MAPR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
SITE_ALT_M = 1641.0                                 # Windcube altitude (report); for ERA5 MSL->AGL

VAD_FILL, MAPR_FILL = -9999.0, -999.0
GOOD_REL, SPD_FLOOR = 0.20, 2.0
REF_TOL_S = 1800
SUBSAMPLE_N = 150_000        # cap rows (RF on millions is slow); None = use all
TEST_FRAC = 0.2              # held-out test fraction (split by DAY, not by row)
RUN_ALL = True

VARS = {"height": ["height"], "wspd": ["wind_speed"], "wdir": ["wind_direction"],
        "residual": ["residual"], "correlation": ["correlation"], "snr": ["mean_snr"],
        "w": ["w"], "npoints": ["w_npoints", "npoints", "u_npoints"]}
BASE_FEATURES = ["height", "residual", "correlation", "snr"]   # confirmed present
LABEL = {"c_speed": "(c) VAD wind speed", "d_passes_strict": "(d) passes strict -33",
         "a_error": "(a) |VAD-ref| error", "b_inband": "(b) in-band flag"}
# -----------------------------------------------------------------------------

def to_nan(v, fill):
    a = v[:]
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    a = np.asarray(a, float); a[a == fill] = np.nan
    return a

def getv(ds, key, fill=VAD_FILL):
    for nm in VARS[key]:
        if nm in ds.variables:
            return to_nan(ds[nm], fill)
    return None

def epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    if "time" in v:
        t = np.asarray(v["time"][:], float)
        return t if np.nanmax(t) > 1e8 else (float(np.asarray(v["base_time"][...])) + t
                                             if "base_time" in v else t)
    return None

def datemap(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.nc")):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if m: out[m.group(1)] = f
    return out

def circ(a, b): return np.abs(((a - b + 180) % 360) - 180)

# ----------------------------------------------------------------------------- load VAD
def load_vad():
    m40 = datemap(PRODUCT_DIR)
    m33 = datemap(STRICT_DIR) if STRICT_DIR else {}
    if not m40:
        raise SystemExit(f"No files in {PRODUCT_DIR}")
    rows, feats_present = [], set()
    for dt in sorted(m40):
        ds = nc.Dataset(m40[dt])
        h = getv(ds, "height"); s = getv(ds, "wspd"); d = getv(ds, "wdir")
        res, cor, snr = getv(ds, "residual"), getv(ds, "correlation"), getv(ds, "snr")
        w, npts = getv(ds, "w"), getv(ds, "npoints")
        t40 = epoch(ds); ds.close()
        if s is None or h is None:
            continue
        h2d = (h.ndim == 2)

        in_strict = None
        if dt in m33:
            d33 = nc.Dataset(m33[dt]); s33 = getv(d33, "wspd"); t33 = epoch(d33); d33.close()
            if s33 is not None:
                in_strict = np.zeros_like(s, bool)
                if t40 is not None and t33 is not None:
                    for i, tt in enumerate(t40):
                        j = int(np.argmin(np.abs(t33 - tt)))
                        if abs(t33[j] - tt) <= 1.0:
                            in_strict[i] = np.isfinite(s33[j])
                elif s33.shape == s.shape:
                    in_strict = np.isfinite(s33)

        nt = s.shape[0]
        for i in range(nt):
            hc = h[i] if h2d else h
            v = np.isfinite(s[i])
            if not v.any():
                continue
            rec = {"day": dt, "height": hc[v], "residual": res[i][v] if res is not None else np.nan,
                   "correlation": cor[i][v] if cor is not None else np.nan,
                   "snr": snr[i][v] if snr is not None else np.nan,
                   "vad_speed": s[i][v], "vad_dir": d[i][v] if d is not None else np.nan,
                   "scan_t": (t40[i] if t40 is not None else np.nan)}
            if w is not None: rec["abs_w"] = np.abs(w[i][v]); feats_present.add("abs_w")
            if npts is not None: rec["npoints"] = npts[i][v]; feats_present.add("npoints")
            if in_strict is not None: rec["passes_strict"] = in_strict[i][v].astype(int)
            rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    features = [f for f in BASE_FEATURES if df[f].notna().any()] + sorted(feats_present)
    if SUBSAMPLE_N and len(df) > SUBSAMPLE_N:
        df = df.sample(SUBSAMPLE_N, random_state=0).reset_index(drop=True)
    return df, features

# ----------------------------------------------------------------------------- reference (a/b)
def ref_profiles(date):
    """Per-day list of (epoch, height_AGL_sorted, speed_sorted) for the chosen REF."""
    out = []
    if REF == "era5":
        for f in sorted(glob.glob(os.path.join(ERA5_DIR, date, f"era5_pressure_{date}_*_ISS1.nc"))):
            ds = nc.Dataset(f)
            u = np.asarray(ds["u"][0, :, 0, 0], float); v = np.asarray(ds["v"][0, :, 0, 0], float)
            z = np.asarray(ds["z"][0, :, 0, 0], float); t = float(np.asarray(ds["valid_time"][0]))
            ds.close()
            ah = z / 9.80665 - SITE_ALT_M; sp = np.hypot(u, v)
            ok = np.isfinite(ah) & np.isfinite(sp)
            if ok.sum() > 1:
                o = np.argsort(ah[ok]); out.append((t, ah[ok][o], sp[ok][o]))
    elif REF == "mapr":
        f = os.path.join(MAPR_DIR, f"prof449.{date}.winds.30.nc")
        if os.path.exists(f):
            ds = nc.Dataset(f)
            alt = float(np.asarray(ds["alt"][...])); t = epoch(ds)
            hh = to_nan(ds["height"], MAPR_FILL); sp = to_nan(ds["wspd"], MAPR_FILL); ds.close()
            agl = hh - alt if np.nanmin(hh) > alt - 200 else hh
            for i in range(sp.shape[0]):
                ok = np.isfinite(agl[i]) & np.isfinite(sp[i])
                if ok.sum() > 1:
                    o = np.argsort(agl[i][ok]); out.append((t[i], agl[i][ok][o], sp[i][ok][o]))
    return out

def add_reference(df):
    if REF == "none":
        return None
    ref_speed = np.full(len(df), np.nan)
    for day, grp in df.groupby("day"):
        prof = ref_profiles(day)
        if not prof:
            continue
        rt = np.array([p[0] for p in prof])
        for idx in grp.index:
            tt, hh = df.at[idx, "scan_t"], df.at[idx, "height"]
            k = int(np.argmin(np.abs(rt - tt)))
            if abs(rt[k] - tt) <= REF_TOL_S:
                _, rh, rs = prof[k]
                if rh.min() <= hh <= rh.max():
                    ref_speed[idx] = np.interp(hh, rh, rs)
    return ref_speed

# ----------------------------------------------------------------------------- targets
def build_target(df, which):
    n = len(df)
    if which == "c_speed":
        return df["vad_speed"].values, "regression", np.ones(n, bool)
    if which == "d_passes_strict":
        return df["passes_strict"].values, "classification", np.ones(n, bool)
    if which in ("a_error", "b_inband"):
        ref = df["ref_speed"].values
        if which == "a_error":
            return np.abs(df["vad_speed"].values - ref), "regression", np.isfinite(ref)
        rel = np.abs(df["vad_speed"].values - ref) / df["vad_speed"].values
        return (rel <= GOOD_REL).astype(int), "classification", np.isfinite(ref) & (df["vad_speed"].values >= SPD_FLOOR)
    raise ValueError(which)

def available_targets(df):
    t = ["c_speed"]
    if "passes_strict" in df: t.append("d_passes_strict")
    if "ref_speed" in df and np.isfinite(df["ref_speed"]).any(): t += ["a_error", "b_inband"]
    return t

# ----------------------------------------------------------------------------- engine
def run_target(df, features, which):
    y_all, task, mask = build_target(df, which)
    sub = df.loc[mask].reset_index(drop=True)
    X, y, g = sub[features], y_all[mask], sub["day"].values
    fin = X.notna().all(axis=1).values & np.isfinite(y)
    X, y, g = X[fin].reset_index(drop=True), y[fin], g[fin]
    # ONE train/test split, held out BY DAY so no day lands on both sides
    # (a random row split would leak: adjacent gates/times are near-identical)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
    tr, te = next(gss.split(X, y, g))
    Model = RandomForestRegressor if task == "regression" else RandomForestClassifier
    scoring = "r2" if task == "regression" else "roc_auc"
    mdl = Model(n_estimators=300, n_jobs=-1, random_state=0).fit(X.iloc[tr], y[tr])
    pred = (mdl.predict(X.iloc[te]) if task == "regression"
            else mdl.predict_proba(X.iloc[te])[:, 1])
    pi = permutation_importance(mdl, X.iloc[te], y[te], scoring=scoring,
                                n_repeats=10, random_state=0, n_jobs=-1)
    return dict(which=which, task=task, scoring=scoring, features=features,
                X_test=X.iloc[te], y_test=y[te], pred=pred, final=mdl,
                imp_mean=pi.importances_mean, imp_std=pi.importances_std,
                n_train=len(tr), n_test=len(te),
                days_train=len(set(g[tr])), days_test=len(set(g[te])))

def report(r):
    score = (r2_score(r["y_test"], r["pred"]) if r["task"] == "regression"
             else roc_auc_score(r["y_test"], r["pred"]))
    extra = (f"  MAE {mean_absolute_error(r['y_test'], r['pred']):.3f}" if r["task"] == "regression"
             else f"  (test positives {100*np.mean(r['y_test']):.1f}%)")
    print(f"\n=== {LABEL[r['which']]} | {r['task']} | "
          f"train {r['n_train']} ({r['days_train']}d) / test {r['n_test']} ({r['days_test']}d) | "
          f"test {r['scoring']} {score:.3f}{extra} ===")
    for i in np.argsort(r["imp_mean"])[::-1]:
        print(f"   {r['features'][i]:13s} {r['imp_mean'][i]:+.4f} ± {r['imp_std'][i]:.4f}")

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

def plot_correlation(df, features):
    sp = df[features].corr(method="spearman")
    print("\nSpearman correlation (|r|>0.8 = redundant):"); print(sp.round(2).to_string())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sp.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{sp.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Spearman r"); ax.set_title("Feature correlation")
    plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- main
def main():
    print("loading vad_cnr_40 ...")
    df, features = load_vad()
    if REF != "none":
        print(f"joining reference ({REF}) ...")
        rs = add_reference(df)
        if rs is not None: df["ref_speed"] = rs
    targets = available_targets(df)
    print(f"rows: {len(df)}   days: {df['day'].nunique()}   features: {features}")
    print(f"available targets: {targets}")
    cache = {}

    def get(w):
        if w not in cache:
            print(f"... computing {LABEL[w]} ...")
            cache[w] = run_target(df, features, w)
        return cache[w]

    if RUN_ALL:
        for w in targets:
            report(get(w))
        plot_correlation(df, features)
        for w in targets:
            plot_target(get(w))
        print("\ndone"); return

    while True:
        print("\n" + "-"*48)
        for k, w in enumerate(targets, 1):
            print(f" {k}. {LABEL[w]}")
        print(f" {len(targets)+1}. feature correlation matrix")
        print(" 0. quit")
        try:
            c = input("select > ").strip()
        except EOFError:
            break
        if c == "0":
            break
        if c.isdigit() and 1 <= int(c) <= len(targets):
            plot_target(get(targets[int(c)-1]))
        elif c == str(len(targets)+1):
            plot_correlation(df, features)
        else:
            print("  not an option")
    print("done")


if __name__ == "__main__":
    main()

