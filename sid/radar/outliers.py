#!/usr/bin/env python
"""
Compare 30-min MAPR 449 MHz radar profiler winds against the Windcube VAD
lidar 30-min consensus winds, at the co-located M2HATS ISS1 site.

Matches in time (nearest, within tolerance) and in height (MAPR interpolated
onto VAD levels, via u/v to avoid direction wrap), within the overlap band.

Displays (does not save):
  speed + direction scatter, coloured by height AGL
  mean speed/direction bias vs height

Run on mercury:  python compare_mapr_vad_30min.py
"""

import matplotlib.pyplot as plt            # needs a display: ssh -X / Jupyter / VS Code remote
import netCDF4 as nc
import numpy as np
import glob, os, re, warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
MAPR_DIR = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/mod_prof/winds_nc"
VAD_DIR  = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus"

MAPR_FILL = -999.0
VAD_FILL  = -9999.0

TIME_TOL_S   = 900       # max |t_mapr - t_vad| for a time match (s); 30-min data
MAPR_MIN_CONS = 0        # require cons_npoints >= this for a MAPR gate (0 = off; try 50)
OVERLAP_PAD  = 0.0       # m; shrink overlap band at both ends if you want margin
SPD_TOL = 2.0            # m/s : speed disagreement that flags an outlier
DIR_TOL = 30.0           # deg : direction disagreement that flags an outlier
# -----------------------------------------------------------------------------

def to_nan(var, fill):
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    a = np.asarray(a, dtype=float)
    a[a == fill] = np.nan
    return a

def mapr_epoch(ds):
    bt = float(np.asarray(ds["base_time"][...]))
    return bt + np.asarray(ds["time"][:], dtype=float)        # center of avg period

def vad_epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    t = np.asarray(v["time"][:], dtype=float)
    if np.nanmax(t) > 1e8:                                    # already epoch seconds
        return t
    if "base_time" in v:
        return float(np.asarray(v["base_time"][...])) + t
    return t

def met_dir_from_uv(u, v):
    """Meteorological 'from' direction (deg east of north) from east/north wind."""
    return (np.degrees(np.arctan2(-u, -v))) % 360.0

def vget(d, name, fill, like):
    """Read a (time,height) VAD var, or return all-NaN of the same shape if absent."""
    if name in d.variables:
        return to_nan(d[name], fill)
    return np.full_like(like, np.nan)

# ----------------------------------------------------------------------------- gather
mapr_files = sorted(glob.glob(os.path.join(MAPR_DIR, "prof449.*.winds.30.nc")))
if not mapr_files:
    raise SystemExit("No MAPR winds.30 files found — check MAPR_DIR")

S_v, S_m, D_v, D_m, H = [], [], [], [], []   # vad/mapr speed, vad/mapr dir, height AGL
SNR, RES, COR, W, UNP, VNP, WNP = [], [], [], [], [], [], []   # VAD quality metrics at match
datum_reported = False
n_days = 0

for mf in mapr_files:
    date = re.search(r"prof449\.(\d{8})\.winds\.30\.nc", os.path.basename(mf)).group(1)
    vf = os.path.join(VAD_DIR, f"30min_winds_{date}.nc")
    if not os.path.exists(vf):
        continue

    # --- MAPR ---
    m = nc.Dataset(mf)
    site_alt = float(np.asarray(m["alt"][...]))
    m_t  = mapr_epoch(m)                       # (time,)
    m_h  = to_nan(m["height"], MAPR_FILL)      # (time, height) MSL-labelled
    m_u  = to_nan(m["u"], MAPR_FILL)
    m_v  = to_nan(m["v"], MAPR_FILL)
    if MAPR_MIN_CONS > 0 and "cons_npoints" in m.variables:
        cons = np.asarray(m["cons_npoints"][:], dtype=float)
        bad = cons < MAPR_MIN_CONS
        m_u[bad] = np.nan; m_v[bad] = np.nan
    m.close()

    # height datum: if min height ~ site elevation -> it's MSL, convert to AGL
    hmin = np.nanmin(m_h)
    if hmin > site_alt - 200.0:
        m_h_agl = m_h - site_alt
        mapr_datum = f"MSL (min={hmin:.0f} ~ alt={site_alt:.0f}) -> subtracted site alt"
    else:
        m_h_agl = m_h
        mapr_datum = f"AGL (min={hmin:.0f}, well below alt={site_alt:.0f})"

    # --- VAD ---
    d = nc.Dataset(vf)
    v_t  = vad_epoch(d)                         # (time,)
    v_h  = to_nan(d["height"], VAD_FILL)        # (height,) meters
    v_sp = to_nan(d["wind_speed"], VAD_FILL)    # (time, height)
    v_di = to_nan(d["wind_direction"], VAD_FILL)
    v_snr = vget(d, "mean_snr",    VAD_FILL, v_sp)
    v_res = vget(d, "residual",    VAD_FILL, v_sp)
    v_cor = vget(d, "correlation", VAD_FILL, v_sp)
    v_w   = vget(d, "w",           VAD_FILL, v_sp)
    v_unp = vget(d, "u_npoints",   VAD_FILL, v_sp)
    v_vnp = vget(d, "v_npoints",   VAD_FILL, v_sp)
    v_wnp = vget(d, "w_npoints",   VAD_FILL, v_sp)
    d.close()
    v_h = np.asarray(v_h, dtype=float).ravel()

    # VAD datum: same test
    vh_min = np.nanmin(v_h)
    if vh_min > site_alt - 200.0:
        v_h_agl = v_h - site_alt
        vad_datum = f"MSL (min={vh_min:.0f}) -> subtracted site alt"
    else:
        v_h_agl = v_h
        vad_datum = f"AGL (min={vh_min:.0f})"

    if not datum_reported:
        print(f"[datum]  MAPR height: {mapr_datum}")
        print(f"[datum]  VAD  height: {vad_datum}")
        print(f"[datum]  site alt = {site_alt:.0f} m MSL")
        datum_reported = True

    n_days += 1

    # --- time match: each VAD record -> nearest MAPR record within tolerance ---
    for vi in range(len(v_t)):
        dt = np.abs(m_t - v_t[vi])
        mi = int(np.argmin(dt))
        if dt[mi] > TIME_TOL_S:
            continue

        hcol = m_h_agl[mi, :]
        uu, vv = m_u[mi, :], m_v[mi, :]
        ok = np.isfinite(hcol) & np.isfinite(uu) & np.isfinite(vv)
        if ok.sum() < 2:
            continue
        order = np.argsort(hcol[ok])
        h_ref = hcol[ok][order]
        u_ref = uu[ok][order]
        v_ref = vv[ok][order]

        lo, hi = h_ref.min() + OVERLAP_PAD, h_ref.max() - OVERLAP_PAD
        sel = (np.isfinite(v_sp[vi]) & np.isfinite(v_di[vi]) &
               (v_h_agl >= lo) & (v_h_agl <= hi))
        if sel.sum() == 0:
            continue

        tgt_h = v_h_agl[sel]
        ui = np.interp(tgt_h, h_ref, u_ref)
        vi_ = np.interp(tgt_h, h_ref, v_ref)
        m_spd = np.hypot(ui, vi_)
        m_dir = met_dir_from_uv(ui, vi_)

        S_v.extend(v_sp[vi][sel]); S_m.extend(m_spd)
        D_v.extend(v_di[vi][sel]); D_m.extend(m_dir)
        H.extend(tgt_h)
        SNR.extend(v_snr[vi][sel]); RES.extend(v_res[vi][sel]); COR.extend(v_cor[vi][sel])
        W.extend(v_w[vi][sel])
        UNP.extend(v_unp[vi][sel]); VNP.extend(v_vnp[vi][sel]); WNP.extend(v_wnp[vi][sel])

S_v, S_m = np.array(S_v), np.array(S_m)
D_v, D_m = np.array(D_v), np.array(D_m)
H = np.array(H)
SNR, RES, COR, W = map(np.array, (SNR, RES, COR, W))
UNP, VNP, WNP = map(np.array, (UNP, VNP, WNP))
n = len(S_v)
if n == 0:
    raise SystemExit("No matched points — check date overlap / TIME_TOL_S / datum.")

# ----------------------------------------------------------------------------- stats
dir_diff = ((D_m - D_v + 180.0) % 360.0) - 180.0          # signed, [-180,180]
spd_bias = S_m - S_v
print(f"\n[match]  days with both instruments: {n_days}")
print(f"[match]  matched points: {n}")
print(f"[speed]  bias (MAPR-VAD): {np.mean(spd_bias):+.2f}  MAD {np.median(np.abs(spd_bias - np.median(spd_bias))):.2f}  "
      f"RMSE {np.sqrt(np.mean(spd_bias**2)):.2f}  r {np.corrcoef(S_v, S_m)[0,1]:.3f}  m/s")
print(f"[dir]    bias (MAPR-VAD): {np.mean(dir_diff):+.1f}  MAD {np.median(np.abs(dir_diff - np.median(dir_diff))):.1f}  deg")

# ----------------------------------------------------------------------------- plot 1
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
sc = ax[0].scatter(S_v, S_m, c=H, s=6, alpha=0.4, cmap="viridis")
mx = np.nanmax([S_v.max(), S_m.max()]) * 1.05
ax[0].plot([0, mx], [0, mx], "k--", lw=1)
ax[0].set_xlim(0, mx); ax[0].set_ylim(0, mx)
ax[0].set_xlabel("VAD lidar speed (m/s)"); ax[0].set_ylabel("MAPR profiler speed (m/s)")
ax[0].set_title(f"Wind speed  (n={n})\nbias {np.mean(spd_bias):+.2f}  RMSE {np.sqrt(np.mean(spd_bias**2)):.2f} m/s")
fig.colorbar(sc, ax=ax[0], label="height AGL (m)")

sc2 = ax[1].scatter(D_v, D_m, c=H, s=6, alpha=0.4, cmap="viridis")
ax[1].plot([0, 360], [0, 360], "k--", lw=1)
ax[1].set_xlim(0, 360); ax[1].set_ylim(0, 360)
ax[1].set_xlabel("VAD lidar dir (deg)"); ax[1].set_ylabel("MAPR profiler dir (deg)")
ax[1].set_title(f"Wind direction\nbias {np.mean(dir_diff):+.1f} deg")
fig.colorbar(sc2, ax=ax[1], label="height AGL (m)")
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- plot 2
bins = np.arange(0, np.nanmax(H) + 100, 100.0)
ctr = 0.5 * (bins[:-1] + bins[1:])
sb_mean, db_mean, sb_sd, cnt = [], [], [], []
for i in range(len(bins) - 1):
    s = (H >= bins[i]) & (H < bins[i + 1])
    cnt.append(s.sum())
    if s.sum() < 5:
        sb_mean.append(np.nan); db_mean.append(np.nan); sb_sd.append(np.nan); continue
    sb_mean.append(np.mean(spd_bias[s])); sb_sd.append(np.std(spd_bias[s]))
    db_mean.append(np.mean(dir_diff[s]))
sb_mean, db_mean, sb_sd = map(np.array, (sb_mean, db_mean, sb_sd))

fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
ax[0].axvline(0, color="gray", lw=0.8)
ax[0].errorbar(sb_mean, ctr, xerr=sb_sd, fmt="o-", ms=4, capsize=2)
ax[0].set_xlabel("speed bias MAPR-VAD (m/s)"); ax[0].set_ylabel("height AGL (m)")
ax[0].set_title("Speed bias profile (±1σ)")
ax[1].axvline(0, color="gray", lw=0.8)
ax[1].plot(db_mean, ctr, "o-", ms=4, color="crimson")
ax[1].set_xlabel("direction bias MAPR-VAD (deg)"); ax[1].set_title("Direction bias profile")
ax[2].plot(cnt, ctr, "o-", ms=4, color="k")
ax[2].set_xlabel("matched points"); ax[2].set_title("Sampling per level")
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- plot 3
# outlier = MAPR-VAD disagreement beyond tolerance, binned across the quality
# metrics we used before. Gray = all matched points, crimson = outliers,
# black line = outlier fraction per bin (the bit that actually matters).
outlier = (np.abs(spd_bias) > SPD_TOL) | (np.abs(dir_diff) > DIR_TOL)
print(f"[outlier] {100*outlier.mean():.1f}% of {n} points (Δspeed>{SPD_TOL} m/s or Δdir>{DIR_TOL} deg)")

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
    ok = np.isfinite(arr)
    a, o = arr[ok], outlier[ok]
    if a.size == 0:
        axp.set_title(f"{lab}\n(no data)"); axp.set_axis_off(); continue
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if lo == hi:
        lo, hi = a.min(), a.max() + 1e-9
    edges = np.linspace(lo, hi, 21)
    axp.hist(a, bins=edges, color="lightgray", label="all")
    axp.hist(a[o], bins=edges, color="crimson", alpha=0.85, label="outliers")
    axp.set_xlabel(lab); axp.set_ylabel("count")
    # outlier fraction per bin on a twin axis
    cen = 0.5 * (edges[:-1] + edges[1:])
    tot, _ = np.histogram(a, bins=edges)
    out, _ = np.histogram(a[o], bins=edges)
    frac = np.divide(out, tot, out=np.zeros(len(tot)), where=tot > 0)
    axt = axp.twinx()
    axt.plot(cen, 100 * frac, "k.-", ms=3, lw=0.8)
    axt.set_ylabel("outlier %", fontsize=8); axt.set_ylim(0, 100)
axes.ravel()[0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"MAPR-VAD outliers across quality metrics  ({100*outlier.mean():.1f}% of {n} pts)", y=1.0)
plt.tight_layout(); plt.show()

print("\ndone")

