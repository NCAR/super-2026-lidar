#!/usr/bin/env python
"""
LOTOS Marshall (refined): Spearman screening matrix for |dV|.

Uses the CLEAN matcher (stored variables, nearest gate <= 25 m, no interpolation)
that fixed the bias artifact. |dV| built from stored speeds/directions:
  dV = sqrt(dU^2 + dV_comp^2) where dU,dV from each product's speed+direction.

Screens |dV| against MAPR quality + derived candidates. Sorted dV row printed;
heatmap shown. Near-0 -> drop; strong -> candidate; prune correlated clusters.
Shows plot; saves nothing. Needs numpy, pandas, matplotlib, netCDF4.
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
warnings.filterwarnings("ignore")

MAPR_DIR  = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/mod_prof/winds_nc"
LIDAR_DIR = "/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr31"
SITE_ALT_FALLBACK = 1742.0
MAPR_FILL, VAD_FILL = -999.0, -9999.0
TIME_TOL_S = 300
HEIGHT_TOL = 25.0

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
    snrw, sw = mg(m, "snrw"), mg(m, "specWid")
    ud, vd, wd = mg(m, "u_dispersion"), mg(m, "v_dispersion"), mg(m, "w_dispersion")
    cn, wv = mg(m, "cons_npoints"), mg(m, "wvert")
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
            rows.append(dict(
                dV=np.hypot(mu - lu, mvv - lvv),
                specWid=sw[i][g] if sw is not None else np.nan,
                snrw=snrw[i][g] if snrw is not None else np.nan,
                u_dispersion=ud[i][g] if ud is not None else np.nan,
                v_dispersion=vd[i][g] if vd is not None else np.nan,
                w_dispersion=wd[i][g] if wd is not None else np.nan,
                cons_npoints=cn[i][g] if cn is not None else np.nan,
                abs_wvert=abs(wv[i][g]) if wv is not None else np.nan,
                height=hm, wind_speed=0.5*(sm + vsp[k]),
            ))

df = pd.DataFrame(rows)
df = df[[c for c in df.columns if df[c].notna().any()]]
print(f"matched gates: {len(df)}   candidates: {df.shape[1]-1}\n")

sp = df.corr(method="spearman")
dv_row = sp["dV"].drop("dV")
dv_row = dv_row.reindex(dv_row.abs().sort_values(ascending=False).index)
print("Spearman vs dV (sorted by |r|):")
for k, val in dv_row.items():
    print(f"   {k:16s} {val:+.3f}")

order = ["dV"] + list(dv_row.index)
M = sp.loc[order, order]; k = len(order)
fig, ax = plt.subplots(figsize=(1.05*k + 2, 0.95*k + 1))
im = ax.imshow(M.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(k)); ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_yticks(range(k)); ax.set_yticklabels(order)
for a in range(k):
    for b in range(k):
        ax.text(b, a, f"{M.values[a,b]:.2f}", ha="center", va="center", fontsize=7)
ax.axhline(0.5, color="k", lw=1.5); ax.axvline(0.5, color="k", lw=1.5)
fig.colorbar(im, label="Spearman r")
ax.set_title("LOTOS (refined): |ΔV| vs candidates — top row/col = target")
plt.tight_layout(); plt.show()
print("done")

