#!/usr/bin/env python
"""
Compare two VAD wind products that differ only in the CNR acceptance cutoff:
  -33 dB (stricter, the report's operational threshold)  vs  -40 dB (looser).

Both are processed from the SAME scans, so per (time,height) gate the difference
is which gates survive QC and the wind fit at the gates that do. The script
classifies every gate as valid-in-both, only-in-(-40), or only-in-(-33), and
reports:
  * coverage gained by -40 (how many more gates, and where in height)
  * agreement where both are valid (should be small; large = the looser cutoff
    is letting noisy beams into the fit)
  * quality of the gates -40 rescues (residual / correlation / SNR / CNR of the
    "only -40" gates vs the "both" gates) -- is the extra data usable or noise?
  * how much higher each product reaches (max valid height per profile)

Assumes ARM-convention variable names and fill -9999 (same family as the
30-min consensus). It auto-detects which variables exist and prints that, so if
a name is off you'll see it. Shows plots; saves nothing.
"""

import numpy as np
import matplotlib.pyplot as plt            # ssh -X / Jupyter / VS Code remote
import netCDF4 as nc
import glob, os, re, warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
BASE = "/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube"
DIR33 = os.path.join(BASE, "vad_cnr33")     # exact names from your listing
DIR40 = os.path.join(BASE, "vad_cnr_40")    #  (note: cnr33 vs cnr_40)
FILL = -9999.0
TIME_TOL_S = 1.0                            # same scans -> timestamps should match
VARS = {                                    # logical -> candidate file names
    "height":      ["height"],
    "wspd":        ["wind_speed"],
    "wdir":        ["wind_direction"],
    "residual":    ["residual"],
    "correlation": ["correlation"],
    "snr":         ["mean_snr"],
    "cnr":         ["cnr", "CNR", "mean_cnr"],
}
# -----------------------------------------------------------------------------

def to_nan(v):
    a = v[:]
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    a = np.asarray(a, float); a[a == FILL] = np.nan
    return a

def getv(ds, key):
    for nm in VARS[key]:
        if nm in ds.variables:
            return to_nan(ds[nm])
    return None

def epoch(ds):
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    if "time" in v:
        t = np.asarray(v["time"][:], float)
        if np.nanmax(t) > 1e8: return t
        if "base_time" in v: return float(np.asarray(v["base_time"][...])) + t
        return t
    return None

def datemap(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.nc")):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if m: out[m.group(1)] = f
    return out

def circ(a, b): return np.abs(((a - b + 180) % 360) - 180)

# ----------------------------------------------------------------------------- gather
m33, m40 = datemap(DIR33), datemap(DIR40)
dates = sorted(set(m33) & set(m40))
if not dates:
    raise SystemExit(f"No shared dates.\n  {DIR33}: {len(m33)} files\n  {DIR40}: {len(m40)} files")
print(f"shared dates: {len(dates)}  ({dates[0]}..{dates[-1]})")

H, V33, V40 = [], [], []                 # per-gate height, valid-in-33, valid-in-40
DSP, DDR = [], []                        # speed/dir diff where BOTH valid
RES_b, RES_o, COR_b, COR_o = [], [], [], []   # quality at both vs only-40 gates
SNR_b, SNR_o, CNR_o = [], [], []
maxh33, maxh40 = [], []                  # per-profile max valid height
reported = False

for dt in dates:
    a, b = nc.Dataset(m33[dt]), nc.Dataset(m40[dt])
    if not reported:
        print("\nvariables found:")
        for k in VARS:
            print(f"  {k:11s}  -33:{'Y' if getv(a,k) is not None else '-'}"
                  f"   -40:{'Y' if getv(b,k) is not None else '-'}")
        reported = True

    h = getv(a, "height"); s33 = getv(a, "wspd"); d33 = getv(a, "wdir")
    s40 = getv(b, "wspd"); d40 = getv(b, "wdir")
    res40, cor40, snr40, cnr40 = (getv(b, "residual"), getv(b, "correlation"),
                                  getv(b, "snr"), getv(b, "cnr"))
    t33, t40 = epoch(a), epoch(b)
    a.close(); b.close()
    if s33 is None or s40 is None or h is None:
        continue
    h2d = (h.ndim == 2)

    # align profiles by timestamp (same scans -> usually identical)
    if t33 is not None and t40 is not None:
        pairs = []
        for i, tt in enumerate(t33):
            j = int(np.argmin(np.abs(t40 - tt)))
            if abs(t40[j] - tt) <= TIME_TOL_S:
                pairs.append((i, j))
    elif s33.shape[0] == s40.shape[0]:
        pairs = [(i, i) for i in range(s33.shape[0])]
    else:
        print(f"  {dt}: cannot align (no time, unequal shapes) -- skipped"); continue

    for i, j in pairs:
        hcol = h[i] if h2d else h
        a_s, b_s = s33[i], s40[j]
        v3, v4 = np.isfinite(a_s), np.isfinite(b_s)
        if v3.any(): maxh33.append(np.nanmax(hcol[v3]))
        if v4.any(): maxh40.append(np.nanmax(hcol[v4]))
        both, only = v3 & v4, v4 & ~v3
        H.extend(hcol); V33.extend(v3); V40.extend(v4)
        if d33 is not None and d40 is not None:
            DSP.extend((b_s - a_s)[both]); DDR.extend(circ(d40[j], d33[i])[both])
        for src, lb, lo in [(res40, RES_b, RES_o), (cor40, COR_b, COR_o), (snr40, SNR_b, SNR_o)]:
            if src is not None:
                lb.extend(src[j][both]); lo.extend(src[j][only])
        if cnr40 is not None:
            CNR_o.extend(cnr40[j][only])

H = np.array(H); V33 = np.array(V33, bool); V40 = np.array(V40, bool)
DSP, DDR = np.array(DSP), np.array(DDR)
nboth = int((V33 & V40).sum()); nonly40 = int((V40 & ~V33).sum()); nonly33 = int((V33 & ~V40).sum())

# ----------------------------------------------------------------------------- findings
def med(x): x = np.asarray(x); x = x[np.isfinite(x)]; return np.median(x) if x.size else np.nan
print("\n" + "="*58)
print("DIFFERENCES: -33 dB (strict) vs -40 dB (loose) CNR cutoff")
print("="*58)
print(f"total gates examined: {len(V33)}")
print(f"  valid in -33: {V33.mean()*100:5.1f}%     valid in -40: {V40.mean()*100:5.1f}%")
print(f"  both: {nboth}   only -40 (rescued): {nonly40}   only -33: {nonly33}")
print(f"  -> -40 keeps {(V40.sum()-V33.sum())/max(V33.sum(),1)*100:+.1f}% more valid gates")
if DSP.size:
    print(f"\nwhere BOTH valid (same gate, both cutoffs):")
    print(f"  speed diff (-40 minus -33): median {med(DSP):+.3f}  MAD {med(np.abs(DSP-med(DSP))):.3f} m/s")
    print(f"  dir |diff|: median {med(DDR):.2f} deg   (large => loose cutoff perturbs the fit)")
print(f"\nfit quality (-40 product) at rescued vs solid gates:")
for nm, b_, o_ in [("residual", RES_b, RES_o), ("correlation", COR_b, COR_o), ("mean_snr", SNR_b, SNR_o)]:
    if b_ and o_:
        print(f"  {nm:11s} both {med(b_):+.3f}   only-40 {med(o_):+.3f}")
if CNR_o:
    print(f"  rescued-gate CNR: median {med(CNR_o):.1f} dB  range [{np.nanmin(CNR_o):.1f}, {np.nanmax(CNR_o):.1f}]")
print(f"\nmax valid height per profile: -33 median {med(maxh33):.0f} m   -40 median {med(maxh40):.0f} m")
print("="*58 + "\n")

# ----------------------------------------------------------------------------- plots
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
edges = np.linspace(np.nanmin(H), np.nanmax(H), 31)
cen = 0.5*(edges[:-1]+edges[1:])
c33 = np.array([V33[(H>=edges[k])&(H<edges[k+1])].sum() for k in range(len(cen))])
c40 = np.array([V40[(H>=edges[k])&(H<edges[k+1])].sum() for k in range(len(cen))])
ax[0,0].plot(c33, cen, "o-", ms=3, label="-33 dB")
ax[0,0].plot(c40, cen, "o-", ms=3, label="-40 dB")
ax[0,0].fill_betweenx(cen, c33, c40, color="seagreen", alpha=0.2, label="rescued by -40")
ax[0,0].set_xlabel("valid gates"); ax[0,0].set_ylabel("height (m)")
ax[0,0].set_title("Coverage vs height"); ax[0,0].legend()

if DSP.size:
    ax[0,1].hist(DSP, bins=60, color="steelblue")
    ax[0,1].axvline(0, color="k", lw=0.8)
    ax[0,1].set_xlabel("speed diff -40 minus -33 (m/s), both valid")
    ax[0,1].set_title(f"Agreement where both valid (med {med(DSP):+.2f})")
else:
    ax[0,1].set_title("no wind_direction/both-valid data"); ax[0,1].axis("off")

if RES_b and RES_o:
    rng = (0, np.nanpercentile(np.concatenate([RES_b, RES_o]), 99))
    ax[1,0].hist(RES_b, bins=40, range=rng, density=True, alpha=0.6, label="both")
    ax[1,0].hist(RES_o, bins=40, range=rng, density=True, alpha=0.6, label="only -40 (rescued)")
    ax[1,0].set_xlabel("residual (m/s)"); ax[1,0].set_ylabel("density")
    ax[1,0].set_title("Fit quality of rescued gates"); ax[1,0].legend()
else:
    ax[1,0].set_title("no residual variable"); ax[1,0].axis("off")

mh = [x for x in [np.nanmax(H)] if np.isfinite(x)]
hb = np.linspace(0, np.nanmax(H), 30)
ax[1,1].hist(maxh33, bins=hb, alpha=0.6, label="-33 dB")
ax[1,1].hist(maxh40, bins=hb, alpha=0.6, label="-40 dB")
ax[1,1].set_xlabel("max valid height per profile (m)"); ax[1,1].set_ylabel("profiles")
ax[1,1].set_title("How high each cutoff reaches"); ax[1,1].legend()

plt.tight_layout(); plt.show()
print("done")

