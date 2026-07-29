# SUPER 2026 — Radar Wind Profiler vs. Doppler Lidar Intercomparison

Analysis code from the **NCAR/EOL SUPER 2026** internship (Siddha Guha). The project
compares horizontal winds from a **449 MHz Modular Wind Profiler (MAPR)** against a
**Windcube Doppler lidar (VAD retrievals)** across two field campaigns:

- **M2HATS** — Tonopah, NV, summer 2023 (both instruments as 30-min consensus winds)
- **LOTOS-2025** — Marshall, CO, full year (5-min profiler consensus vs. per-scan VAD)

The central quantity is the **vector wind disagreement**
`|ΔV| = |V_profiler − V_lidar| = sqrt((u_p − u_l)² + (v_p − v_l)²)`,
capturing speed and direction error in one number. The goal: understand *when and why*
the two instruments disagree, and derive a validated, physically interpretable relation
predicting the disagreement from instrument-reported quality metrics.

## Key results

- **M2HATS:** disagreement is driven by the profiler's **Doppler spectral width**
  (turbulent broadening of the return spectrum). A held-out-validated relation,
  `|ΔV| ≈ 3.6·specWid + 0.07·wind + 0.1` (median), cuts prediction error ~22% vs. a
  constant baseline. Spectral width — not SNR or the u/v/w dispersions — carries the signal.
- **LOTOS-2025 (Marshall):** the spectral-width relation *transfers* (comparable slope),
  but no term clears held-out validation — the scatter is dominated by a
  **sampling-mismatch floor** (5-min consensus vs. near-instantaneous VAD), visible as an
  elevated intercept. The Marshall relation is a median QC guide, not a gate-level predictor.
- A data-handling bug (interpolating lidar winds across gappy VAD profiles / deriving
  speed from consensus-averaged components) once produced a spurious ~3 m/s proportional
  bias; correcting the matcher (stored variables, nearest-gate matching, no interpolation)
  removed it and reversed that conclusion.

## Methods pipeline

`screen → identify shape → forward-select → fit with uncertainty`

1. **Match** profiler and lidar gates in time and height; compute `|ΔV|`.
2. **Screen** candidates with **Spearman rank correlation** and a **random forest** with
   **permutation importance** (held-out); prune collinear clusters.
3. **Identify the functional form** by linearization (linear / semi-log / log-log).
4. **Forward selection**: keep a term only if it lowers **held-out MAE > 1%**, on a
   **day-grouped 80/20 split** (no autocorrelation leakage).
5. **Fit** with OLS (mean) and **median/quantile regression** (typical-disagreement QC
   relation); **95% CIs via day-block bootstrap**.

## Conventions

- **Python**: `numpy`, `pandas`, `scikit-learn`, `netCDF4`, `matplotlib`, `scipy`, `statsmodels`.
- Scripts **display plots with `plt.show()` and do not write figures to disk** — on a
  headless server, run under `ssh -X`; export via the plot window's save button.
- Data paths, fill values, and matching tolerances are **constants at the top of each
  script** — set them for your environment.
- Corrected pipelines match on **stored** wind variables with **nearest-gate** height
  matching and **no interpolation** (see `utils/match.py` and `equation/eqn_marshall/`).
- Shared helpers live in `utils/` — import from there rather than re-copying.

## Repository layout

### `utils/` — shared helpers
Common functions used across all analysis folders (extracted from ~30 scripts that had
duplicated them). Pure functions only — no plotting, no file writes.
- `io.py` — NetCDF loading/decoding: `to_nan`, `get_var`, `mapr_epoch`, `lidar_epoch`,
  `to_agl`, `site_altitude`, plus `MAPR_FILL` / `VAD_FILL`.
- `winds.py` — wind-vector math (meteorological convention): `uv`, `met_dir`, `circ`,
  `vector_dV`.
- `match.py` — corrected profiler/lidar gate matching: `nearest_scan`, `match_profile`
  (nearest gate, **no interpolation** — see the note in the file).
- `stats.py` — `binned_median`, `day_split`, `mae`, `day_bootstrap_ols` (grouped/blocked
  by day to respect autocorrelation).
- See `utils/README.md` for the import pattern (flat repo, path-relative).

### `equation/` — equation discovery (the core result)
- `vel_snr.py` — `|ΔV|` vs. SNR exploration (density, binned medians, linearization);
  establishes that SNR does not explain the disagreement.
- **`eqn/`** (M2HATS):
  - `test_specwid.py` — shape identification for `|ΔV|` vs. spectral width (hexbin +
    binned medians on linear / semi-log / log-log axes).
  - `fwd_sel.py` — forward selection of equation terms, judged on held-out days.
  - `final_fit.py` — final two-term fit (`specWid` + `wind_speed`): OLS + median
    regression with day-bootstrap confidence intervals.
  - `final2.py` — extended fit adding `specWid²` and a `specWid × wind` interaction;
    a held-out shootout selects the form.
- **`eqn_marshall/`** (LOTOS-2025):
  - `test_scatter.py` — clean-matcher speed/direction scatter vs. 1:1 (stored variables,
    nearest gate ≤ 25 m, no interpolation).
  - `dv_spearman.py` — wide Spearman screening matrix for `|ΔV|` on the clean matcher.
  - `forward_selection.py` — forward selection for the Marshall equation.
  - `specwid_fit.py` — specWid shape + fit for Marshall (median/OLS, bootstrap CIs).

### `ML/` — feature-importance screening
- `cnr33_40.py` — compares the lidar `vad_cnr33` vs. `vad_cnr40` QC products
  (clean-winds coverage vs. uncensored quality range).
- `fold5.py` — random-forest feature importance with day-grouped 5-fold CV and
  permutation importance.
- `split_8.py` — same screening on a single day-grouped 80/20 train/test split.
- `split_8_w_model.py` — 80/20 screening with an **ERA5 model reference** added
  (reference-error targets + a "with-model" correlation matrix).

### `wind_prof/` — wind-profiler pipeline
- `wind_prof_05.py` — the `winds.05` (5-min) profiler vs. per-scan VAD intercomparison
  and feature-importance pipeline (Spearman screen + outlier/in-band histograms + models).
- `rep.py` — reporting / summary helper.

### `radar/` — profiler-primary comparison
- `agreement.py` — profiler-vs-lidar agreement metrics.
- `outliers.py` — outlier (large-disagreement) histograms across quality parameters.
- `radar_con.py` — profiler consensus-wind ingest / handling.

### `marshall/` — Marshall (LOTOS) diagnostics
Histogram, linear-regression, and filtered/unfiltered variants used while diagnosing the
Marshall disagreement (`histogram.py`, `test_linear_reg.py`, `test_no_filter.py`, `test.py`).

### `one_hour/` — early agreement studies
Exploratory analyses at hourly, daily, and monthly resolution (profiler/lidar and reference
agreement, altitude-resolved comparisons, parameter-stratified histograms) that informed the
final pipeline. Retained for provenance.

## Next steps

The core contribution here is a reusable pipeline — screen candidate predictors,
identify the functional form, forward-select on held-out days, and fit with honest
uncertainty — not a single equation. Several directions build on it:

- **Apples-to-apples re-test at Marshall.** Re-run LOTOS with matched averaging
  (30-min profiler consensus vs. a 30-min lidar consensus product) to confirm the
  sampling-mismatch floor collapses and spectral width's predictive skill recovers.
- **Independent reference.** Bring in radiosondes or ERA5 as a third, independent
  estimate so the analysis no longer treats the lidar as ground truth, and
  disagreement can be attributed to a specific instrument.
- **More parameters and instruments.** Apply the same screen → select → validate
  workflow to additional quality metrics, other profiler products (e.g. RASS,
  Doppler moment files), and other instrument pairs (profiler vs. sonde, lidar vs.
  sonde) to build a broader library of validated QC relations.
- **More campaigns and datasets.** Extend the comparison to further field campaigns
  and sites to test how well the spectral-width relation generalizes across
  climates, seasons, and instrument configurations.
- **Mechanism of broadening.** Stratify the disagreement by hour and height and use
  the Doppler moment data to separate convective, nocturnal, and precipitation-driven
  spectral broadening.

## Attribution

Analysis code by Siddha Guha, NCAR/EOL SUPER 2026. Instruments and campaign data courtesy of the
NCAR Earth Observing Laboratory (In-situ Sensing Facility). Mentored by William Brown, Isabel Suhr, Mya Sears and Jacquie Witte

