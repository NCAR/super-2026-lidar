# M2HATS/LOTOS-2025 — VAD Lidar vs. Model Wind Intercomparison

Analysis code from Adrian's work at NCAR/UCAR comparing **Windcube Doppler lidar
VAD wind retrievals** against two numerical weather prediction models —
**HRRR** and **ERA5** — across two field campaigns:

- **M2HATS** — Tonopah, NV (ISS1), summer 2023, 30-min VAD consensus winds
- **LOTOS-2025** — Marshall, CO (ISS2), VAD winds at CNR35/CNR40 (with injected error) quality tiers

The guiding question is *where and why VAD lidar winds disagree with model wind profiles*, and whether that disagreement can be explained by predictors
such as: height, time of day, model wind speed, model type, wind shear, and lapse rate. The centerpiece result — a random-forest error model with permutation importance and partial dependence, compared side-by-side for HRRR vs. ERA5 — lives in **`project-focus/poster/`** and is what the summer student conference poster is built around (for M2HATS more specifically).

## Poster focus (`project-focus/poster/`)

This is the core deliverable of the project: a random-forest pipeline that predicts **absolute wind speed error** and **absolute wind direction error** (VAD vs. model) from six features — `model` (HRRR/ERA5 flag), `hour` (UTC), `height` (m AGL), `model_ws`, local wind-speed `shear`, and temperature `lapse` rate — then asks *which of those features actually drive the error*, separately for each model. One note is that "error" more accurately refers to "disagreement" between the windcube and the numerical models.

- **`make-pdps-final.py`** — the main poster script. Matches VAD and model (HRRR + ERA5) profiles at M2HATS (100–2000 m AGL, 25 m height tolerance, 15-min time tolerance), engineers the shear/lapse/hour features, then:
  1. Fits a `RandomForestRegressor` (`n_estimators=300`, `max_depth=12`,
     `min_samples_leaf=30`) separately for the speed-error and
     direction-error targets, using 5 repeated **day-grouped 75/25
     train/test splits** (`GroupShuffleSplit`, grouped by day-of-epoch so
     no day's data crosses the split) to get a stable R² and permutation
     importance rather than trusting a single split.
  2. Renders a **table figure** per target (`rf_z_table_*.png`) — features
     ranked by permutation importance, with impurity importance and R²
     (train/test) alongside.
  3. Renders **partial dependence plots** two ways: pooled across both
     models (`rf_z_pdp_*.png`), and **overlaid HRRR vs. ERA5**
     (`rf_z_pdp_bymodel_*.png`, Okabe-Ito colors — green for HRRR, pink for
     ERA5) by pinning the model flag and sweeping each feature across its
     5th–95th percentile range. The by-model PDPs are the poster figures.
- **`time-series-M2HATS.py`** / **`time-series-all.py`** — diurnal-cycle
  companion figures: mean *signed* difference (VAD − model) by UTC hour,
  stacked speed/direction panels. `time-series-M2HATS.py` is HRRR vs. ERA5
  within M2HATS only; `time-series-all.py` extends the same comparison to
  LOTOS-2025 alongside M2HATS (4 lines: 2 campaigns × 2 models).
- **`print-counts.py`** — a data-availability check, not a comparison: tallies
  matched-point counts per UTC hour for all four campaign/model combinations,
  used to sanity-check coverage and inform CNR/QC threshold choices before
  committing to a mask.

## Repository layout

### `m2hats_test/` top level — early M2HATS-only exploration
Single-campaign VAD-vs-HRRR scripts developed while first building out the
matching/plotting logic, later generalized into the `project-focus/` pipeline:
histograms of matched-point differences and outliers (`make-histogram.py`,
`histogram-by-PL.py`), diurnal and daily-mean difference time series
(`plot-by-TOD.py`, `plot-daily-mean-diff.py`), u/v-component and
speed-vs-direction scatterplots (`plot-by-uv.py`, `plot-by-uv-dir.py`,
`wspd-vs-diff.py`), and SNR-threshold sensitivity checks
(`multi-snr-cutoffs.py`, `snr-with-error.py`, `snr-filtered-comparison.py`).

### `machine-learning/` — first ML pass
`ml-script.py` — an early, more exploratory random-forest/gradient-boosting
script (also touches `RandomForestClassifier` and ROC/AUC) before the
pipeline settled on the regressor-only, day-grouped-split approach used in
`project-focus/`. `snr-resid-histogram.py` looks at residuals binned by SNR.

### `sounding-comp/` — VAD vs. radiosonde
A parallel comparison using radiosondes as reference instead of a model:
scatter plots, outlier histograms by pressure band, and daily-mean/diurnal
difference time series (`comparison.py`, `histogram.py`,
`daily-mean-diff.py`, `diurnal-avg-diff.py`). Same VAD matching conventions
as the model comparisons, swapped onto the sonde archive.

### `project-focus/M2HATS/` and `project-focus/LOTOS/` — per-campaign pipeline
The generalized VAD-vs-HRRR-vs-ERA5 scripts for each campaign individually,
run at several VAD quality tiers (`vad_consensus`, `vad_cnr33`, `vad_cnr35`,
`vad_cnr40`): diurnal mean-difference plots by model (`plot-by-TOD-*.py`),
combined summary statistics (`combined-stats-compare-LOTOS*.py`), and
per-campaign random-forest + PDP scripts (`ml_cnr35_with-pdp.py`,
`ml_vad_pdp_modified1.py`, `rf_z_pdps-by-model.py`) — earlier iterations of
what was finalized as `project-focus/poster/make-pdps-final.py`.

### `project-focus/poster/` — **poster figures (see above)**

### `marshall_test/` — **early Marshall/LOTOS scatter and histogram checks**

A separate top-level sibling to m2hats_test/, and much smaller: just two scripts, both doing a first-look VAD-vs-HRRR comparison at Marshall, CO (CNR35 tier) over the LOTOS-2025 period. Note that there are more scripts relevant to Marshall field data present in the LOTOS subfolder of the m2hats_test subfolder as described above.

- compare.py — scatter plots of matched HRRR vs. VAD wind speed and direction (1:1 reference), annotated with point count, mean absolute difference, standard deviation, and a linear fit.
- make-histogram.py — histograms of the VAD − HRRR speed and direction differences, with the same summary stats.

These use the identical matching conventions as the rest of the repo (100–2000 m AGL, 25 m height tolerance, 15-min time tolerance, direction only above 2 m/s) but are earlier/looser drafts — commented-out geopotential-height code, plt.show() instead of saving figures, no day-grouped splitting (there's no ML here yet). The substantive Marshall analysis is not actually here — despite the folder name, the developed LOTOS-2025 (Marshall) pipeline — diurnal comparisons, combined stats, and the random-forest/PDP scripts — lives under m2hats_test/project-focus/LOTOS/ (see above), which picks up where these two scripts left off.

## Conventions

- **Python**: `netCDF4`, `numpy`, `matplotlib`, `scikit-learn`
  (`RandomForestRegressor`, `permutation_importance`, `GroupShuffleSplit`).
- **Height matching**: model AGL height vs. nearest VAD height level,
  **25 m tolerance**, restricted to **100–2000 m AGL**.
- **Time matching**: nearest VAD profile in time, rejected if more than
  **15 minutes (900 s)** away.
- **Direction error** is only computed when both VAD and model wind speed
  exceed **2 m/s** (direction is unreliable at low speed), and is wrapped
  into `[-180, 180]` before taking an absolute value.
- **Train/test splits are grouped by day** (`day = epoch_time // 86400`) so
  a day's points never appear on both sides of a split — this avoids
  leaking within-day autocorrelation into the reported R²/importances.
- **ERA5 heights** are derived from geopotential (`z / 9.80665 − site_alt`);
  M2HATS's VAD `alt` variable is masked/unusable, so `site_alt = 1739.0 m
  MSL` is hardcoded from ERA5 surface geopotential at ISS1. LOTOS reads its
  site elevation directly from the VAD file's `alt` variable.
- **Palette**: Okabe-Ito colorblind-safe colors throughout — `#0072B2`
  (blue) and `#D55E00` (orange) distinguish campaigns/models in the
  time-series figures; `#009E73` (green, HRRR) and `#CC79A7` (pink, ERA5)
  in the by-model PDPs.
- Figures save to disk at 300 dpi (`plt.savefig(..., dpi=300,
  bbox_inches='tight')`) rather than displaying inline, following the
  filename conventions established across each script.

## Next steps

- **Extend the by-model PDP comparison to LOTOS-2025** using the same
  `make-pdps-final.py` structure, to see whether the M2HATS feature
  ranking (and the HRRR-vs-ERA5 contrast) transfers to a different site
  and VAD quality tier.
- **Reconcile the CNR/QC tier choice,** settle on a single consensus/CNR mask per campaign rather than
  carrying multiple parallel VAD datasets (`vad_consensus`,
  `vad_cnr33`, `vad_cnr35`, `vad_cnr40`) through the pipeline, especially for LOTOS.
- **Bring the radiosonde comparison (`sounding-comp/`) into the same
  feature-importance framework** so VAD disagreement can be checked
  against an independent reference, not just against models.
- **Consolidate the repeated matching logic** (`read_vad`, the
  time/height matching loop) that's currently copy-pasted across nearly
  every script here into a shared `utils/`-style module, mirroring the
  approach in `sid/utils/`.

## Attribution

Analysis code by Adrian, NCAR/UCAR. Instruments and campaign data courtesy
of the NCAR Earth Observing Laboratory (In-situ Sensing Facility). HRRR and
ERA5 model data via UCAR server archives.
