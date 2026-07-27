# `utils/` — shared helpers

Common functions used across the analysis folders (`equation/`, `ML/`,
`radar/`, `wind_prof/`, `marshall/`, `one_hour/`). Import from here instead
of re-copying; the originals were duplicated in ~30 scripts.

| Module | Contents |
|---|---|
| `io.py` | NetCDF loading/decoding: `to_nan`, `get_var`, `mapr_epoch`, `lidar_epoch`, `to_agl`, `site_altitude`; `MAPR_FILL`/`VAD_FILL` |
| `winds.py` | Wind-vector math (met. convention): `uv`, `met_dir`, `circ`, `vector_dV` |
| `match.py` | Corrected profiler/lidar gate matching: `nearest_scan`, `match_profile` (nearest gate, **no interpolation**) |
| `stats.py` | `binned_median`, `day_split`, `mae`, `day_bootstrap_ols` (all grouped/blocked by day) |

## Conventions
- Fill values: MAPR `-999`, Windcube VAD `-9999`.
- Heights returned AGL (`to_agl` auto-detects MSL-labelled grids).
- Time is float epoch seconds.
- Matching uses **stored** variables, **nearest gate**, **no interpolation** — see the note in `match.py`.
- Splits and bootstraps are grouped/blocked **by day** to respect autocorrelation.
- Pure functions only: no plotting, no file writes.

## Importing (flat repo, no packaging)
Scripts live at varying depths, so add `utils/` to the path relative to the
script, then import:

```python
import os, sys
HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "utils"))   # depth: equation/eqn/ -> repo root
from io import to_nan, lidar_epoch, get_var, to_agl, site_altitude, VAD_FILL, MAPR_FILL
from winds import uv, met_dir, vector_dV
from match import nearest_scan, match_profile
from stats import binned_median, day_split, mae, day_bootstrap_ols
```

Use `".."` (one level) for folders like `radar/` or `wind_prof/`, and
`"..", ".."` (two levels) for `equation/eqn/`, `equation/eqn_marshall/`.

> Note: `io` shadows Python's standard-library `io` module when imported
> this way. That's fine as long as a script doesn't also need the stdlib
> `io`; if it does, import as `import io as ncio` from the path, or rename
> this module to `ncio.py`.

