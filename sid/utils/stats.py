"""
stats.py -- binning, leakage-safe splitting, and day-block bootstrap for
the SUPER 2026 intercomparison.

The recurring statistical machinery: binned median + IQR for shape
inspection, a train/test split grouped BY DAY (so autocorrelated
neighbouring gates never straddle the split), and a day-block bootstrap
for honest coefficient confidence intervals.

Why group / block by day
-------------------------
Adjacent gates and adjacent times are highly autocorrelated. A random
row-wise split leaks near-duplicate samples across train and test and
inflates skill; a row-wise bootstrap shrinks confidence intervals
dishonestly. Both operations therefore work on whole days as the unit.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression


def binned_median(x, y, edges, min_count=30):
    """Bin y by x and return (centres, median, q25, q75) per bin.

    Bins with fewer than `min_count` samples return NaN (so sparse tails
    don't masquerade as signal).

    Returns
    -------
    cen, med, q1, q3 : numpy.ndarray
        Bin centres and the median / 25th / 75th percentile of y in each bin.
    """
    cen = 0.5 * (edges[:-1] + edges[1:])
    med, q1, q3 = [], [], []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() < min_count:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
            continue
        med.append(np.median(y[s]))
        q1.append(np.percentile(y[s], 25))
        q3.append(np.percentile(y[s], 75))
    return cen, np.array(med), np.array(q1), np.array(q3)


def day_split(df, day_col="day", test_frac=0.2, random_state=0):
    """Single train/test split grouped by day (no day on both sides).

    Returns
    -------
    (train_df, test_df) : pandas.DataFrame
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    tr, te = next(gss.split(df, groups=df[day_col]))
    return df.iloc[tr], df.iloc[te]


def mae(actual, pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def day_bootstrap_ols(df, features, target, n_boot=300, day_col="day", random_state=0):
    """Day-block bootstrap of OLS coefficients; returns 95% CIs.

    Resamples whole days (with replacement), refits OLS each time, and
    returns the 2.5/97.5 percentile interval for each coefficient and the
    intercept. Resampling by day (not by row) respects autocorrelation so
    the intervals are not dishonestly tight.

    Returns
    -------
    dict
        {feature_name: (lo, hi), ..., "intercept": (lo, hi)}
    """
    days = df[day_col].unique()
    grouped = {d: g for d, g in df.groupby(day_col)}
    rng = np.random.default_rng(random_state)
    boots = []
    import pandas as pd
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        bs = pd.concat([grouped[p] for p in pick], ignore_index=True)
        r = LinearRegression().fit(bs[features], bs[target])
        boots.append(list(r.coef_) + [r.intercept_])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    out = {f: (lo[k], hi[k]) for k, f in enumerate(features)}
    out["intercept"] = (lo[-1], hi[-1])
    return out

