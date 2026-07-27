"""
match.py -- profiler/lidar gate matching for the SUPER 2026 intercomparison.

Provides the CORRECTED matching strategy: for each profiler profile, find
the nearest lidar scan in time (within a tolerance), then for each profiler
range gate take the single nearest lidar gate in height (within a
tolerance). Matched winds come from each product's STORED variables.

Why "no interpolation" (read this before changing it)
-----------------------------------------------------
An earlier version interpolated lidar u/v onto the profiler height grid.
Because per-scan VAD profiles are gappy, np.interp bridged across NaN gaps
and fabricated lidar values, and deriving speed from consensus-averaged
components compounded the error. Together these produced a spurious ~3 m/s
proportional speed deficit at Marshall that vanished once matching used
stored variables + nearest-gate + no interpolation. Do not reintroduce
interpolation across gaps here.

The functions are generic over which fields you pull (speed/direction, or
u/v, or quality metrics) -- you pass the per-scan lidar arrays you want and
get back the nearest-gate values aligned to the profiler gates.
"""

import numpy as np


def nearest_scan(profiler_t, lidar_t, i, time_tol_s):
    """Index of the lidar scan nearest profiler time step `i`, or None.

    Returns None if the nearest lidar scan is farther than `time_tol_s`.
    """
    j = int(np.argmin(np.abs(lidar_t - profiler_t[i])))
    if abs(lidar_t[j] - profiler_t[i]) > time_tol_s:
        return None
    return j


def match_profile(prof_h, prof_fields, lid_h, lid_fields, height_tol=25.0):
    """Match one profiler profile to one lidar scan by nearest gate (no interp).

    For each profiler gate with finite height and finite values in every
    requested profiler field, find the nearest lidar gate (also finite in
    every requested lidar field). Keep the pair only if the gates are within
    `height_tol` metres. No interpolation is performed.

    Parameters
    ----------
    prof_h : (Nz_p,) array
        Profiler gate heights (AGL) for this profile.
    prof_fields : dict[str, (Nz_p,) array]
        Profiler quantities to carry through (e.g. {"spd": ..., "dir": ...}).
    lid_h : (Nz_l,) array
        Lidar gate heights (AGL) for the matched scan.
    lid_fields : dict[str, (Nz_l,) array]
        Lidar quantities to carry through, keyed independently of prof_fields.
    height_tol : float
        Maximum profiler-lidar gate separation to accept (m).

    Returns
    -------
    list of dict
        One dict per matched gate, containing "height" plus every key in
        prof_fields and lid_fields (lidar keys prefixed if they collide is
        the caller's responsibility -- choose distinct names).
    """
    # lidar gates valid in height and in every requested lidar field
    lok = np.isfinite(lid_h)
    for arr in lid_fields.values():
        lok &= np.isfinite(arr)
    if lok.sum() < 1:
        return []
    order = np.argsort(lid_h[lok])
    vh = lid_h[lok][order]
    vfields = {k: arr[lok][order] for k, arr in lid_fields.items()}

    rows = []
    for g in range(len(prof_h)):
        hm = prof_h[g]
        if not np.isfinite(hm):
            continue
        if any(not np.isfinite(arr[g]) for arr in prof_fields.values()):
            continue
        k = int(np.argmin(np.abs(vh - hm)))
        if abs(vh[k] - hm) > height_tol:
            continue                       # nearest gate only -- NO interpolation
        row = {"height": hm}
        row.update({name: arr[g] for name, arr in prof_fields.items()})
        row.update({name: arr[k] for name, arr in vfields.items()})
        rows.append(row)
    return rows

