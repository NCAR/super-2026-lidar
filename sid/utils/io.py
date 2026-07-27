"""
io.py -- NetCDF loading and decoding helpers for the SUPER 2026 wind
profiler / lidar intercomparison.

These are the functions that appear in nearly every analysis script:
reading a variable and converting fill values to NaN, decoding the
non-CF time bases, resolving variable-name aliases, and putting heights
on AGL. Import these instead of re-copying them.

Conventions
-----------
- Fill values: MAPR profiler uses -999, Windcube VAD uses -9999.
- Time: MAPR is base_time + time; Windcube VAD is base_time + time_offset
  (with fallbacks). Both return float epoch seconds.
- Heights are returned on AGL. Some MAPR files label height as MSL but
  store AGL-valued numbers; to_agl() auto-detects and only subtracts the
  site altitude when the values are actually MSL.

All functions are pure (no plotting, no file writes).
"""

import numpy as np
import netCDF4 as nc

MAPR_FILL = -999.0
VAD_FILL = -9999.0


def to_nan(var, fill):
    """Read a netCDF variable and return a float array with fill -> NaN.

    Handles masked arrays (filling masked entries with NaN) and replaces
    the sentinel `fill` value with NaN.

    Parameters
    ----------
    var : netCDF4.Variable
        The variable to read (e.g. ds["wspd"]).
    fill : float
        Sentinel fill value to convert to NaN (MAPR_FILL or VAD_FILL).

    Returns
    -------
    numpy.ndarray of float
    """
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    a = np.asarray(a, float)
    a[a == fill] = np.nan
    return a


def get_var(ds, names, fill):
    """Return the first present variable among `names`, decoded via to_nan.

    Lets a script tolerate naming differences between products
    (e.g. lidar speed as "wind_speed" vs "wspd" vs "speed").

    Parameters
    ----------
    ds : netCDF4.Dataset
    names : str or list of str
        Candidate variable name(s), tried in order.
    fill : float
        Fill value passed to to_nan.

    Returns
    -------
    numpy.ndarray or None
        Decoded array, or None if no candidate name is present.
    """
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in ds.variables:
            return to_nan(ds[n], fill)
    return None


def mapr_epoch(ds):
    """Epoch seconds for a MAPR profiler file: base_time + time."""
    return float(np.asarray(ds["base_time"][...])) + np.asarray(ds["time"][:], float)


def lidar_epoch(ds):
    """Epoch seconds for a Windcube VAD file.

    Prefers base_time + time_offset (ARM convention). Falls back to a
    "time" variable, treating it as absolute epoch if it already looks
    like epoch seconds, otherwise adding base_time when available.

    Returns
    -------
    numpy.ndarray of float, or None if no usable time variable exists.
    """
    v = ds.variables
    if "base_time" in v and "time_offset" in v:
        return float(np.asarray(v["base_time"][...])) + np.asarray(v["time_offset"][:], float)
    if "time" in v:
        t = np.asarray(v["time"][:], float)
        if np.nanmax(t) > 1e8:          # already absolute epoch seconds
            return t
        if "base_time" in v:
            return float(np.asarray(v["base_time"][...])) + t
        return t
    return None


def to_agl(height, site_alt, margin=200.0):
    """Return heights on AGL, auto-detecting an MSL-labelled grid.

    Some MAPR files store height as MSL. If the minimum height sits well
    above the site altitude (min > site_alt - margin), the grid is treated
    as MSL and `site_alt` is subtracted; otherwise it is assumed already
    AGL and returned unchanged.

    Parameters
    ----------
    height : numpy.ndarray
        Height array (1-D or 2-D), fill values already converted to NaN.
    site_alt : float
        Site altitude in metres MSL. If NaN, pass a known fallback in.
    margin : float
        Tolerance (m) for the MSL-vs-AGL decision.

    Returns
    -------
    numpy.ndarray, same shape as `height`.
    """
    if np.nanmin(height) > site_alt - margin:
        return height - site_alt
    return height


def site_altitude(ds, fallback):
    """Read the file's `alt`, substituting `fallback` if it is NaN/missing.

    The LOTOS MAPR files had a NaN station altitude; passing a known
    fallback (e.g. 1742 m for Marshall, 1641 m for M2HATS/Tonopah) keeps
    the AGL conversion and matching from silently breaking.
    """
    try:
        alt = float(np.asarray(ds["alt"][...]))
    except (IndexError, KeyError):
        alt = np.nan
    return fallback if not np.isfinite(alt) else alt


def open_dataset(path):
    """Thin wrapper around netCDF4.Dataset for symmetry / future hooks."""
    return nc.Dataset(path)

