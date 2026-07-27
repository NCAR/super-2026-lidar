"""
winds.py -- wind-vector math for the SUPER 2026 intercomparison.

Conversions between (speed, direction) and (u, v) components in
METEOROLOGICAL convention, plus angular differencing. Kept separate from
io.py because these are pure math with no netCDF dependency.

Meteorological convention
--------------------------
Direction is the direction the wind blows FROM, in degrees clockwise from
north. A wind FROM the north (0 deg) blows toward the south, so it has
u = 0, v = -|V|. Hence the minus signs below. uv() and met_dir() are
inverses of each other.
"""

import numpy as np


def uv(spd, direction):
    """(speed, direction) -> (u, v) components, meteorological convention.

    Parameters
    ----------
    spd : float or numpy.ndarray
        Wind speed (m/s).
    direction : float or numpy.ndarray
        Direction the wind is FROM (deg clockwise from north).

    Returns
    -------
    (u, v) : eastward and northward components (m/s).
    """
    r = np.radians(direction)
    return -spd * np.sin(r), -spd * np.cos(r)


def met_dir(u, v):
    """(u, v) components -> meteorological direction (deg FROM, 0-360)."""
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def circ(a, b):
    """Absolute angular difference between directions a and b, in [0, 180].

    Wraps correctly across 0/360 (e.g. circ(350, 10) == 20), so it is safe
    for comparing wind directions.
    """
    return np.abs(((a - b + 180) % 360) - 180)


def vector_dV(u1, v1, u2, v2):
    """Vector wind disagreement |dV| = |V1 - V2| between two (u, v) pairs.

    This is the central target of the intercomparison: it folds speed AND
    direction error into a single non-negative number (m/s).
    """
    return np.hypot(u1 - u2, v1 - v2)

