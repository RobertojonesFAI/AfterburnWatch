"""Burn severity: BAER classification of dNBR.

The index math lives in indices.py so it's shared with the debris-flow
identification features (identify.py). The ``compute_*`` names below are
kept as thin aliases so existing code and tests keep working.

dNBR can come from either source:
  * our own Sentinel-2 / Landsat pre/post pair (indices.dnbr), or
  * the authoritative dNBR shipped in a NASA RECOVER package (recover.py),
    which is the version federal stakeholders already trust. Use it to
    cross-check ours wherever both exist.

References
----------
- BAER breaks: standard Burned Area Emergency Response 4-class scheme.
- Key & Benson, Landscape Assessment (FIREMON) -- NBR/dNBR definition.
"""

from __future__ import annotations

import numpy as np

from afterburn_watch.indices import dnbr as _dnbr
from afterburn_watch.indices import nbr as _nbr

# Standard BAER burn-severity breaks (on dNBR * 1000 scale)
BAER_BREAKS = {
    "unburned_low": (-np.inf, 100),
    "low": (100, 270),
    "moderate": (270, 660),
    "high": (660, np.inf),
}

# Moderate + high: the classes that count toward USGS M1's "T" term
# (proportion of upslope area burned at moderate/high severity on steep
# slopes).
MODERATE_HIGH_MIN_DNBR = 270


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Alias of indices.nbr."""
    return _nbr(nir, swir2)


def compute_dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """dNBR = (NBR_pre - NBR_post) * 1000. Alias of indices.dnbr."""
    return _dnbr(nbr_pre, nbr_post)


def classify_baer(dnbr: np.ndarray) -> np.ndarray:
    """Classify a dNBR raster into standard BAER severity classes.

    Returns an integer array: 0=unburned/low, 1=low, 2=moderate, 3=high.
    NaN inputs are preserved as -1 (no data).
    """
    dnbr = np.asarray(dnbr, dtype="float32")
    out = np.full(dnbr.shape, -1, dtype="int8")
    valid = ~np.isnan(dnbr)
    out[valid & (dnbr < 100)] = 0
    out[valid & (dnbr >= 100) & (dnbr < 270)] = 1
    out[valid & (dnbr >= 270) & (dnbr < 660)] = 2
    out[valid & (dnbr >= 660)] = 3
    return out
