"""Spectral indices, independent of sensor.

Inputs are surface-reflectance arrays already scaled to 0-1 (see
sensors.scale_offset) and addressed by band *role*, never by band number, so
the same code serves Sentinel-2 and Landsat.

    NBR   = (NIR - SWIR2) / (NIR + SWIR2)
    dNBR  = (NBR_pre - NBR_post) * 1000
    NDVI  = (NIR - Red) / (NIR + Red)
    dNDVI = NDVI_pre - NDVI_post
    RdNBR = dNBR / sqrt(|NBR_pre|)          (Miller & Thode 2007)

NBR/dNBR follow Keith Weber's Sept 18 email and Key & Benson (the paper in
the shared articles folder). dNBR is kept on the conventional x1000 scale so
the BAER breaks in severity.py apply unchanged.
"""

from __future__ import annotations

import numpy as np


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom != 0, (a - b) / denom, np.nan)
    return out.astype("float32")


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio, in [-1, 1]."""
    return _normalized_difference(nir, swir2)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index, in [-1, 1]."""
    return _normalized_difference(nir, red)


def dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Differenced NBR (pre - post), x1000 by convention."""
    return ((np.asarray(nbr_pre) - np.asarray(nbr_post)) * scale).astype("float32")


def dndvi(ndvi_pre: np.ndarray, ndvi_post: np.ndarray) -> np.ndarray:
    """Differenced NDVI (pre - post). Positive = vegetation lost."""
    return (np.asarray(ndvi_pre) - np.asarray(ndvi_post)).astype("float32")


def rdnbr(dnbr_x1000: np.ndarray, nbr_pre: np.ndarray, min_abs_pre: float = 0.001) -> np.ndarray:
    """Relative dNBR (Miller & Thode 2007).

    Divides out the pre-fire vegetation signal, so severity compares better
    between sparse and dense stands. ``nbr_pre`` is unscaled (-1 to 1); pixels
    with ``|nbr_pre| < min_abs_pre`` return NaN to avoid dividing by ~0.
    """
    nbr_pre = np.asarray(nbr_pre, dtype="float32")
    denom = np.sqrt(np.abs(nbr_pre))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(nbr_pre) >= min_abs_pre, np.asarray(dnbr_x1000) / denom, np.nan)
    return out.astype("float32")


def reflectance(dn: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Convert stored DN to surface reflectance: ``dn * scale + offset``."""
    return (np.asarray(dn, dtype="float32") * scale + offset).astype("float32")
