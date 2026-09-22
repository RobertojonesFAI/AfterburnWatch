"""Cloud, shadow and smoke masking -- the "Cloud Masking" note on the whiteboard.

Keith Weber's Sept 18 briefing: imagery must be cloud-free *and* smoke-free,
and smoke from other, unrelated fires nearby can contaminate post-fire
imagery for months. Two consequences for this module:

1. Mask per scene with each sensor's own QA product
   (Sentinel-2 SCL, Landsat QA_PIXEL), plus an optional aerosol/smoke
   test, because neither QA product has a dedicated smoke class.
2. Don't depend on one clean scene. ``masked_median_composite`` combines
   several masked scenes, so a pixel only needs to be clear in *some* of
   them. More revisits help here, which is one argument for using both
   sensors (or the harmonized HLS product).

All masks follow the same convention: ``True`` = *valid* (keep),
``False`` = masked out.
"""

from __future__ import annotations

import warnings

import numpy as np

# --- Sentinel-2 L2A Scene Classification Layer (SCL) ------------------------
SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_or_cast_shadow",
    3: "cloud_shadow",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_ice",
}

# Class 2 (dark area / cast shadow) is deliberately NOT masked by default:
# fresh burn scars are dark in every band and are often put in class 2 (and
# sometimes class 3). Masking them would erase the very burned areas we're
# trying to measure. Check a few scenes over the Wapiti burn before
# tightening this.
SCL_INVALID_DEFAULT = frozenset({0, 1, 3, 8, 9, 10})


def scl_valid_mask(scl: np.ndarray, invalid_classes=SCL_INVALID_DEFAULT) -> np.ndarray:
    """Valid-pixel mask from a Sentinel-2 SCL array."""
    return ~np.isin(scl, list(invalid_classes))


def aot_valid_mask(aot: np.ndarray, max_aot: float, scale: float = 0.001) -> np.ndarray:
    """Mask pixels whose aerosol optical thickness suggests heavy smoke.

    Sentinel-2 L2A AOT is stored as DN with scale 0.001. ``max_aot`` is in
    physical AOT units (for example 0.5). No threshold is universally right,
    so tune it on real smoke-affected scenes.
    """
    return (aot.astype("float32") * scale) <= max_aot


# --- Landsat Collection 2 QA_PIXEL bit flags --------------------------------
QA_PIXEL_BITS = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "clear": 6,
    "water": 7,
}
QA_PIXEL_INVALID_DEFAULT = ("fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow")


def qa_pixel_valid_mask(qa: np.ndarray, invalid_flags=QA_PIXEL_INVALID_DEFAULT) -> np.ndarray:
    """Valid-pixel mask from a Landsat C2 QA_PIXEL array (bit flags)."""
    qa = qa.astype("uint16")
    bad = np.zeros(qa.shape, dtype=bool)
    for flag in invalid_flags:
        bad |= ((qa >> QA_PIXEL_BITS[flag]) & 1).astype(bool)
    return ~bad


def qa_aerosol_valid_mask(qa_aerosol: np.ndarray, max_level: int = 2) -> np.ndarray:
    """Mask Landsat pixels whose SR_QA_AEROSOL level exceeds ``max_level``.

    Bits 6-7 hold the aerosol level: 0 climatology, 1 low, 2 medium,
    3 high. The default drops "high" only, which is where smoke plumes
    land. Check these bit positions against the USGS Landsat 8-9 C2 L2
    Science Product Guide before relying on them.
    """
    level = (qa_aerosol.astype("uint8") >> 6) & 0b11
    return level <= max_level


# --- Multi-scene compositing ------------------------------------------------
def masked_median_composite(stack: np.ndarray, valid: np.ndarray, min_valid: int = 1):
    """Per-pixel median across time, using only valid observations.

    Parameters
    ----------
    stack : array (time, rows, cols) of one band or index.
    valid : bool array, same shape, True = usable observation.
    min_valid : minimum number of valid observations for a pixel to get a
        value. Pixels with fewer come back NaN.

    Returns
    -------
    composite : (rows, cols) float32, NaN where not enough clear views.
    n_valid : (rows, cols) count of valid observations used.
    """
    if stack.shape != valid.shape:
        raise ValueError(f"stack {stack.shape} and valid {valid.shape} differ")
    data = np.where(valid, stack.astype("float32"), np.nan)
    n_valid = valid.sum(axis=0)
    with warnings.catch_warnings():
        # all-NaN pixels (never clear) are expected; they stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(data, axis=0)
    composite[n_valid < min_valid] = np.nan
    return composite.astype("float32"), n_valid
