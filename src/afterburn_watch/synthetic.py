"""SYNTHETIC data for tests and the offline demo -- not real observations.

Nothing here is measured. It exists so the whole loop can run end-to-end
without network access or real imagery, and so tests can check that each
stage does what it claims (for example, that recalibration moves the
coefficients toward the truth that generated the data).

The spectral signatures are simple, plausible placeholders:
  * burned hillslope: dark in red/NIR, moderately bright in SWIR2
  * fresh debris-flow deposit along a channel: brighter, especially SWIR2
  * confounders: rain-wetted soil (everything darker) and road grading
    (everything brighter, but NBR barely moves)
Replace every number here with real scenes before drawing conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from afterburn_watch.predict import STALEY2017_M1

# A made-up "regional truth" that differs from the published Staley 2017
# coefficients. It stands in for "Idaho behaves differently" in the demo.
SYNTHETIC_REGIONAL_TRUTH = (-4.5, 0.25, 1.10, 0.20)


@dataclass
class SyntheticFire:
    fire_id: str
    region: str
    basins: pd.DataFrame  # basin_id, lon, lat, T, F, S
    basin_labels: np.ndarray  # int raster, basin ids (0 = none)
    channel: np.ndarray  # bool raster, drainage network
    pixel_to_lonlat: Callable


def make_fire(
    rng: np.random.Generator,
    fire_id: str,
    region: str,
    first_basin_id: int,
    n_by: int = 5,
    n_bx: int = 8,
    block: int = 12,
    lon0: float = -115.5,
    lat0: float = 44.3,
) -> SyntheticFire:
    """A grid of square basins, each with one north-south channel."""
    rows, cols = n_by * block, n_bx * block
    labels = np.zeros((rows, cols), dtype=int)
    channel = np.zeros((rows, cols), dtype=bool)
    recs = []
    px_deg = 0.0003  # ~30 m
    bid = first_basin_id
    for by in range(n_by):
        for bx in range(n_bx):
            r0, c0 = by * block, bx * block
            labels[r0 : r0 + block, c0 : c0 + block] = bid
            cc = c0 + block // 2 + int(rng.integers(-2, 3))
            channel[r0 + 1 : r0 + block - 1, cc] = True
            recs.append(
                {
                    "basin_id": bid,
                    "lon": lon0 + (c0 + block / 2) * px_deg,
                    "lat": lat0 - (r0 + block / 2) * px_deg,
                    "T": float(rng.uniform(0.0, 0.8)),
                    "F": float(rng.uniform(0.1, 0.9)),
                    "S": float(rng.uniform(0.1, 0.4)),
                }
            )
            bid += 1

    def pixel_to_lonlat(r, c):
        return lon0 + (np.asarray(c) + 0.5) * px_deg, lat0 - (np.asarray(r) + 0.5) * px_deg

    return SyntheticFire(fire_id, region, pd.DataFrame(recs), labels, channel, pixel_to_lonlat)


def storm_rainfall(rng, basins: pd.DataFrame, mean_R: float, spread: float = 0.35) -> pd.Series:
    """Per-basin rainfall (mm) for one storm: convective storms are patchy,
    so each basin gets a lognormal draw around the storm mean."""
    r = mean_R * rng.lognormal(0.0, spread, len(basins))
    return pd.Series(r, index=basins["basin_id"].to_numpy())


def simulate_responses(rng, basins: pd.DataFrame, coefs, R) -> np.ndarray:
    """Draw 0/1 debris-flow outcomes for one storm from M1 with ``coefs``.

    ``R`` is one number or a per-basin Series (see storm_rainfall)."""
    B, Ct, Cf, Cs = coefs
    if isinstance(R, pd.Series):
        R = R.reindex(basins["basin_id"]).to_numpy()
    x = B + R * (Ct * basins["T"] + Cf * basins["F"] + Cs * basins["S"])
    p = 1 / (1 + np.exp(-x.to_numpy()))
    return (rng.random(len(p)) < p).astype(int)


def render_scenes(
    rng: np.random.Generator,
    fire: SyntheticFire,
    responses: np.ndarray,
    sensor: str = "sentinel2",
    cloud_fraction: float = 0.1,
    confounder_rate: float = 0.3,
):
    """Before/after reflectance for one storm, plus cloud mask and truth.

    Returns ``(before, after, valid, flow_truth)``: band dicts, the
    valid-pixel mask (False under cloud) and the true deposit pixels.
    """
    shape = fire.basin_labels.shape
    noise = 0.010 if sensor == "sentinel2" else 0.015
    base = {"red": 0.09, "nir": 0.16, "swir2": 0.20}
    before = {b: (v + rng.normal(0, noise, shape)).astype("float32") for b, v in base.items()}
    after = {b: (before[b] + rng.normal(0, noise * 0.7, shape)).astype("float32") for b in base}

    flow = np.zeros(shape, dtype=bool)
    deposit = {"red": 0.05, "nir": 0.015, "swir2": 0.09 if sensor == "sentinel2" else 0.08}
    ids = fire.basins["basin_id"].to_numpy()
    for bid, resp in zip(ids, responses):
        inside = fire.basin_labels == bid
        ch = np.argwhere(inside & fire.channel)
        if resp == 1:
            n = max(4, int(len(ch) * rng.uniform(0.6, 1.0)))
            start = int(rng.integers(0, max(1, len(ch) - n + 1)))
            seg = ch[start : start + n]
            flow[seg[:, 0], seg[:, 1]] = True
        elif rng.random() < confounder_rate:
            # a non-debris-flow change somewhere in the basin
            rr, cc = np.argwhere(inside)[int(rng.integers(0, inside.sum()))]
            patch = (slice(max(rr - 1, 0), rr + 2), slice(max(cc - 1, 0), cc + 2))
            delta = -0.03 if rng.random() < 0.5 else 0.05  # wet soil vs grading
            for b in after:
                after[b][patch] += delta
    for b, d in deposit.items():
        after[b][flow] += d + rng.normal(0, 0.01, flow.sum())

    valid = np.ones(shape, dtype=bool)
    if cloud_fraction > 0:
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        target = cloud_fraction * valid.size
        while (~valid).sum() < target:
            cy, cx = rng.integers(0, shape[0]), rng.integers(0, shape[1])
            ry, rx = rng.integers(3, 10), rng.integers(4, 14)
            valid &= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 > 1
        for b in after:
            after[b][~valid] = 0.5  # bright cloud tops
    return before, after, valid, flow


def seed_inventory(rng, n_fires: int = 10, basins_per_fire: int = 60, coefs=None) -> pd.DataFrame:
    """Stand-in for the USGS inventory: verified observations generated
    with the published Staley 2017 (15-min) coefficients, so the published
    model fits it."""
    coefs = coefs or STALEY2017_M1[15]
    rows = []
    for f in range(n_fires):
        T = rng.uniform(0, 0.8, basins_per_fire)
        F = rng.uniform(0.1, 0.9, basins_per_fire)
        S = rng.uniform(0.1, 0.4, basins_per_fire)
        R = rng.uniform(2, 20, basins_per_fire)
        B, Ct, Cf, Cs = coefs
        p = 1 / (1 + np.exp(-(B + R * (Ct * T + Cf * F + Cs * S))))
        y = (rng.random(basins_per_fire) < p).astype(int)
        for i in range(basins_per_fire):
            rows.append(
                {
                    "obs_id": f"inv-{f}-{i}",
                    "fire_id": f"INV-FIRE-{f:02d}",
                    "region": "inventory",
                    "response": int(y[i]),
                    "source": "usgs_inventory",
                    "confidence": 1.0,
                    "verified": True,
                    "lon": -112.0 + f * 0.5 + i * 0.001,
                    "lat": 36.0 + f * 0.3,
                    "event_date": f"20{15 + f % 9}-08-01",
                    "T": T[i],
                    "F": F[i],
                    "S": S[i],
                    "R": R[i],
                }
            )
    return pd.DataFrame(rows)


def training_pixels(rng, fire: SyntheticFire, stack: dict, responses: np.ndarray, neg_per_pos: float = 1.0):
    """Channel pixels labeled with their basin's response (as with a real
    basin-level inventory), negatives subsampled. Returns (rows, cols, y)."""
    resp_by_basin = dict(zip(fire.basins["basin_id"], responses))
    rr, cc = np.nonzero(fire.channel)
    y = np.array([resp_by_basin[b] for b in fire.basin_labels[rr, cc]])
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    n_neg = min(len(neg), max(20, int(len(pos) * neg_per_pos)))
    pick = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)]) if len(neg) else pos
    return rr[pick], cc[pick], y[pick]
