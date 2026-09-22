"""Actual debris-flow observations -- "DF Actual" / "Actual DF Observations".

This is the loop's memory. It starts from the USGS inventory and grows as
Model 1 identifies new debris flows from Sentinel-2 / Landsat.

Seed data: USGS Runoff-Generated Postfire Debris-Flow Inventory
(Graber, Gorr, Kean, Kostelnik, Rengers, Selander & Thomas, 2026;
doi:10.5066/P1R9VXC4) -- 8,981 observations from 80 burned areas in AZ, CA,
CO, MT, NM, UT, WA, WY, of which 1,140 are observed debris flows
(Response = 1). Found by Troy (see Articles/Additional Articles.xlsx).

TODO: download ``USGS_RG_PFDF_Inventory_v1.csv`` + its README and confirm
the column mapping. ``load_inventory`` guesses common names and fails
loudly with the list of actual columns if it can't find the required ones,
so pass ``column_map`` to fix it.

Canonical columns
-----------------
obs_id        unique id
fire_id       fire / burned-area name
region        region used for regional recalibration (e.g. state or EPA
              ecoregion -- Graber et al. 2026 fit regional models)
response      1 = debris flow, 0 = no debris flow
source        "usgs_inventory", "field", "sat_sentinel2", "sat_landsat", "sat_fused"
confidence    0-1 (1.0 for inventory/field observations)
verified      True if confirmed by a person (inventory, field visit, imagery review)
lon, lat      WGS84 degrees (needed to sample imagery for Model 1)
event_date    date of the triggering storm / observation, if known
basin_id      basin/segment the observation belongs to (see attach_basin_attributes)
T, F, S, R    USGS M1 variables for the basin (needed to recalibrate DF Prediction)

Why "verified" and "confidence" matter: a feedback loop that trains on
its own unverified detections can drift and reinforce its own mistakes.
``trusted`` is the gate: by default only verified observations, or
satellite detections above a high confidence, flow back into the models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = [
    "obs_id",
    "fire_id",
    "region",
    "response",
    "source",
    "confidence",
    "verified",
    "lon",
    "lat",
    "event_date",
    "basin_id",
    "T",
    "F",
    "S",
    "R",
]
REQUIRED = ["fire_id", "response"]
SAT_PREFIX = "sat_"  # any satellite-derived source: sat_sentinel2, sat_landsat, sat_fused, ...

# Lower-cased candidate names tried, in order, when no column_map is given.
# Deliberately short: only unambiguous names are guessed. T, F, S, R and
# projected x/y coordinates are never guessed -- a wrong silent match there
# would corrupt the recalibration, so map them explicitly.
_GUESSES = {
    "obs_id": ["obs_id", "objectid", "observationid"],
    "fire_id": ["fire_id", "firename", "fire_name", "fire"],
    "response": ["response", "debrisflow", "debris_flow"],
    "lon": ["lon", "longitude"],
    "lat": ["lat", "latitude"],
    "event_date": ["event_date", "storm_date", "stormdate"],
}


def empty_inventory() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL})


def _guess_columns(columns: list[str]) -> dict[str, str]:
    lower = {c.lower().replace(" ", ""): c for c in columns}
    found = {}
    for canon, candidates in _GUESSES.items():
        for cand in candidates:
            if cand in lower:
                found[lower[cand]] = canon
                break
    return found


def load_inventory(
    path: str | Path,
    column_map: dict[str, str] | None = None,
    source: str = "usgs_inventory",
) -> pd.DataFrame:
    """Load an observation table (e.g. the USGS inventory CSV) into canonical form.

    ``column_map`` maps *source* column names to canonical names, e.g.
    ``{"Fire_Name": "fire_id", "Response": "response"}``. Without it,
    common names are guessed.
    """
    raw = pd.read_csv(path)
    mapping = column_map if column_map is not None else _guess_columns(list(raw.columns))
    df = raw.rename(columns=mapping)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"could not find required column(s) {missing} in {path}. "
            f"Columns present: {list(raw.columns)}. Pass column_map="
            "{'<source name>': '<canonical name>'}."
        )
    df = df[[c for c in df.columns if c in CANONICAL]].copy()
    df["source"] = df.get("source", source)
    df["confidence"] = df.get("confidence", 1.0)
    df["verified"] = df.get("verified", True)
    if "obs_id" not in df.columns:
        df["obs_id"] = [f"{source}-{i}" for i in range(len(df))]
    return validate(df.reindex(columns=CANONICAL))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check canonical columns and value ranges. Returns a cleaned copy."""
    df = df.reindex(columns=CANONICAL).copy()
    resp = pd.to_numeric(df["response"], errors="coerce")
    if resp.isna().any() or not resp.isin([0, 1]).all():
        raise ValueError("response must be 0 or 1 for every row")
    df["response"] = resp.astype(int)
    conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)
    if ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must be within [0, 1]")
    df["confidence"] = conf
    df["verified"] = df["verified"].fillna(False).astype(bool)
    for c in ["lon", "lat", "T", "F", "S", "R"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["obs_id"].duplicated().any():
        raise ValueError("obs_id values must be unique")
    return df


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in meters (vectorized, broadcasting)."""
    r = 6_371_008.8
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype="float64")) for v in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def merge_new_observations(
    inventory: pd.DataFrame,
    new: pd.DataFrame,
    dedupe_distance_m: float = 150.0,
) -> tuple[pd.DataFrame, dict]:
    """Add new observations ("New Observations" box), skipping duplicates.

    A new row is a duplicate if an existing observation has the same fire,
    the same response and the same event date, and lies within
    ``dedupe_distance_m``. The same basin observed after two different
    storms counts as two observations. Returns ``(merged, stats)``.
    """
    inventory = validate(inventory) if len(inventory) else empty_inventory()
    new = validate(new) if len(new) else empty_inventory()
    keep = np.ones(len(new), dtype=bool)
    for i, row in enumerate(new.itertuples(index=False)):
        same = inventory[
            (inventory["fire_id"] == row.fire_id)
            & (inventory["response"] == row.response)
            & (inventory["event_date"].astype(str) == str(row.event_date))
        ]
        same = same.dropna(subset=["lon", "lat"])
        if len(same) and not (np.isnan(row.lon) or np.isnan(row.lat)):
            d = haversine_m(row.lon, row.lat, same["lon"].values, same["lat"].values)
            if (d <= dedupe_distance_m).any():
                keep[i] = False
    added = new[keep]
    clash = set(added["obs_id"]) & set(inventory["obs_id"])
    if clash:
        raise ValueError(f"obs_id collision with existing inventory: {sorted(clash)[:5]}")
    merged = pd.concat([inventory, added], ignore_index=True) if len(inventory) else added.reset_index(drop=True)
    stats = {"offered": int(len(new)), "added": int(keep.sum()), "duplicates": int((~keep).sum())}
    return validate(merged), stats


def trusted(df: pd.DataFrame, min_sat_confidence: float = 0.9) -> pd.DataFrame:
    """Observations allowed to feed back into model training/recalibration.

    Verified rows always pass. Unverified satellite detections pass only
    at or above ``min_sat_confidence``. This is the guard against the loop
    reinforcing its own mistakes.
    """
    is_sat = df["source"].astype(str).str.startswith(SAT_PREFIX)
    ok = df["verified"] | (is_sat & (df["confidence"] >= min_sat_confidence))
    return df[ok].copy()


def attach_basin_attributes(
    obs: pd.DataFrame,
    basins: pd.DataFrame,
    max_distance_m: float = 500.0,
) -> pd.DataFrame:
    """Fill missing T, F, S (and basin_id) from ``basins``.

    ``basins`` needs basin_id, lon, lat, T, F, S, e.g. from a wildcat/pfdf
    run on Lemhi (see predict.py). Rows that already carry a basin_id are
    joined on it. The rest take the nearest basin within
    ``max_distance_m``, or stay NaN and are skipped by recalibration.
    R (the storm's rainfall) is event-specific and must come with the
    observation itself, e.g. MRMS rainfall for that storm.
    """
    out = obs.copy()
    if not len(basins) or not len(out):
        return out
    by_id = basins.set_index("basin_id")
    for idx, row in out.iterrows():
        b = None
        if pd.notna(row.get("basin_id")) and row["basin_id"] in by_id.index:
            b = by_id.loc[row["basin_id"]]
        elif not (pd.isna(row["lon"]) or pd.isna(row["lat"])):
            d = haversine_m(row["lon"], row["lat"], basins["lon"].values, basins["lat"].values)
            j = int(np.argmin(d))
            if d[j] <= max_distance_m:
                b = basins.iloc[j]
                out.at[idx, "basin_id"] = b["basin_id"]
        if b is None:
            continue
        for c in ("T", "F", "S"):
            if pd.isna(out.at[idx, c]):
                out.at[idx, c] = b[c]
    return out
