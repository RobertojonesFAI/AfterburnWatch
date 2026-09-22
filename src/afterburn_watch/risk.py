"""Map Risk -- turn DF Prediction into the layer a fire manager sees.

Whiteboard (red): "DF Prediction" -> "Map Risk".

Output is plain GeoJSON, so it can go to ArcGIS Online (the hazard
assessment package / ESRI Dashboard in the Project Framework) or to the
Streamlit/Folium app, without extra GIS dependencies.

Besides the likelihood itself, every basin that has an actual observation
gets an outcome label. The whole point of the loop is to see where the
model and reality disagree:

    hit               predicted >= threshold, debris flow observed
    false_alarm       predicted >= threshold, no debris flow observed
    miss              predicted <  threshold, debris flow observed
    correct_negative  predicted <  threshold, no debris flow observed

"Predicted debris flows that never occurred" (Troy's kickoff idea) are the
false alarms.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from afterburn_watch.predict import M1Coefficients, likelihood, rainfall_threshold

# USGS hazard maps bin likelihood into five 20% classes.
LIKELIHOOD_BREAKS = (0.2, 0.4, 0.6, 0.8)
LIKELIHOOD_LABELS = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")


def likelihood_class(p) -> np.ndarray:
    """Class index 0-4 for each likelihood (NaN -> -1)."""
    p = np.asarray(p, dtype="float64")
    out = np.digitize(p, LIKELIHOOD_BREAKS).astype(int)
    out[np.isnan(p)] = -1
    return out


def outcome(predicted_p, observed, threshold: float = 0.5) -> np.ndarray:
    """hit / false_alarm / miss / correct_negative, or None where not observed."""
    p = np.asarray(predicted_p, dtype="float64")
    obs = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy()
    pos = p >= threshold
    labels = np.where(
        pos,
        np.where(obs == 1, "hit", "false_alarm"),
        np.where(obs == 1, "miss", "correct_negative"),
    ).astype(object)
    labels[np.isnan(obs) | np.isnan(p)] = None
    return labels


def basin_risk_table(
    basins: pd.DataFrame,
    coef: M1Coefficients,
    design_R: float,
    observations: pd.DataFrame | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Likelihood, class, rainfall threshold and observed outcome per basin.

    ``basins`` needs basin_id, T, F, S (and lon/lat for mapping).
    ``design_R`` is the design-storm rainfall accumulation in mm for the
    coefficient set's duration. ``observations`` (optional) needs basin_id
    and response. A basin with several observations takes the max, i.e.
    "a debris flow was observed there at least once".
    """
    out = basins.copy()
    out["likelihood"] = likelihood(coef, out["T"], out["F"], out["S"], design_R)
    out["likelihood_class"] = likelihood_class(out["likelihood"])
    out["likelihood_label"] = [LIKELIHOOD_LABELS[c] if c >= 0 else None for c in out["likelihood_class"]]
    out["R_threshold_mm_p50"] = rainfall_threshold(coef, out["T"], out["F"], out["S"], p=0.5)
    out["observed_response"] = np.nan
    if observations is not None and len(observations) and "basin_id" in observations:
        seen = observations.dropna(subset=["basin_id"]).groupby("basin_id")["response"].max()
        out["observed_response"] = out["basin_id"].map(seen)
    out["outcome"] = outcome(out["likelihood"], out["observed_response"], threshold)
    out["coef_source"] = coef.source
    out["design_R_mm"] = design_R
    return out


def to_geojson(table: pd.DataFrame, properties: list[str] | None = None) -> dict:
    """Point FeatureCollection from lon/lat columns (basin outlets or centroids).

    If the table has a ``geometry`` column holding GeoJSON geometry dicts
    (e.g. basin polygons exported from wildcat), that's used instead.
    """
    props = properties or [c for c in table.columns if c not in ("lon", "lat", "geometry")]
    features = []
    for _, row in table.iterrows():
        if "geometry" in table.columns and isinstance(row.get("geometry"), dict):
            geom = row["geometry"]
        else:
            if pd.isna(row.get("lon")) or pd.isna(row.get("lat")):
                continue
            geom = {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]}
        p = {}
        for k in props:
            v = row[k]
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            p[k] = v
        features.append({"type": "Feature", "geometry": geom, "properties": p})
    return {"type": "FeatureCollection", "features": features}


def write_geojson(table: pd.DataFrame, path, properties: list[str] | None = None) -> None:
    with open(path, "w") as fh:
        json.dump(to_geojson(table, properties), fh)
