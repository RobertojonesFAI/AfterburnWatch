"""The official USGS hazard assessment as the baseline "DF Prediction".

For a real fire, the first prediction to score is the one USGS already
published. For the Wapiti Fire, the BAER report (Oct 2024,
https://discoversawtooth.org/wp-content/uploads/2025/01/Wapiti-Fire-Burned-Area-Report-BAER.pdf)
says soil burn severity was passed to the USGS Landslide Hazards Program
for likelihood/volume modeling. The 2024 assessments are published in the
data release doi:10.5066/P13GTMAX:
https://www.usgs.gov/data/2024-post-wildfire-debris-flow-hazard-assessments

Output schema (seen in the BigRockFire example the team has in
``Resources/Source Code/PostFireDebrisFlowOutput/``), one row per basin:

    BP_Legend    likelihood class, e.g. "20-40%"
    BV_Legend    volume class
    BCH_Legend   combined hazard class
    fire_id, assessment, version, start_date

These are *classes*, not continuous probabilities, and the table has no T,
F, S. So the official product is good for scoring (hit / miss / false
alarm against actual observations) but not for recalibration. For that we
run pfdf ourselves on Lemhi (LiDAR DEM + severity + soil KF) to get T, F, S
per basin.

Reading the shapefile / geodatabase needs the ``gis`` extra
(``pip install -e ".[gis]"``). The table functions work on any DataFrame.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from afterburn_watch.risk import outcome

USGS_2024_ASSESSMENTS_DOI = "10.5066/P13GTMAX"
LEGEND_FIELDS = {"likelihood": "BP_Legend", "volume": "BV_Legend", "combined": "BCH_Legend"}


def parse_percent_range(label) -> tuple[float, float]:
    """Parse a likelihood class label into (low, high) as fractions.

    Handles "20-40%", "20 - 40 %", "20–40%" (en dash), "<20%", ">80%",
    "80-100%". Returns (nan, nan) if the label can't be read.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return (np.nan, np.nan)
    s = str(label).strip().replace("–", "-").replace("—", "-").replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%?-(\d+(?:\.\d+)?)%?", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo / 100, hi / 100)
    m = re.fullmatch(r"(<|<=|≤)(\d+(?:\.\d+)?)%?", s)
    if m:
        return (0.0, float(m.group(2)) / 100)
    m = re.fullmatch(r"(>|>=|≥)(\d+(?:\.\d+)?)%?", s)
    if m:
        return (float(m.group(2)) / 100, 1.0)
    return (np.nan, np.nan)


def assessment_table(df: pd.DataFrame, basin_id_col: str | None = None, point: str = "mid") -> pd.DataFrame:
    """Normalize a USGS assessment attribute table.

    ``point`` picks the single likelihood used for scoring: "low", "mid" or
    "high" end of each class. With "mid" and a 0.5 threshold, the 40-60%
    class counts as a predicted debris flow. Report the other two as a
    sensitivity check rather than picking the most flattering one.
    """
    if point not in ("low", "mid", "high"):
        raise ValueError("point must be 'low', 'mid' or 'high'")
    if LEGEND_FIELDS["likelihood"] not in df.columns:
        raise ValueError(
            f"no {LEGEND_FIELDS['likelihood']!r} column; columns present: {list(df.columns)}"
        )
    out = pd.DataFrame(index=df.index)
    out["basin_id"] = df[basin_id_col] if basin_id_col else np.arange(1, len(df) + 1)
    bounds = df[LEGEND_FIELDS["likelihood"]].map(parse_percent_range)
    out["likelihood_lo"] = [b[0] for b in bounds]
    out["likelihood_hi"] = [b[1] for b in bounds]
    out["likelihood"] = {
        "low": out["likelihood_lo"],
        "high": out["likelihood_hi"],
        "mid": (out["likelihood_lo"] + out["likelihood_hi"]) / 2,
    }[point]
    out["likelihood_label"] = df[LEGEND_FIELDS["likelihood"]].to_numpy()
    for key in ("volume", "combined"):
        col = LEGEND_FIELDS[key]
        out[f"{key}_label"] = df[col].to_numpy() if col in df.columns else None
    for col in ("fire_id", "assessment", "version", "start_date"):
        if col in df.columns:
            out[col] = df[col].to_numpy()
    for col in ("lon", "lat", "geometry"):
        if col in df.columns:
            out[col] = df[col].to_numpy() if col != "geometry" else df[col]
    return out.reset_index(drop=True)


def score_against_observations(
    assessment: pd.DataFrame,
    observations: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Label each assessed basin hit / miss / false alarm / correct negative.

    ``observations`` needs basin_id (matching ``assessment``) and response.
    A basin observed several times takes the max ("a debris flow was seen
    there at least once"). Returns (table, counts).
    """
    t = assessment.copy()
    seen = observations.dropna(subset=["basin_id"]).groupby("basin_id")["response"].max()
    t["observed_response"] = t["basin_id"].map(seen)
    t["outcome"] = outcome(t["likelihood"], t["observed_response"], threshold)
    counts = {k: int(v) for k, v in t["outcome"].value_counts().items()}
    return t, counts


def load_assessment(path, layer: str | None = None, basin_id_col: str | None = None, point: str = "mid"):
    """Read a USGS assessment shapefile / geodatabase layer (needs the gis extra).

    Adds lon/lat of each basin's representative point (WGS84) so the
    table can go straight to risk.to_geojson or be joined to observations.
    """
    import geopandas as gpd  # gis extra

    gdf = gpd.read_file(path, layer=layer)
    pts = gdf.to_crs(4326).representative_point() if gdf.crs else gdf.representative_point()
    gdf = gdf.assign(lon=pts.x.to_numpy(), lat=pts.y.to_numpy())
    table = assessment_table(gdf, basin_id_col, point)  # keeps the geometry column
    return gpd.GeoDataFrame(table, geometry="geometry", crs=gdf.crs)


def attach_basin_ids_by_location(observations: pd.DataFrame, assessment_gdf) -> pd.DataFrame:
    """Give each observation the basin_id of the USGS basin polygon it falls in.

    Observations need lon/lat (WGS84). Those outside every basin keep their
    existing basin_id (or NaN). Needs the gis extra.
    """
    import geopandas as gpd  # gis extra

    obs = observations.copy()
    pts = gpd.GeoDataFrame(obs, geometry=gpd.points_from_xy(obs["lon"], obs["lat"]), crs=4326)
    polys = assessment_gdf[["basin_id", "geometry"]]
    if polys.crs is not None and polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(4326)
    joined = gpd.sjoin(pts, polys, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    right = "basin_id_right" if "basin_id_right" in joined.columns else "basin_id"
    matched = joined[right]
    if "basin_id" in obs.columns:
        obs["basin_id"] = matched.where(matched.notna(), obs["basin_id"])
    else:
        obs["basin_id"] = matched
    return obs
