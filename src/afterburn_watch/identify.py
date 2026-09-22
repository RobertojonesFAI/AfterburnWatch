"""Model 1 -- debris-flow identification from single-satellite parameters.

Whiteboard (blue):  DF Actual + Sentinel -> Model 1 ("DF Ident / Single
Sat Param") -> [apply to new cloud-masked Sentinel, ? Landsat ?, check
against DF Actual] -> New Observations.

Whiteboard (red):   "Identify by Sat-DF" -> "Actual DF Observations".

Not to be confused with USGS "M1" (the likelihood model in predict.py).

How it works
------------
1. Training ("Model 1" box): at locations where the outcome is known (the
   USGS inventory, "DF Actual"), sample spectral change features from ONE
   sensor ("single sat param") and fit a classifier: debris flow vs not.
2. Application (second box): compute the same features on new,
   cloud-masked imagery after a storm, predict a per-pixel probability,
   group adjacent hits into candidate events, and emit them as new
   observations. Skill is measured against the inventory with
   ``cross_validate``, which holds out whole fires.
3. Landsat ("? LandSat ?"): train and apply a separate Model 1 per sensor
   (their reflectances aren't identical), then ``fuse_sensors`` to raise
   confidence where both sensors agree. NASA's Harmonized Landsat
   Sentinel-2 (HLS) product would let one model serve both, with more
   frequent clear views; that's an open option.

Features (all from one sensor)
------------------------------
"before" = post-fire, pre-storm scene; "after" = post-storm scene.
A debris flow scours channels and leaves fresh sediment, so the after
scene tends to get brighter in SWIR and lose any remaining vegetation
signal along the flow path.

    nbr_before, nbr_after, event_dnbr   (NBR change across the storm, x1000)
    ndvi_before, ndvi_after, event_dndvi
    red_after, nir_after, swir2_after
    brightness_change                   (mean of red/nir/swir2, after - before)
    fire_dnbr                           (burn severity, pre-fire -> post-fire; optional)

Candidate methodology to extend this: Zhou et al. 2025 (NDVI -> Disturbance
Index -> Channel Disturbance Index), in the shared Articles folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from afterburn_watch import indices
from afterburn_watch.observations import CANONICAL, haversine_m

FEATURES = [
    "nbr_before",
    "nbr_after",
    "event_dnbr",
    "ndvi_before",
    "ndvi_after",
    "event_dndvi",
    "red_after",
    "nir_after",
    "swir2_after",
    "brightness_change",
    "fire_dnbr",
]


def compute_feature_stack(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    fire_dnbr: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-pixel Model 1 features from one sensor's before/after scenes.

    ``before`` / ``after`` map band roles ("red", "nir", "swir2") to 2-D
    reflectance arrays on the same grid. ``valid`` (True = keep) is the
    combined cloud/smoke mask from masking.py; masked pixels become NaN.
    ``fire_dnbr`` is the burn-severity dNBR (ours or RECOVER's); NaN if
    not given.
    """
    nbr_b = indices.nbr(before["nir"], before["swir2"])
    nbr_a = indices.nbr(after["nir"], after["swir2"])
    ndvi_b = indices.ndvi(before["nir"], before["red"])
    ndvi_a = indices.ndvi(after["nir"], after["red"])
    bright_b = (before["red"] + before["nir"] + before["swir2"]) / 3.0
    bright_a = (after["red"] + after["nir"] + after["swir2"]) / 3.0
    shape = nbr_b.shape
    stack = {
        "nbr_before": nbr_b,
        "nbr_after": nbr_a,
        "event_dnbr": indices.dnbr(nbr_b, nbr_a),
        "ndvi_before": ndvi_b,
        "ndvi_after": ndvi_a,
        "event_dndvi": indices.dndvi(ndvi_b, ndvi_a),
        "red_after": np.asarray(after["red"], dtype="float32"),
        "nir_after": np.asarray(after["nir"], dtype="float32"),
        "swir2_after": np.asarray(after["swir2"], dtype="float32"),
        "brightness_change": (bright_a - bright_b).astype("float32"),
        "fire_dnbr": (
            np.asarray(fire_dnbr, dtype="float32") if fire_dnbr is not None else np.full(shape, np.nan, "float32")
        ),
    }
    if valid is not None:
        for k in stack:
            stack[k] = np.where(valid, stack[k], np.nan).astype("float32")
    return stack


def sample_features(stack: dict[str, np.ndarray], rows, cols) -> pd.DataFrame:
    """Feature values at given pixel (row, col) positions -- e.g. inventory sites."""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    return pd.DataFrame({k: stack[k][rows, cols] for k in FEATURES})


@dataclass
class DebrisFlowIdentifier:
    """Model 1: a random forest over single-sensor spectral-change features."""

    sensor: str
    n_estimators: int = 300
    min_samples_leaf: int = 5
    random_state: int = 0
    model: object | None = None
    metadata: dict = field(default_factory=dict)

    def _new_model(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",  # debris flows are the minority class (~13% in the USGS inventory)
            random_state=self.random_state,
            n_jobs=-1,
        )

    @staticmethod
    def _clean(X: pd.DataFrame) -> np.ndarray:
        arr = X[FEATURES].to_numpy(dtype="float32")
        return np.where(np.isnan(arr), -9999.0, arr)

    def fit(self, X: pd.DataFrame, y, groups=None) -> "DebrisFlowIdentifier":
        y = np.asarray(y, dtype=int)
        keep = ~X[[f for f in FEATURES if f != "fire_dnbr"]].isna().any(axis=1).to_numpy()
        self.model = self._new_model()
        self.model.fit(self._clean(X[keep]), y[keep])
        self.metadata = {
            "sensor": self.sensor,
            "features": FEATURES,
            "n_samples": int(keep.sum()),
            "n_positive": int(y[keep].sum()),
            "n_fires": int(len(set(np.asarray(groups)[keep]))) if groups is not None else None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model 1 is not trained yet -- call fit() first")
        p = self.model.predict_proba(self._clean(X))[:, 1]
        core_missing = X[[f for f in FEATURES if f != "fire_dnbr"]].isna().any(axis=1).to_numpy()
        p[core_missing] = np.nan  # masked (cloud/smoke) pixels get no call
        return p

    def cross_validate(self, X: pd.DataFrame, y, groups, n_splits: int = 5, threshold: float = 0.5) -> dict:
        """Grouped CV: whole fires are held out, never split across folds.

        Pixels in the same fire are strongly correlated, so a random split
        would overstate skill. Holding out whole fires answers the question
        we care about: "how well does it work on a fire it has never seen?"
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import GroupKFold

        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups)
        n_splits = min(n_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError("need at least 2 fires for grouped cross-validation")
        oof = np.full(len(y), np.nan)
        for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = DebrisFlowIdentifier(self.sensor, self.n_estimators, self.min_samples_leaf, self.random_state)
            m.fit(X.iloc[train], y[train])
            oof[test] = m.predict_proba(X.iloc[test])
        ok = ~np.isnan(oof)
        pred = (oof[ok] >= threshold).astype(int)
        return {
            "n": int(ok.sum()),
            "n_folds": n_splits,
            "auc": float(roc_auc_score(y[ok], oof[ok])) if len(np.unique(y[ok])) == 2 else None,
            "precision": float(precision_score(y[ok], pred, zero_division=0)),
            "recall": float(recall_score(y[ok], pred, zero_division=0)),
            "f1": float(f1_score(y[ok], pred, zero_division=0)),
            "threshold": threshold,
        }

    def feature_importances(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return dict(sorted(zip(FEATURES, self.model.feature_importances_.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"sensor": self.sensor, "model": self.model, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DebrisFlowIdentifier":
        import joblib

        d = joblib.load(path)
        obj = cls(sensor=d["sensor"])
        obj.model, obj.metadata = d["model"], d["metadata"]
        return obj


def detect(
    identifier: DebrisFlowIdentifier,
    stack: dict[str, np.ndarray],
    sensor: str,
    threshold: float = 0.5,
    channel_mask: np.ndarray | None = None,
    allow_cross_sensor: bool = False,
) -> np.ndarray:
    """Per-pixel debris-flow probability for a feature stack (NaN where masked).

    ``channel_mask`` (True = on the drainage network, e.g. rasterized
    pfdf/wildcat stream segments) restricts calls to channels, where debris
    flows actually run. Pixels off the network get probability 0.
    """
    if sensor != identifier.sensor and not allow_cross_sensor:
        raise ValueError(
            f"Model 1 was trained on {identifier.sensor!r} but applied to {sensor!r}. "
            "Train one model per sensor (or use HLS), or pass allow_cross_sensor=True."
        )
    shape = stack[FEATURES[0]].shape
    flat = pd.DataFrame({k: stack[k].ravel() for k in FEATURES})
    prob = identifier.predict_proba(flat).reshape(shape)
    if channel_mask is not None:
        prob = np.where(channel_mask, prob, np.where(np.isnan(prob), np.nan, 0.0))
    return prob


def candidates_to_observations(
    prob: np.ndarray,
    threshold: float,
    pixel_to_lonlat: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    fire_id: str,
    sensor: str,
    event_date: str | None = None,
    min_pixels: int = 3,
    id_prefix: str = "sat",
) -> pd.DataFrame:
    """Group adjacent above-threshold pixels into candidate debris flows.

    Each connected group (8-connectivity) of at least ``min_pixels`` pixels
    becomes one observation at its probability-weighted centroid. Its
    confidence is the group's mean probability, and it is marked
    ``verified=False`` until a person confirms it.

    ``pixel_to_lonlat(rows, cols)`` converts pixel centers to WGS84, e.g.
    built from the odc-geo geobox returned by ingest.load_bands.
    """
    from scipy import ndimage

    hits = np.nan_to_num(prob, nan=0.0) >= threshold
    labels, n = ndimage.label(hits, structure=np.ones((3, 3), dtype=int))
    rows_out, cols_out, conf, size = [], [], [], []
    for lab in range(1, n + 1):
        rr, cc = np.nonzero(labels == lab)
        if len(rr) < min_pixels:
            continue
        w = prob[rr, cc]
        rows_out.append(float(np.average(rr, weights=w)))
        cols_out.append(float(np.average(cc, weights=w)))
        conf.append(float(w.mean()))
        size.append(len(rr))
    if not rows_out:
        return pd.DataFrame(columns=CANONICAL + ["area_px"])
    lon, lat = pixel_to_lonlat(np.array(rows_out), np.array(cols_out))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    df = pd.DataFrame(
        {
            "obs_id": [f"{id_prefix}-{sensor}-{fire_id}-{stamp}-{i}" for i in range(len(conf))],
            "fire_id": fire_id,
            "response": 1,
            "source": f"sat_{sensor}",
            "confidence": conf,
            "verified": False,
            "lon": lon,
            "lat": lat,
            "event_date": event_date,
            "area_px": size,
        }
    )
    return df.reindex(columns=CANONICAL + ["area_px"])


def summarize_by_basin(
    prob: np.ndarray,
    valid: np.ndarray,
    basin_labels: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-basin summary of a Model 1 probability raster.

    ``basin_labels`` is an integer raster on the same grid whose values are
    basin ids (0 = outside any basin), e.g. rasterized wildcat/pfdf basins
    or just their channel segments. Returns one row per basin: max
    probability, number of detected pixels, and the fraction of the basin
    the satellite could actually see (clear of cloud/smoke).
    """
    rows = []
    for bid in np.unique(basin_labels):
        if bid == 0:
            continue
        inside = basin_labels == bid
        clear = inside & valid
        p = prob[clear]
        p = p[~np.isnan(p)]
        rows.append(
            {
                "basin_id": int(bid),
                "clear_fraction": float(clear.sum() / inside.sum()),
                "max_prob": float(p.max()) if len(p) else np.nan,
                "detected_px": int((p >= threshold).sum()),
            }
        )
    return pd.DataFrame(rows)


def _rain_for(R, basin_ids) -> np.ndarray:
    ids = pd.Series(basin_ids).to_numpy()
    if R is None:
        return np.full(len(ids), np.nan)
    if isinstance(R, (dict, pd.Series)):
        lookup = R if isinstance(R, dict) else R.to_dict()
        return np.array([lookup.get(b, np.nan) for b in ids], dtype="float64")
    return np.full(len(ids), float(R))


def basin_observations(
    summary: pd.DataFrame,
    basins: pd.DataFrame,
    fire_id: str,
    sensor: str,
    event_date: str | None,
    R: float | dict | pd.Series | None,
    threshold: float = 0.5,
    min_detected_px: int = 3,
    min_clear: float = 0.8,
    region: str | None = None,
) -> pd.DataFrame:
    """Turn a basin summary into observations -- positives AND negatives.

    Recalibrating DF Prediction needs both outcomes. A basin becomes:
      * response 1 if Model 1 flagged at least ``min_detected_px`` pixels
        in it; confidence = its max probability.
      * response 0 if the satellite saw at least ``min_clear`` of it clear
        and flagged nothing; confidence = the clear fraction ("we looked
        and saw nothing").
      * no observation if it was too cloudy or smoky to tell.

    ``R`` is the storm's rainfall accumulation (mm) over the coefficient
    duration, from rainfall records (MRMS): one number for the whole storm,
    or per basin as a dict / Series keyed by basin_id. Per-basin is much
    better. With a single value per storm, the intercept and the rainfall
    terms of M1 are nearly impossible to separate. Without R the
    observation is still stored but can't be used to recalibrate.
    """
    merged = summary.merge(basins[["basin_id", "lon", "lat", "T", "F", "S"]], on="basin_id", how="left")
    pos = merged["detected_px"] >= min_detected_px
    neg = (~pos) & (merged["clear_fraction"] >= min_clear) & (merged["detected_px"] == 0)
    keep = pos | neg
    m = merged[keep].copy()
    m["response"] = pos[keep].astype(int).to_numpy()
    m["confidence"] = np.where(m["response"] == 1, m["max_prob"], m["clear_fraction"]).clip(0, 1)
    ev = event_date or "unknown"
    out = pd.DataFrame(
        {
            "obs_id": [f"sat-{sensor}-{fire_id}-{ev}-b{b}" for b in m["basin_id"]],
            "fire_id": fire_id,
            "region": region,
            "response": m["response"].to_numpy(),
            "source": f"sat_{sensor}",
            "confidence": m["confidence"].to_numpy(),
            "verified": False,
            "lon": m["lon"].to_numpy(),
            "lat": m["lat"].to_numpy(),
            "event_date": event_date,
            "basin_id": m["basin_id"].to_numpy(),
            "T": m["T"].to_numpy(),
            "F": m["F"].to_numpy(),
            "S": m["S"].to_numpy(),
            "R": _rain_for(R, m["basin_id"]),
        }
    )
    return out.reindex(columns=CANONICAL)


def fuse_sensors(a: pd.DataFrame, b: pd.DataFrame, max_distance_m: float = 60.0) -> pd.DataFrame:
    """Combine Sentinel-2 and Landsat observations of the same event.

    Rows are matched on ``basin_id`` when both have one (basin-level
    observations), otherwise by distance (``max_distance_m``, pixel-level
    candidates). For a matched pair:
      * same response -> one ``sat_fused`` row, confidence
        ``1 - (1 - pa) * (1 - pb)`` (noisy-OR). That rule assumes the
        sensors' errors are independent, which is only roughly true (same
        smoke, same storm), so treat it as a ranking signal, not a
        calibrated probability.
      * responses disagree -> keep the more confident one with confidence
        ``|pa - pb|``. That normally drops it below the trusted gate, which
        sends it to a person to review.
    Unmatched rows pass through unchanged.
    """
    if not len(a):
        return b.copy()
    if not len(b):
        return a.copy()
    used_b: set[int] = set()
    rows = []
    b_has_basin = "basin_id" in b and b["basin_id"].notna().any()
    for _, ra in a.iterrows():
        j = None
        if b_has_basin and pd.notna(ra.get("basin_id")):
            cand = [
                k
                for k in range(len(b))
                if k not in used_b
                and b.iloc[k]["basin_id"] == ra["basin_id"]
                and str(b.iloc[k]["event_date"]) == str(ra["event_date"])
            ]
            j = cand[0] if cand else None
        else:
            d = haversine_m(ra["lon"], ra["lat"], b["lon"].values, b["lat"].values)
            d = np.where(np.isin(np.arange(len(b)), list(used_b)), np.inf, d)
            k = int(np.argmin(d))
            j = k if d[k] <= max_distance_m else None
        if j is None:
            rows.append(ra)
            continue
        rb = b.iloc[j]
        used_b.add(j)
        fused = ra.copy()
        fused["obs_id"] = f"{ra['obs_id']}+{rb['obs_id']}"
        fused["source"] = "sat_fused"
        if ra["response"] == rb["response"]:
            fused["confidence"] = 1 - (1 - ra["confidence"]) * (1 - rb["confidence"])
        else:
            winner = ra if ra["confidence"] >= rb["confidence"] else rb
            fused["response"] = winner["response"]
            fused["confidence"] = abs(ra["confidence"] - rb["confidence"])
        fused["lon"] = (ra["lon"] + rb["lon"]) / 2
        fused["lat"] = (ra["lat"] + rb["lat"]) / 2
        rows.append(fused)
    rows += [b.iloc[k] for k in range(len(b)) if k not in used_b]
    return pd.DataFrame(rows).reset_index(drop=True)
