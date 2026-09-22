"""The feedback loop (red whiteboard), one iteration at a time.

    Identify by Sat-DF  ->  Actual DF Observations  ->  DF Prediction  ->  Map Risk
          ^                                                                   |
          +-------------------------- next storm / next scene ---------------+

One ``run_iteration`` call:

1. **Identify by Sat-DF** -- takes Model 1's new basin-level observations
   per sensor (identify.detect -> summarize_by_basin -> basin_observations)
   and fuses Sentinel-2 and Landsat observations of the same event.
2. **Actual DF Observations** -- attaches basin attributes (T, F, S) and
   merges the detections into the inventory, skipping duplicates.
3. **DF Prediction** -- recalibrates the USGS M1 coefficients from
   *trusted* observations only (optionally from one region only, which is
   Keith's "would region-specific models be better?" -- Idaho has no fires
   in the USGS inventory, so satellite observations are how an Idaho model
   gets data at all), and accepts the update only if it doesn't make
   predictions worse on fires held out of the fit.
4. **Map Risk** -- recomputes per-basin likelihood and labels every
   observed basin hit / miss / false alarm / correct negative.

Everything is recorded in ``state.history`` so the change across
iterations can be shown: coefficients, skill, number of observations.

Two guardrails keep the loop from reinforcing its own mistakes:
  * the ``trusted`` gate (observations.trusted): unverified satellite
    detections only count at high confidence;
  * the held-out acceptance test in step 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from afterburn_watch import observations as obs_mod
from afterburn_watch.identify import fuse_sensors
from afterburn_watch.predict import M1Coefficients, evaluate, recalibrate
from afterburn_watch.risk import basin_risk_table


@dataclass
class LoopConfig:
    region: str | None = None  # recalibrate a regional model from this region's observations only
    min_sat_confidence: float = 0.9  # trusted gate for unverified satellite detections
    prior_strength: float = 2.0  # how hard M1 coefficients resist moving (tuned on the synthetic demo only)
    min_obs: int = 30  # minimum usable observations to recalibrate
    dedupe_distance_m: float = 150.0
    basin_match_m: float = 500.0
    fuse_distance_m: float = 60.0
    holdout_fraction: float = 0.25  # share of fires held out to test an update
    acceptance_tolerance: float = 0.0  # allowed held-out log-loss increase
    likelihood_threshold: float = 0.5
    random_state: int = 0


@dataclass
class LoopState:
    inventory: pd.DataFrame
    coefficients: M1Coefficients  # current (possibly recalibrated) coefficients
    base: M1Coefficients  # published starting point; every refit is anchored here
    history: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, inventory: pd.DataFrame, duration_min: int = 15) -> "LoopState":
        """Begin from the seed inventory and the published USGS M1 coefficients."""
        base = M1Coefficients.staley2017(duration_min)
        return cls(obs_mod.validate(inventory), base, base)


def _split_by_fire(df: pd.DataFrame, fraction: float, seed: int):
    fires = np.array(sorted(df["fire_id"].astype(str).unique()))
    if len(fires) < 4:
        return df, None  # too few fires to hold any out honestly
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(fires) * fraction)))
    test_fires = set(rng.choice(fires, size=n_test, replace=False))
    is_test = df["fire_id"].astype(str).isin(test_fires)
    return df[~is_test], df[is_test]


def run_iteration(
    state: LoopState,
    detections: dict[str, pd.DataFrame],
    basins: pd.DataFrame,
    design_R: float,
    config: LoopConfig | None = None,
) -> tuple[LoopState, pd.DataFrame, dict]:
    """Run one pass of the loop. Returns ``(new_state, risk_table, report)``.

    ``detections`` maps sensor key ("sentinel2", "landsat") to that
    sensor's new basin-level observations from identify.basin_observations.
    Human-confirmed events can go in too, under any key, with
    ``verified=True``. ``basins`` needs basin_id, lon, lat, T, F, S.
    """
    cfg = config or LoopConfig()
    report: dict = {"iteration": len(state.history) + 1, "started_at": datetime.now(timezone.utc).isoformat()}

    # 1. Identify by Sat-DF: fuse sensors
    s2 = detections.get("sentinel2", pd.DataFrame())
    ls = detections.get("landsat", pd.DataFrame())
    others = [d for k, d in detections.items() if k not in ("sentinel2", "landsat") and len(d)]
    fused = fuse_sensors(s2, ls, cfg.fuse_distance_m) if (len(s2) or len(ls)) else pd.DataFrame()
    new = pd.concat([fused, *others], ignore_index=True) if (len(fused) or others) else obs_mod.empty_inventory()
    report["identify"] = {
        "sentinel2": int(len(s2)),
        "landsat": int(len(ls)),
        "other": int(sum(len(d) for d in others)),
        "after_fusion": int(len(new)),
    }

    # 2. Actual DF Observations: attach basin attributes, merge
    new = obs_mod.attach_basin_attributes(obs_mod.validate(new), basins, cfg.basin_match_m) if len(new) else new
    inventory, merge_stats = obs_mod.merge_new_observations(state.inventory, new, cfg.dedupe_distance_m)
    trusted = obs_mod.trusted(inventory, cfg.min_sat_confidence)
    report["observations"] = {**merge_stats, "inventory_size": int(len(inventory)), "trusted": int(len(trusted))}

    # 3. DF Prediction: recalibrate, accept only if held-out skill holds up.
    # Every refit starts from the published base coefficients using ALL the
    # observations so far. Refitting from the previous refit would count the
    # same observations twice.
    prior = state.coefficients
    pool = trusted if cfg.region is None else trusted[trusted["region"] == cfg.region]
    pool = pool.dropna(subset=["T", "F", "S", "R"])
    train, test = _split_by_fire(pool, cfg.holdout_fraction, cfg.random_state)
    candidate, fit_info = recalibrate(state.base, train, cfg.prior_strength, cfg.min_obs)
    decision = {"region": cfg.region, "pool": int(len(pool)), "fit": {k: v for k, v in fit_info.items() if k not in ("prior", "new")}}
    accepted = False
    if fit_info.get("updated"):
        if test is not None and len(test):
            before, after = evaluate(prior, test), evaluate(candidate, test)
            decision.update(holdout_prior=before, holdout_candidate=after)
            accepted = after["log_loss"] <= before["log_loss"] + cfg.acceptance_tolerance
        else:
            decision["holdout"] = "not enough fires to hold out; update accepted on in-sample fit only"
            accepted = True
    new_coef = prior
    if accepted:
        # refit on all trusted observations now that the update has passed
        new_coef, _ = recalibrate(state.base, pool, cfg.prior_strength, cfg.min_obs)
    decision["accepted"] = accepted
    decision["coefficients_before"] = prior.to_dict()
    decision["coefficients_after"] = new_coef.to_dict()
    report["prediction"] = decision

    # 4. Map Risk
    risk = basin_risk_table(basins, new_coef, design_R, inventory, cfg.likelihood_threshold)
    counts = risk["outcome"].value_counts().to_dict()
    report["map_risk"] = {
        "basins": int(len(risk)),
        "design_R_mm": design_R,
        "outcomes": {k: int(v) for k, v in counts.items()},
        "likelihood_classes": {str(k): int(v) for k, v in risk["likelihood_label"].value_counts().sort_index().items()},
    }
    report["config"] = asdict(cfg)

    new_state = LoopState(inventory=inventory, coefficients=new_coef, base=state.base, history=state.history + [report])
    return new_state, risk, report


def save_state(state: LoopState, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    state.inventory.to_csv(d / "inventory.csv", index=False)
    (d / "coefficients.json").write_text(json.dumps(state.coefficients.to_dict(), indent=2))
    (d / "base_coefficients.json").write_text(json.dumps(state.base.to_dict(), indent=2))
    (d / "history.json").write_text(json.dumps(state.history, indent=2, default=str))


def load_state(directory: str | Path) -> LoopState:
    d = Path(directory)
    inventory = obs_mod.validate(pd.read_csv(d / "inventory.csv"))
    coef = M1Coefficients(**json.loads((d / "coefficients.json").read_text()))
    base = M1Coefficients(**json.loads((d / "base_coefficients.json").read_text()))
    history = json.loads((d / "history.json").read_text())
    return LoopState(inventory=inventory, coefficients=coef, base=base, history=history)
