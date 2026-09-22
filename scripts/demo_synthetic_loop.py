"""Run the whole whiteboard loop end-to-end on SYNTHETIC data.

    python scripts/demo_synthetic_loop.py [--iterations 8] [--out demo_output]

Nothing here is real imagery or real observations (see
src/afterburn_watch/synthetic.py). The point is to show the mechanics
working together before real Sentinel-2 / Landsat scenes and the USGS
inventory are wired in:

  Blue board   DF Actual + Sentinel -> Model 1 -> (cloud-masked Sentinel,
               Landsat) -> New Observations
  Red board    Identify by Sat-DF -> Actual DF Observations -> DF Prediction
               -> Map Risk -> (next storm) ...

The scenario: the published USGS M1 coefficients fit the (synthetic)
inventory, but the (synthetic) Idaho region behaves differently, and
Idaho has no fires in the USGS inventory. Each storm, Model 1 turns
Sentinel-2 and Landsat imagery into basin observations, and the regional
M1 is recalibrated from them. The printout tracks how well each version
predicts a separate Idaho fire it never saw.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from afterburn_watch.identify import (
    DebrisFlowIdentifier,
    basin_observations,
    compute_feature_stack,
    detect,
    sample_features,
    summarize_by_basin,
)
from afterburn_watch.loop import LoopConfig, LoopState, run_iteration, save_state
from afterburn_watch.predict import STALEY2017_M1, evaluate
from afterburn_watch.risk import write_geojson
from afterburn_watch.synthetic import (
    SYNTHETIC_REGIONAL_TRUTH,
    make_fire,
    render_scenes,
    seed_inventory,
    simulate_responses,
    storm_rainfall,
    training_pixels,
)

SENSORS = ("sentinel2", "landsat")
REGION = "idaho-synthetic"


def train_model1(rng, n_fires: int = 6):
    """Blue board, left half: DF Actual + imagery -> Model 1 (one per sensor)."""
    data = {s: {"X": [], "y": [], "g": []} for s in SENSORS}
    for f in range(n_fires):
        fire = make_fire(rng, f"INV-TRAIN-{f}", "inventory", first_basin_id=100_000 + 1000 * f)
        R = rng.uniform(6, 14)
        truth = simulate_responses(rng, fire.basins, STALEY2017_M1[15], R)
        for s in SENSORS:
            before, after, valid, _ = render_scenes(rng, fire, truth, s, cloud_fraction=0.05)
            stack = compute_feature_stack(before, after, valid=valid)
            rows, cols, y = training_pixels(rng, fire, stack, truth)
            data[s]["X"].append(sample_features(stack, rows, cols))
            data[s]["y"].append(y)
            data[s]["g"].append(np.full(len(y), fire.fire_id))
    models, cv = {}, {}
    for s in SENSORS:
        X = pd.concat(data[s]["X"], ignore_index=True)
        y = np.concatenate(data[s]["y"])
        g = np.concatenate(data[s]["g"])
        m = DebrisFlowIdentifier(s, n_estimators=150)
        cv[s] = m.cross_validate(X, y, g, n_splits=3)
        models[s] = m.fit(X, y, groups=g)
    return models, cv


def holdout_fire(rng, n_storms: int = 5) -> pd.DataFrame:
    """A separate Idaho fire, never used in the loop, to score predictions."""
    fire = make_fire(rng, "ID-HOLDOUT", REGION, first_basin_id=900_000)
    rows = []
    for _ in range(n_storms):
        R = storm_rainfall(rng, fire.basins, rng.uniform(4, 16))
        y = simulate_responses(rng, fire.basins, SYNTHETIC_REGIONAL_TRUTH, R)
        rows.append(fire.basins.assign(R=R.to_numpy(), response=y))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("SYNTHETIC DATA ONLY -- mechanics demo, not results.\n")

    # --- Blue board: train Model 1 on DF Actual -----------------------------
    models, cv = train_model1(rng)
    for s in SENSORS:
        c = cv[s]
        print(f"Model 1 [{s:9s}] grouped CV (whole fires held out): "
              f"AUC {c['auc']:.2f}  precision {c['precision']:.2f}  recall {c['recall']:.2f}")
        models[s].save(out / f"model1_{s}.joblib")
    print()

    # --- Red board: the feedback loop ---------------------------------------
    state = LoopState.start(seed_inventory(rng))
    cfg = LoopConfig(region=REGION)
    test = holdout_fire(rng)
    base_skill = evaluate(state.base, test)
    print(f"Held-out Idaho fire, published USGS M1: log loss {base_skill['log_loss']:.3f}  "
          f"AUC {base_skill['auc']:.3f}\n")
    header = f"{'iter':>4} {'storm R':>7} {'S2 obs':>6} {'LS obs':>6} {'basin acc':>9} {'Idaho pool':>10} {'update':>8} {'log loss':>8} {'AUC':>6}"
    print(header)
    print("-" * len(header))

    risk = None
    for k in range(args.iterations):
        fire = make_fire(rng, f"ID-SYN-{k}", REGION, first_basin_id=1000 * (k + 1))
        storm_mean = float(rng.uniform(6, 14))
        R = storm_rainfall(rng, fire.basins, storm_mean)  # per basin, as MRMS would give
        truth = simulate_responses(rng, fire.basins, SYNTHETIC_REGIONAL_TRUTH, R)
        event = (date(2026, 6, 1) + timedelta(days=10 * k)).isoformat()
        channel_labels = np.where(fire.channel, fire.basin_labels, 0)
        dets = {}
        for s in SENSORS:
            before, after, valid, _ = render_scenes(rng, fire, truth, s, cloud_fraction=float(rng.uniform(0.05, 0.3)))
            stack = compute_feature_stack(before, after, valid=valid)  # Cloud Masking
            prob = detect(models[s], stack, s, channel_mask=fire.channel)
            summary = summarize_by_basin(prob, valid, channel_labels)
            dets[s] = basin_observations(summary, fire.basins, fire.fire_id, s, event, R, region=REGION)

        state, risk, report = run_iteration(state, dets, fire.basins, design_R=10.0, config=cfg)

        # how right were the satellite observations about this storm?
        inv = state.inventory
        new = inv[(inv["fire_id"] == fire.fire_id) & (inv["event_date"] == event)].dropna(subset=["basin_id"])
        truth_by_basin = dict(zip(fire.basins["basin_id"], truth))
        acc = np.mean([truth_by_basin[int(b)] == r for b, r in zip(new["basin_id"], new["response"])]) if len(new) else np.nan
        skill = evaluate(state.coefficients, test)
        pred = report["prediction"]
        print(f"{k + 1:>4} {storm_mean:>7.1f} {len(dets['sentinel2']):>6} {len(dets['landsat']):>6} {acc:>9.2f} "
              f"{pred['pool']:>10} {'accepted' if pred['accepted'] else 'kept':>8} "
              f"{skill['log_loss']:>8.3f} {skill['auc']:>6.3f}")

    c0, c1 = state.base, state.coefficients
    print("\nM1 coefficients (B, Ct, Cf, Cs)")
    print(f"  published Staley 2017 : {c0.B:6.2f} {c0.Ct:6.2f} {c0.Cf:6.2f} {c0.Cs:6.2f}")
    print(f"  after the loop        : {c1.B:6.2f} {c1.Ct:6.2f} {c1.Cf:6.2f} {c1.Cs:6.2f}")
    t = SYNTHETIC_REGIONAL_TRUTH
    print(f"  synthetic Idaho truth : {t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f} {t[3]:6.2f}")

    save_state(state, out / "loop_state")
    if risk is not None:
        write_geojson(risk, out / "map_risk_latest.geojson")
    (out / "model1_cv.json").write_text(json.dumps(cv, indent=2))
    print(f"\nSaved loop state, Model 1 files and the latest risk map to {out}/")


if __name__ == "__main__":
    main()
