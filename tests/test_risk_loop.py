import json

import numpy as np
import pandas as pd

from afterburn_watch.identify import basin_observations, summarize_by_basin
from afterburn_watch.loop import LoopConfig, LoopState, load_state, run_iteration, save_state
from afterburn_watch.predict import M1Coefficients
from afterburn_watch.risk import basin_risk_table, likelihood_class, outcome, to_geojson
from afterburn_watch.synthetic import (
    SYNTHETIC_REGIONAL_TRUTH,
    make_fire,
    seed_inventory,
    simulate_responses,
    storm_rainfall,
)


def test_likelihood_classes_and_outcomes():
    assert likelihood_class([0.1, 0.2, 0.5, 0.95, np.nan]).tolist() == [0, 1, 2, 4, -1]
    got = outcome([0.9, 0.9, 0.1, 0.1, 0.9], [1, 0, 1, 0, np.nan]).tolist()
    assert got == ["hit", "false_alarm", "miss", "correct_negative", None]


def test_basin_risk_table_and_geojson():
    basins = pd.DataFrame({"basin_id": [1, 2], "lon": [-115.0, -115.1], "lat": [44.0, 44.1],
                           "T": [0.6, 0.0], "F": [0.8, 0.1], "S": [0.3, 0.1]})
    observed = pd.DataFrame({"basin_id": [2], "response": [1]})
    t = basin_risk_table(basins, M1Coefficients.staley2017(), 12.0, observed)
    assert t.loc[0, "likelihood"] > t.loc[1, "likelihood"]
    assert t.loc[1, "outcome"] == "miss" and pd.isna(t.loc[0, "outcome"])
    gj = to_geojson(t)
    assert gj["features"][0]["geometry"]["coordinates"] == [-115.0, 44.0]
    assert gj["features"][0]["properties"]["outcome"] is None  # NaN -> null, valid JSON
    json.dumps(gj, allow_nan=False)  # must be strict JSON for ArcGIS Online / Folium


def _perfect_satellite_obs(rng, fire, truth, R, event, region):
    """Stand-in for Model 1 output: every basin clearly seen, right answer."""
    labels = np.where(fire.channel, fire.basin_labels, 0)
    prob = np.zeros(labels.shape)
    for bid, y in zip(fire.basins["basin_id"], truth):
        prob[(labels == bid)] = 0.97 if y else 0.0
    summ = summarize_by_basin(prob, np.ones(labels.shape, bool), labels)
    return basin_observations(summ, fire.basins, fire.fire_id, "sentinel2", event, R, region=region)


def test_loop_moves_regional_model_toward_truth_and_roundtrips(tmp_path):
    rng = np.random.default_rng(11)
    state = LoopState.start(seed_inventory(rng, n_fires=4, basins_per_fire=40))
    cfg = LoopConfig(region="idaho-test", min_obs=30)
    for k in range(8):
        fire = make_fire(rng, f"ID-{k}", "idaho-test", first_basin_id=1000 * (k + 1))
        R = storm_rainfall(rng, fire.basins, rng.uniform(6, 14))
        truth = simulate_responses(rng, fire.basins, SYNTHETIC_REGIONAL_TRUTH, R)
        dets = {"sentinel2": _perfect_satellite_obs(rng, fire, truth, R, f"2026-07-{k + 1:02d}", "idaho-test")}
        state, risk, report = run_iteration(state, dets, fire.basins, 10.0, cfg)

    assert len(state.history) == 8
    assert report["prediction"]["pool"] == 8 * 40  # only the region's observations are used
    dist_before = np.abs(state.base.theta - np.array(SYNTHETIC_REGIONAL_TRUTH)).sum()
    dist_after = np.abs(state.coefficients.theta - np.array(SYNTHETIC_REGIONAL_TRUTH)).sum()
    assert dist_after < 0.5 * dist_before
    assert state.base.theta.tolist() == M1Coefficients.staley2017().theta.tolist()
    assert set(risk["outcome"].dropna()) <= {"hit", "false_alarm", "miss", "correct_negative"}

    save_state(state, tmp_path / "s")
    back = load_state(tmp_path / "s")
    assert len(back.inventory) == len(state.inventory)
    np.testing.assert_allclose(back.coefficients.theta, state.coefficients.theta)
    assert len(back.history) == 8


def test_untrusted_detections_do_not_move_the_model():
    rng = np.random.default_rng(12)
    state = LoopState.start(seed_inventory(rng, n_fires=2, basins_per_fire=20))
    fire = make_fire(rng, "ID-X", "idaho-test", first_basin_id=5000)
    R = storm_rainfall(rng, fire.basins, 10.0)
    truth = simulate_responses(rng, fire.basins, SYNTHETIC_REGIONAL_TRUTH, R)
    dets = _perfect_satellite_obs(rng, fire, truth, R, "2026-07-01", "idaho-test")
    dets["confidence"] = 0.5  # low-confidence, unverified
    state2, _, report = run_iteration(state, {"sentinel2": dets}, fire.basins, 10.0,
                                      LoopConfig(region="idaho-test", min_obs=10))
    assert report["prediction"]["pool"] == 0
    assert state2.coefficients.theta.tolist() == state.coefficients.theta.tolist()
    assert report["observations"]["added"] == len(dets)  # still recorded for human review
