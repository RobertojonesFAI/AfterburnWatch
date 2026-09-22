import numpy as np
import pandas as pd
import pytest

from afterburn_watch import identify
from afterburn_watch.predict import STALEY2017_M1
from afterburn_watch.synthetic import make_fire, render_scenes, simulate_responses, training_pixels


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(7)
    X, y, g = [], [], []
    for f in range(4):
        fire = make_fire(rng, f"F{f}", "test", first_basin_id=1000 * (f + 1))
        truth = simulate_responses(rng, fire.basins, STALEY2017_M1[15], 10.0)
        b, a, valid, _ = render_scenes(rng, fire, truth, "sentinel2", cloud_fraction=0.0)
        stack = identify.compute_feature_stack(b, a, valid=valid)
        rows, cols, lab = training_pixels(rng, fire, stack, truth)
        X.append(identify.sample_features(stack, rows, cols))
        y.append(lab)
        g.append(np.full(len(lab), fire.fire_id))
    X, y, g = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(g)
    model = identify.DebrisFlowIdentifier("sentinel2", n_estimators=80)
    cv = model.cross_validate(X, y, g, n_splits=2)
    model.fit(X, y, groups=g)
    return model, cv, rng


def test_feature_stack_masks_cloudy_pixels():
    ones = {k: np.full((2, 2), 0.2, "float32") for k in ("red", "nir", "swir2")}
    valid = np.array([[True, False], [True, True]])
    st = identify.compute_feature_stack(ones, ones, valid=valid)
    assert set(st) == set(identify.FEATURES)
    assert np.isnan(st["event_dnbr"][0, 1]) and st["event_dnbr"][0, 0] == 0


def test_model1_learns_on_held_out_fires(trained):
    _, cv, _ = trained
    assert cv["auc"] > 0.8  # synthetic signal; only checks the plumbing works


def test_detect_guards_and_channel_mask(trained):
    model, _, rng = trained
    fire = make_fire(rng, "NEW", "test", first_basin_id=50_000)
    truth = np.ones(len(fire.basins), dtype=int)
    b, a, valid, _ = render_scenes(rng, fire, truth, "sentinel2", cloud_fraction=0.2)
    stack = identify.compute_feature_stack(b, a, valid=valid)
    with pytest.raises(ValueError, match="trained on"):
        identify.detect(model, stack, "landsat")
    prob = identify.detect(model, stack, "sentinel2", channel_mask=fire.channel)
    off = ~fire.channel & valid
    assert np.nanmax(prob[off]) == 0.0
    assert np.isnan(prob[~valid]).all()


def test_basin_observations_positive_negative_and_cloudy():
    labels = np.array([[1, 1, 2, 2, 3, 3]])
    prob = np.array([[0.9, 0.95, 0.1, 0.0, 0.8, np.nan]])
    valid = np.array([[True, True, True, True, True, False]])
    summ = identify.summarize_by_basin(prob, valid, labels)
    assert summ.set_index("basin_id").loc[3, "clear_fraction"] == 0.5
    basins = pd.DataFrame({"basin_id": [1, 2, 3], "lon": [0, 1, 2], "lat": [0, 0, 0],
                           "T": [.1, .2, .3], "F": [.1, .2, .3], "S": [.1, .2, .3]})
    o = identify.basin_observations(summ, basins, "F", "sentinel2", "2025-07-01",
                                    R={1: 9.0, 2: 7.0}, min_detected_px=2, region="idaho")
    got = o.set_index("basin_id")
    assert got.loc[1, "response"] == 1 and got.loc[1, "confidence"] == pytest.approx(0.95)
    assert got.loc[2, "response"] == 0 and got.loc[2, "confidence"] == 1.0
    assert 3 not in got.index  # half hidden by cloud and only 1 hit: no call either way
    assert got.loc[1, "R"] == 9.0 and got.loc[1, "region"] == "idaho"
    assert (o["verified"] == False).all()  # noqa: E712


def test_fuse_sensors_agreement_and_conflict():
    def o(sensor, resp, conf, basin):
        return {"obs_id": f"{sensor}-{basin}", "fire_id": "F", "response": resp, "source": f"sat_{sensor}",
                "confidence": conf, "verified": False, "lon": 0.0, "lat": 0.0,
                "event_date": "2025-07-01", "basin_id": basin}
    s2 = pd.DataFrame([o("sentinel2", 1, 0.8, 1), o("sentinel2", 1, 0.9, 2), o("sentinel2", 0, 0.9, 3)])
    ls = pd.DataFrame([o("landsat", 1, 0.7, 1), o("landsat", 0, 0.6, 2), o("landsat", 0, 0.95, 4)])
    f = identify.fuse_sensors(s2, ls).set_index("basin_id")
    assert f.loc[1, "source"] == "sat_fused" and f.loc[1, "confidence"] == pytest.approx(0.94)
    assert f.loc[2, "response"] == 1 and f.loc[2, "confidence"] == pytest.approx(0.3)  # conflict -> low
    assert f.loc[3, "source"] == "sat_sentinel2" and f.loc[4, "source"] == "sat_landsat"


def test_candidates_group_pixels_into_events():
    prob = np.zeros((6, 6))
    prob[1:4, 1] = 0.9  # 3-pixel event
    prob[5, 5] = 0.99  # single pixel: below min_pixels
    df = identify.candidates_to_observations(
        prob, 0.5, lambda r, c: (c * 1.0, -r * 1.0), "F", "sentinel2", "2025-07-01", min_pixels=3
    )
    assert len(df) == 1
    assert df.loc[0, "lon"] == pytest.approx(1.0) and df.loc[0, "lat"] == pytest.approx(-2.0)
    assert df.loc[0, "area_px"] == 3 and df.loc[0, "source"] == "sat_sentinel2"


def test_save_load_roundtrip(trained, tmp_path):
    model, _, _ = trained
    model.save(tmp_path / "m.joblib")
    back = identify.DebrisFlowIdentifier.load(tmp_path / "m.joblib")
    assert back.sensor == "sentinel2" and back.metadata["n_samples"] == model.metadata["n_samples"]
