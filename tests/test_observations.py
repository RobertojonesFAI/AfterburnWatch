import numpy as np
import pandas as pd
import pytest

from afterburn_watch import observations as obs


def _row(i, **kw):
    base = dict(obs_id=f"o{i}", fire_id="A", response=1, source="usgs_inventory", confidence=1.0,
                verified=True, lon=-115.0, lat=44.0, event_date="2025-07-01")
    base.update(kw)
    return base


def test_load_inventory_guesses_unambiguous_columns(tmp_path):
    p = tmp_path / "inv.csv"
    pd.DataFrame({"Fire_Name": ["X", "Y"], "Response": [1, 0], "Latitude": [44.0, 44.1],
                  "Longitude": [-115, -115.1], "Acc15": [8.0, 9.0]}).to_csv(p, index=False)
    df = obs.load_inventory(p)
    assert df["fire_id"].tolist() == ["X", "Y"]
    assert df["response"].tolist() == [1, 0]
    assert df["verified"].all() and (df["confidence"] == 1.0).all()
    assert df["R"].isna().all()  # rainfall is never guessed -- must be mapped explicitly


def test_load_inventory_explicit_map_and_helpful_error(tmp_path):
    p = tmp_path / "inv.csv"
    pd.DataFrame({"burn": ["X"], "DF": [1], "Acc15": [8.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Columns present"):
        obs.load_inventory(p)
    df = obs.load_inventory(p, column_map={"burn": "fire_id", "DF": "response", "Acc15": "R"})
    assert df.loc[0, "R"] == 8.0


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        obs.validate(pd.DataFrame([_row(1, response=2)]))
    with pytest.raises(ValueError):
        obs.validate(pd.DataFrame([_row(1, confidence=1.5)]))
    with pytest.raises(ValueError):
        obs.validate(pd.DataFrame([_row(1), _row(1)]))


def test_haversine_one_degree_latitude():
    assert obs.haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_merge_dedupes_same_event_but_keeps_other_storms():
    inv = pd.DataFrame([_row(1)])
    new = pd.DataFrame([
        _row(2, lon=-115.0005),  # ~40 m away, same storm -> duplicate
        _row(3, lon=-115.0005, event_date="2025-08-15"),  # same place, later storm -> keep
        _row(4, lon=-114.9),  # far away -> keep
    ])
    merged, stats = obs.merge_new_observations(inv, new)
    assert stats == {"offered": 3, "added": 2, "duplicates": 1}
    assert set(merged["obs_id"]) == {"o1", "o3", "o4"}


def test_trusted_gate():
    df = obs.validate(pd.DataFrame([
        _row(1),  # verified inventory
        _row(2, source="sat_sentinel2", verified=False, confidence=0.95),
        _row(3, source="sat_landsat", verified=False, confidence=0.6),
        _row(4, source="field", verified=False, confidence=0.99),  # not satellite, not verified
    ]))
    assert obs.trusted(df)["obs_id"].tolist() == ["o1", "o2"]


def test_attach_basin_attributes_by_id_then_nearest():
    basins = pd.DataFrame({"basin_id": [10, 11], "lon": [-115.0, -114.0], "lat": [44.0, 44.0],
                           "T": [0.3, 0.6], "F": [0.4, 0.7], "S": [0.2, 0.3]})
    o = obs.validate(pd.DataFrame([_row(1, basin_id=11, lon=np.nan, lat=np.nan), _row(2, lon=-115.001)]))
    out = obs.attach_basin_attributes(o, basins)
    assert out.loc[0, "T"] == 0.6
    assert out.loc[1, "basin_id"] == 10 and out.loc[1, "F"] == 0.4
