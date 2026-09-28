import numpy as np
import pandas as pd
import pytest

from afterburn_watch import usgs_assessment as ua


@pytest.mark.parametrize(
    "label,expected",
    [("20-40%", (0.2, 0.4)), ("20 - 40 %", (0.2, 0.4)), ("0–20%", (0.0, 0.2)),
     ("80-100%", (0.8, 1.0)), ("<20%", (0.0, 0.2)), (">80%", (0.8, 1.0))],
)
def test_parse_percent_range(label, expected):
    assert ua.parse_percent_range(label) == pytest.approx(expected)


def test_parse_percent_range_unreadable():
    assert all(np.isnan(v) for v in ua.parse_percent_range("High"))
    assert all(np.isnan(v) for v in ua.parse_percent_range(None))


def _bigrock_like():
    # same field names as the USGS output the team has (BigRockFire example)
    return pd.DataFrame({
        "BP_Legend": ["0-20%", "40-60%", "80-100%", "20-40%"],
        "BV_Legend": ["<1,000", "1,000-10,000", "10,000-100,000", "<1,000"],
        "BCH_Legend": ["Low", "Moderate", "High", "Low"],
        "fire_id": "wapiti",
        "Segment_ID": [11, 12, 13, 14],
    })


def test_assessment_table_points_and_labels():
    t = ua.assessment_table(_bigrock_like(), basin_id_col="Segment_ID")
    assert t["basin_id"].tolist() == [11, 12, 13, 14]
    assert t["likelihood"].tolist() == pytest.approx([0.1, 0.5, 0.9, 0.3])
    assert ua.assessment_table(_bigrock_like(), point="low")["likelihood"].tolist() == pytest.approx([0, .4, .8, .2])
    assert t.loc[2, "combined_label"] == "High" and t.loc[0, "fire_id"] == "wapiti"
    with pytest.raises(ValueError, match="BP_Legend"):
        ua.assessment_table(pd.DataFrame({"x": [1]}))


def test_score_against_observations():
    t = ua.assessment_table(_bigrock_like(), basin_id_col="Segment_ID")
    obs = pd.DataFrame({"basin_id": [11, 12, 13, 13], "response": [1, 0, 0, 1]})
    scored, counts = ua.score_against_observations(t, obs)
    got = scored.set_index("basin_id")["outcome"]
    assert got.loc[[11, 12, 13]].tolist() == ["miss", "false_alarm", "hit"]
    assert pd.isna(got.loc[14])  # never observed: no score
    assert counts == {"miss": 1, "false_alarm": 1, "hit": 1}


def test_load_and_spatial_join(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        _bigrock_like().iloc[:2],
        geometry=[box(-115.3, 44.0, -115.2, 44.1), box(-115.2, 44.0, -115.1, 44.1)],
        crs=4326,
    ).to_crs(32611)  # stored projected, like real assessments
    path = tmp_path / "assessment.gpkg"
    gdf.to_file(path)
    a = ua.load_assessment(path, basin_id_col="Segment_ID")
    assert a["lon"].between(-115.3, -115.1).all() and a.crs.to_epsg() == 32611
    obs = pd.DataFrame({"obs_id": ["a", "b", "c"], "lon": [-115.25, -115.15, -110.0],
                        "lat": [44.05, 44.05, 40.0], "response": [1, 0, 1]})
    joined = ua.attach_basin_ids_by_location(obs, a)
    assert joined["basin_id"].tolist()[:2] == [11, 12] and pd.isna(joined["basin_id"].iloc[2])
