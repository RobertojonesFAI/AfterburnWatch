from datetime import date

import pytest

from afterburn_watch.ingest import fire_windows, search_params
from afterburn_watch.sensors import LANDSAT_C2_L2, SENTINEL2_L2A

BBOX = (-115.6, 44.0, -115.0, 44.5)


def test_search_params_sentinel2():
    p = search_params(SENTINEL2_L2A, BBOX, date(2024, 6, 1), date(2024, 7, 1), 20)
    assert p["collections"] == ["sentinel-2-l2a"]
    assert p["datetime"] == "2024-06-01/2024-07-01"
    assert p["query"] == {"eo:cloud_cover": {"lt": 20}}


def test_search_params_landsat_restricts_to_8_and_9():
    p = search_params(LANDSAT_C2_L2, BBOX, date(2024, 6, 1), date(2024, 7, 1))
    assert p["query"]["platform"] == {"in": ["landsat-9", "landsat-8"]}


def test_bad_bbox():
    with pytest.raises(ValueError):
        search_params(SENTINEL2_L2A, (1, 2, 3), date(2024, 1, 1), date(2024, 2, 1))


def test_fire_windows():
    w = fire_windows(date(2024, 7, 22), date(2024, 10, 1), storm_date=date(2025, 7, 10))
    assert w.pre[1] == date(2024, 7, 21)
    assert w.post_fire[0] == date(2024, 10, 2)
    assert w.post_event[0] == date(2025, 7, 11)
    assert fire_windows(date(2024, 7, 22), date(2024, 10, 1)).post_event is None


def test_storm_windows():
    from afterburn_watch.ingest import storm_windows

    before, after = storm_windows(date(2025, 8, 27), before_days=60, after_days=45)
    assert before == (date(2025, 6, 28), date(2025, 8, 26))
    assert after == (date(2025, 8, 28), date(2025, 10, 11))


def test_summarize_scenes_sorts_by_date():
    from datetime import datetime
    from types import SimpleNamespace

    from afterburn_watch.ingest import summarize_scenes

    items = [
        SimpleNamespace(id="b", datetime=datetime(2025, 9, 5), properties={"platform": "landsat-9", "eo:cloud_cover": 3.0}),
        SimpleNamespace(id="a", datetime=datetime(2025, 8, 20), properties={"platform": "sentinel-2b", "eo:cloud_cover": 12.5}),
    ]
    t = summarize_scenes(items)
    assert t["id"].tolist() == ["a", "b"] and t.loc[1, "platform"] == "landsat-9"
    assert summarize_scenes([]).empty
