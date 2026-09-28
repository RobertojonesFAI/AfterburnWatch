import zipfile

import pandas as pd

from afterburn_watch import recover


def _fake_package(tmp_path):
    z = tmp_path / "pkg.zip"
    files = {
        "Wapiti/Wapiti_dNBR.tif": b"x" * 100,
        "Wapiti/Wapiti_RdNBR.tif": b"x" * 90,
        "Wapiti/DEM_LiDAR_1m.tif": b"x" * 400,
        "Wapiti/Slope.tif": b"x" * 50,
        "Wapiti/LANDFIRE_EVT.tif": b"x" * 60,
        "Wapiti/NDVI_Median.tif": b"x" * 70,
        "Wapiti/event_log.tif": b"x" * 10,  # "event" must not read as LANDFIRE "evt"
        "Wapiti/Wapiti.gdb/a00000001.gdbtable": b"x" * 30,
        "Wapiti/Wapiti.gdb/a00000002.gdbtable": b"x" * 20,
        "Wapiti/Wapiti_Severity.lyrx": b"{}",
        "Wapiti/report.html": b"<html/>",
    }
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return z


def test_catalog_zip_groups_geodatabase_and_assigns_roles(tmp_path):
    cat = recover.catalog_zip(_fake_package(tmp_path)).set_index("path")
    gdb = cat.loc["Wapiti/Wapiti.gdb"]
    assert gdb["kind"] == "geodatabase" and gdb["n_files"] == 2 and gdb["size_bytes"] == 50
    assert cat.loc["Wapiti/Wapiti_dNBR.tif", "role"] == "severity"
    assert cat.loc["Wapiti/DEM_LiDAR_1m.tif", "role"] == "dem"
    assert cat.loc["Wapiti/LANDFIRE_EVT.tif", "role"] == "landfire"
    assert pd.isna(cat.loc["Wapiti/event_log.tif", "role"])
    assert cat.loc["Wapiti/Wapiti_Severity.lyrx", "kind"] == "layerfile"
    assert cat.loc["Wapiti/report.html", "kind"] == "report"


def test_extract_members_only_takes_requested_rasters(tmp_path):
    z = _fake_package(tmp_path)
    got = recover.extract_members(z, tmp_path / "out", roles=("severity", "dem"))
    names = sorted(p.name for p in got)
    assert names == ["DEM_LiDAR_1m.tif", "Wapiti_RdNBR.tif", "Wapiti_dNBR.tif"]
    with_gdb = recover.extract_members(z, tmp_path / "out2", roles=(), include_geodatabases=True)
    assert all(".gdb" in str(p) for p in with_gdb) and len(with_gdb) == 2


def test_find_dnbr_prefers_continuous_dnbr(tmp_path):
    z = _fake_package(tmp_path)
    d = recover.load_recover_package(z, tmp_path / "full")
    assert recover.find_dnbr_raster(d).name == "Wapiti_dNBR.tif"
    assert [p.name for p in recover.find_rasters(d, "dem")] == ["DEM_LiDAR_1m.tif"]
    assert recover.catalog_dir(d).shape[0] == recover.catalog_zip(z).shape[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert recover.find_dnbr_raster(empty) is None
