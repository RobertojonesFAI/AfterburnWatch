"""NASA RECOVER data packages -- authoritative inputs for DF Prediction.

Project Framework: "USFS Server -> RECOVER package". A RECOVER package for
a fire bundles the authoritative burn severity, perimeter, soils and
terrain layers.

Role in the loop:
  * Its severity, soils and DEM are what pfdf/wildcat need to compute each
    basin's T, F, S for DF Prediction (predict.py). These are the same
    inputs federal stakeholders already trust.
  * Its severity is the reference to cross-check the dNBR we compute from
    Sentinel-2 / Landsat (indices.py), and it can feed Model 1 as the
    ``fire_dnbr`` feature (identify.py).

What's in a package (per the RECOVER tutorial,
https://fsapps.nwcg.gov/RECOVER/GettingFamiliarWithRECOVER.pdf)
----------------------------------------------------------------------
  * a file geodatabase: fire perimeter, Fires1950_Present, Soils_SSURGO
    (erosion ratings, hydrologic group), surface management agency,
    WBD / HU12 watersheds
  * ~19 GeoTIFFs: NDVI_Median, LANDFIRE EVT / BPS / EVC / FVT, DEM, slope, ...
  * ArcGIS layer files (.lyrx), an HTML report and a toolbox

Caution: the preliminary burn severity (dNBR) may be delivered as a web
service referenced from the layer files rather than as a GeoTIFF. Run
``catalog_zip`` first and check before assuming a severity raster exists.

The Wapiti package
------------------
Keith Weber (Sept 23, 2026) shared the full Wapiti RECOVER package plus
the LiDAR DEM: ``WAPITI_PACKAGE`` below, 36.1 GB. Download it straight to
the team's shared directory on Lemhi, then catalog it without extracting
everything:

    from afterburn_watch.recover import catalog_zip, extract_members
    cat = catalog_zip("WapitiFire_2024_IDBOF_000683.zip")
    print(cat.groupby(["kind", "role"]).size())
    extract_members("WapitiFire_2024_IDBOF_000683.zip", "wapiti/", roles=("severity", "dem"))

Other fires: RECOVER ArcGIS dashboard
(https://www.arcgis.com/apps/dashboards/68f43718f2474f369c587a97d32cd0cf).
Click the fire's polygon (past fires: hamburger menu -> bulk download ->
previous years), then "Download recover data package".
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

RECOVER_DASHBOARD_URL = "https://www.arcgis.com/apps/dashboards/68f43718f2474f369c587a97d32cd0cf"
RECOVER_TUTORIAL_URL = "https://fsapps.nwcg.gov/RECOVER/GettingFamiliarWithRECOVER.pdf"

WAPITI_PACKAGE = {
    "url": "https://giscenter-sl.isu.edu/AOC/TEMP/",
    "file": "WapitiFire_2024_IDBOF_000683.zip",
    "size_gb": 36.1,
    "contents": "full RECOVER package + LiDAR DEM",
    "from": "Keith Weber, Sept 23 2026",
}

# Layer roles recognized from file names. Matching is on name tokens (split
# on anything that isn't a letter or digit), so "evt" won't match "event".
ROLE_KEYWORDS = {
    "severity": ("dnbr", "rdnbr", "barc", "sbs", "severity", "burnseverity"),
    "dem": ("dem", "lidar", "elevation", "dtm"),
    "slope": ("slope",),
    "ndvi": ("ndvi",),
    "landfire": ("landfire", "evt", "bps", "evc", "fvt"),
    "perimeter": ("perimeter", "perim"),
    "soils": ("soil", "soils", "ssurgo", "kf"),
}

_KINDS = {
    "raster": (".tif", ".tiff", ".img", ".vrt"),
    "vector": (".shp", ".geojson", ".gpkg", ".kml", ".kmz"),
    "layerfile": (".lyrx", ".lyr"),
    "report": (".html", ".htm", ".pdf"),
    "toolbox": (".atbx", ".tbx", ".pyt"),
}


def _kind(name: str) -> str:
    low = name.lower()
    for kind, exts in _KINDS.items():
        if low.endswith(exts):
            return kind
    return "other"


def _tokens(name: str) -> list[str]:
    stem = PurePosixPath(name).name.lower()
    stem = re.sub(r"\.(tif|tiff|img|vrt|shp|geojson|gpkg|lyrx|lyr|gdb|html|pdf)$", "", stem)
    return [t for t in re.split(r"[^a-z0-9]+", stem) if t]


def _role(name: str) -> str | None:
    toks = _tokens(name)
    for role, kws in ROLE_KEYWORDS.items():
        for kw in kws:
            if any(t == kw or (len(kw) >= 4 and t.startswith(kw)) for t in toks):
                return role
    return None


def _catalog(entries: list[tuple[str, int]]) -> pd.DataFrame:
    """Group (path, size) pairs; a .gdb folder counts as one entry."""
    rows: dict[str, dict] = {}
    for path, size in entries:
        p = path.replace("\\", "/")
        m = re.search(r"^(.*?\.gdb)(/|$)", p, flags=re.IGNORECASE)
        if m:
            key = m.group(1)
            row = rows.setdefault(key, {"path": key, "kind": "geodatabase", "role": None, "size_bytes": 0, "n_files": 0})
        else:
            if p.endswith("/"):
                continue
            key = p
            row = rows.setdefault(key, {"path": key, "kind": _kind(p), "role": _role(p), "size_bytes": 0, "n_files": 0})
        row["size_bytes"] += int(size)
        row["n_files"] += 1
    cols = ["path", "kind", "role", "size_bytes", "n_files"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows.values(), columns=cols).sort_values(["kind", "path"], ignore_index=True)


def catalog_zip(zip_path: str | Path) -> pd.DataFrame:
    """List what's in a RECOVER zip without extracting it.

    Columns: path, kind (raster / geodatabase / vector / layerfile /
    report / toolbox / other), role (severity / dem / slope / ndvi /
    landfire / perimeter / soils, or None), size_bytes, n_files.
    """
    with zipfile.ZipFile(zip_path) as zf:
        return _catalog([(i.filename, i.file_size) for i in zf.infolist()])


def catalog_dir(package_dir: str | Path) -> pd.DataFrame:
    """Same as ``catalog_zip`` for an already extracted package."""
    root = Path(package_dir)
    return _catalog([(str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file()])


def extract_members(
    zip_path: str | Path,
    dest: str | Path,
    roles: tuple[str, ...] = ("severity", "dem"),
    include_geodatabases: bool = False,
) -> list[Path]:
    """Extract only the rasters with the given roles (and optionally the
    geodatabase), instead of all 36 GB. Returns the extracted paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            in_gdb = re.search(r"\.gdb/", name, flags=re.IGNORECASE) is not None
            wanted = (in_gdb and include_geodatabases) or (
                not in_gdb and _kind(name) == "raster" and _role(name) in roles
            )
            if wanted:
                out.append(Path(zf.extract(info, dest)))
    return out


def load_recover_package(zip_path: str | Path, extract_to: str | Path) -> Path:
    """Extract a whole RECOVER package zip.

    For large packages (Wapiti: 36.1 GB) prefer ``catalog_zip`` +
    ``extract_members``.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def find_rasters(package_dir: str | Path, role: str) -> list[Path]:
    """Rasters in an extracted package whose file name matches ``role``."""
    root = Path(package_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and _kind(p.name) == "raster" and _role(p.name) == role)


def find_dnbr_raster(package_dir: str | Path) -> Path | None:
    """Best guess at the burn-severity raster, preferring a continuous dNBR.

    Returns None when there isn't one. In that case the severity may be a
    web service; check the .lyrx files and the HTML report.
    """
    candidates = find_rasters(package_dir, "severity")
    if not candidates:
        return None
    for pref in ("dnbr", "rdnbr", "barc", "sbs"):
        for c in candidates:
            if pref in _tokens(c.name) or any(t.startswith(pref) for t in _tokens(c.name)):
                return c
    return candidates[0]
