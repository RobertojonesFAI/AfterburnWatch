"""NASA RECOVER data package retrieval -- authoritative inputs for DF Prediction.

Project Framework: "USFS Server -> RECOVER package". A RECOVER package for
a fire bundles the authoritative dNBR, fire perimeter, roads and soils,
produced by the responsible agency (USFS for USDA land, USGS for DOI land).

Role in the loop:
  * Its dNBR and soils are the inputs wildcat/pfdf use to compute each
    basin's T, F, S for DF Prediction (predict.py), the same inputs
    federal stakeholders already trust.
  * Its dNBR is the reference to cross-check the dNBR we compute ourselves
    from Sentinel-2 / Landsat (indices.py), and it can feed Model 1 as the
    ``fire_dnbr`` feature (identify.py).

Manual process today (per our domain advisor, describing the ArcGIS
dashboard):
  1. Open the RECOVER ArcGIS dashboard:
     https://www.arcgis.com/apps/dashboards/68f43718f2474f369c587a97d32cd0cf
  2. Click the target fire's polygon (default: 2024 Wapiti Fire, central
     Idaho). For a past fire, use the hamburger menu -> bulk download ->
     previous years to find it in the archive.
  3. Click "Download recover data package" (the button is labeled "view"
     but actually downloads a zip, ~3 GB typical). The zip contains dNBR,
     fire perimeter, roads, soils, and links to source imagery.

TODO: this module is currently a stub. Automating step 1-3 needs the
underlying ArcGIS REST/feature-service endpoint the dashboard calls (not
yet confirmed -- check the ISU GIS Center's "RECOVER Web Services
Reference" PDF, linked from
https://giscenter.isu.edu/research/Techpg/nasa_RECOVER2/index.htm, for a
documented API before assuming one). Until then, download manually and
point `load_recover_package` at the local zip.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

RECOVER_DASHBOARD_URL = (
    "https://www.arcgis.com/apps/dashboards/68f43718f2474f369c587a97d32cd0cf"
)


def load_recover_package(zip_path: str | Path, extract_to: str | Path) -> Path:
    """Extract a manually-downloaded RECOVER data package zip.

    Parameters
    ----------
    zip_path : path to the downloaded RECOVER package zip.
    extract_to : directory to extract into (created if missing).

    Returns
    -------
    Path to the extraction directory. Look inside for the dNBR raster,
    fire perimeter, roads, and soils layers -- exact filenames vary by
    package and haven't been catalogued here yet (see TODO above; this
    should be filled in against a real downloaded Wapiti Fire package).
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def find_dnbr_raster(package_dir: str | Path) -> Path | None:
    """Best-effort search for the dNBR raster inside an extracted package.

    TODO: replace this heuristic once we've inspected a real Wapiti Fire
    RECOVER package and know its actual file naming convention.
    """
    package_dir = Path(package_dir)
    candidates = list(package_dir.rglob("*dnbr*.tif")) + list(
        package_dir.rglob("*dNBR*.tif")
    )
    return candidates[0] if candidates else None
