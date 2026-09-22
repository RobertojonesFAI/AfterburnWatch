"""Sensor definitions for Sentinel-2 and Landsat 8/9.

Whiteboard: the "Sentinel" boxes and the "? LandSat ?" arrow.

Band *numbers* differ between sensors, so everything in this package refers
to bands by *role* ("nir", "swir2", "red", ...) and this module maps each
role to the right band for each sensor. This follows Keith Weber's Sept 18
guidance: use the correct NIR and SWIR bands and never rely on band numbers.

    Role    Sentinel-2 MSI         Landsat 8/9 OLI
    ------  ---------------------  ------------------
    red     B04  (665 nm, 10 m)    B4 (655 nm, 30 m)
    nir     B08  (842 nm, 10 m)    B5 (865 nm, 30 m)
    swir1   B11 (1610 nm, 20 m)    B6 (1609 nm, 30 m)
    swir2   B12 (2190 nm, 20 m)    B7 (2201 nm, 30 m)

    NBR = (NIR - SWIR2) / (NIR + SWIR2)
      Sentinel-2:  (B08 - B12) / (B08 + B12)
      Landsat 8/9: (B5  - B7)  / (B5  + B7)

Trade-off noted by Keith: Sentinel-2 has the better *spatial* resolution
(10-20 m), Landsat the better *spectral* resolution; Landsat 9 is preferred
over Landsat 8. Landsat's QA bands are also the more informative ones for
cloud masking (see masking.py).

STAC asset keys below match Element84 Earth Search v1 (Sentinel-2 L2A) and
Microsoft Planetary Computer (Landsat Collection 2 Level-2). Both catalogs
name assets by common name, not band number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BandInfo:
    """One spectral band of one sensor."""

    asset: str  # STAC asset key
    band_id: str  # sensor's own band label, for humans
    center_nm: float | None
    resolution_m: float


@dataclass(frozen=True)
class Sensor:
    """Everything the pipeline needs to know about one sensor/product."""

    name: str
    platforms: tuple[str, ...]
    catalog_url: str
    collection: str
    bands: dict[str, BandInfo]
    qa_asset: str  # cloud/quality band used by masking.py
    # Default reflectance scaling (DN * scale + offset). Prefer the values in
    # each STAC item's raster:bands metadata when present -- see
    # scale_offset() -- because Sentinel-2 processing baseline 04.00+
    # introduced a -0.1 offset. An additive offset does NOT cancel out of
    # NBR/NDVI, so getting it wrong biases every index.
    default_scale: float = 1.0
    default_offset: float = 0.0
    requires_signing: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def asset_for(self, role: str) -> str:
        try:
            return self.bands[role].asset
        except KeyError as exc:
            raise KeyError(
                f"{self.name} has no band role {role!r}; "
                f"available: {sorted(self.bands)}"
            ) from exc


SENTINEL2_L2A = Sensor(
    name="sentinel-2-l2a",
    platforms=("sentinel-2a", "sentinel-2b", "sentinel-2c"),
    catalog_url="https://earth-search.aws.element84.com/v1",
    collection="sentinel-2-l2a",
    bands={
        "blue": BandInfo("blue", "B02", 490, 10),
        "green": BandInfo("green", "B03", 560, 10),
        "red": BandInfo("red", "B04", 665, 10),
        "nir": BandInfo("nir", "B08", 842, 10),
        "swir1": BandInfo("swir16", "B11", 1610, 20),
        "swir2": BandInfo("swir22", "B12", 2190, 20),
        "aot": BandInfo("aot", "AOT", None, 20),
    },
    qa_asset="scl",
    default_scale=0.0001,
    default_offset=0.0,
    notes=(
        "Scene Classification Layer (SCL) is the cloud/shadow mask.",
        "AOT (aerosol optical thickness) can flag heavy smoke; SCL has no smoke class.",
    ),
)

LANDSAT_C2_L2 = Sensor(
    name="landsat-c2-l2",
    platforms=("landsat-9", "landsat-8"),  # 9 preferred per Keith
    catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    collection="landsat-c2-l2",
    bands={
        "blue": BandInfo("blue", "B2", 482, 30),
        "green": BandInfo("green", "B3", 562, 30),
        "red": BandInfo("red", "B4", 655, 30),
        "nir": BandInfo("nir08", "B5", 865, 30),
        "swir1": BandInfo("swir16", "B6", 1609, 30),
        "swir2": BandInfo("swir22", "B7", 2201, 30),
        "qa_aerosol": BandInfo("qa_aerosol", "SR_QA_AEROSOL", None, 30),
    },
    qa_asset="qa_pixel",
    default_scale=0.0000275,
    default_offset=-0.2,
    requires_signing=True,  # Planetary Computer asset URLs must be signed
    notes=(
        "QA_PIXEL bit flags are the cloud/shadow mask.",
        "SR_QA_AEROSOL bits 6-7 report aerosol level (high aerosol ~ smoke).",
    ),
)

SENSORS: dict[str, Sensor] = {
    "sentinel2": SENTINEL2_L2A,
    "landsat": LANDSAT_C2_L2,
}


def get_sensor(key: str) -> Sensor:
    """Look up a sensor by short key ("sentinel2" or "landsat")."""
    try:
        return SENSORS[key]
    except KeyError as exc:
        raise KeyError(f"unknown sensor {key!r}; choose from {sorted(SENSORS)}") from exc


def scale_offset(item_assets: dict, sensor: Sensor, role: str) -> tuple[float, float]:
    """Reflectance scale/offset for a band, preferring STAC item metadata.

    Parameters
    ----------
    item_assets : the ``assets`` mapping of a STAC item as a plain dict
        (``item.to_dict()["assets"]``).
    sensor, role : which band.

    Falls back to the sensor's defaults when the item doesn't carry
    ``raster:bands`` scale/offset.
    """
    asset = item_assets.get(sensor.asset_for(role), {})
    bands = asset.get("raster:bands") or []
    if bands:
        b = bands[0]
        return float(b.get("scale", sensor.default_scale)), float(
            b.get("offset", sensor.default_offset)
        )
    return sensor.default_scale, sensor.default_offset
