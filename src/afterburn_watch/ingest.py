"""Scene discovery and loading for Sentinel-2 and Landsat.

Whiteboard: the "Sentinel" inputs (to Model 1 and to the second stage) and
"? LandSat ?".

Sentinel-2 L2A comes from Element84 Earth Search (public, no auth needed).
Landsat C2 L2 comes from Microsoft Planetary Computer. Its asset URLs must be
signed with the ``planetary-computer`` package, which is free and needs no
account.

Install the imagery extras first:  pip install -e ".[imagery]"

``search_params`` and ``fire_windows`` are pure functions and are unit
tested. ``find_scenes`` and ``load_bands`` hit the network and have not been
run end-to-end yet. Try them on the Wapiti fire bbox first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

import numpy as np

from afterburn_watch.sensors import Sensor, get_sensor, scale_offset


@dataclass(frozen=True)
class Windows:
    """Date windows for one fire."""

    pre: tuple[date, date]  # vegetation baseline before the fire
    post_fire: tuple[date, date]  # burn severity (dNBR)
    post_event: tuple[date, date] | None = None  # after a storm: debris-flow evidence


def fire_windows(
    ignition: date,
    containment: date,
    pre_days: int = 60,
    post_fire_days: int = 60,
    storm_date: date | None = None,
    post_event_days: int = 45,
) -> Windows:
    """Standard pre / post-fire / post-event windows.

    Ideally the pre-fire scene matches the post-fire season (for example
    the same month a year earlier) so vegetation phenology doesn't show up
    as "burn". Here the pre window is simply the ``pre_days`` before
    ignition; adjust per fire. ``post_event`` is only set when a triggering storm date
    is known (from rainfall records / MRMS -- the Debris Flow Hunters team's
    area). That's the imagery Model 1 looks at for new debris flows.
    """
    pre = (ignition - timedelta(days=pre_days), ignition - timedelta(days=1))
    post_fire = (containment + timedelta(days=1), containment + timedelta(days=post_fire_days))
    post_event = None
    if storm_date is not None:
        post_event = (storm_date + timedelta(days=1), storm_date + timedelta(days=post_event_days))
    return Windows(pre=pre, post_fire=post_fire, post_event=post_event)


def search_params(
    sensor: Sensor,
    bbox: Sequence[float],
    start: date,
    end: date,
    max_cloud_cover: float = 30.0,
    limit: int = 50,
) -> dict:
    """Keyword arguments for ``pystac_client.Client.search`` for one sensor."""
    if len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    query: dict = {"eo:cloud_cover": {"lt": max_cloud_cover}}
    if sensor.name == "landsat-c2-l2":
        # Landsat 8/9 only: Landsat 7 has the SLC-off striping and
        # Landsat 5 a different sensor (TM).
        query["platform"] = {"in": list(sensor.platforms)}
    return {
        "collections": [sensor.collection],
        "bbox": list(bbox),
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "query": query,
        "limit": limit,
    }


def find_scenes(
    sensor_key: str,
    bbox: Iterable[float],
    start: date,
    end: date,
    max_cloud_cover: float = 30.0,
    limit: int = 50,
):
    """Search the sensor's STAC catalog. Returns a list of pystac Items."""
    from pystac_client import Client  # imagery extra

    sensor = get_sensor(sensor_key)
    modifier = None
    if sensor.requires_signing:
        import planetary_computer  # imagery extra

        modifier = planetary_computer.sign_inplace
    catalog = Client.open(sensor.catalog_url, modifier=modifier)
    params = search_params(sensor, list(bbox), start, end, max_cloud_cover, limit)
    return list(catalog.search(**params).items())


def load_bands(
    items,
    sensor_key: str,
    roles: Sequence[str] = ("red", "nir", "swir2"),
    bbox: Sequence[float] | None = None,
    resolution: float | None = None,
    crs: str | None = None,
):
    """Load bands (as reflectance) plus the QA band into numpy arrays.

    Returns ``(bands, qa, geobox)`` where ``bands[role]`` is a float32 array
    shaped (time, rows, cols), ``qa`` is the raw QA array (SCL or QA_PIXEL)
    with the same shape, and ``geobox`` is odc-geo's grid description
    (CRS + affine transform), which you need to turn pixels back into
    coordinates.

    ``resolution`` defaults to 20 m for Sentinel-2 (the SWIR2 native
    resolution, so NIR is resampled down rather than SWIR2 up) and 30 m
    for Landsat.
    """
    import odc.stac  # imagery extra

    sensor = get_sensor(sensor_key)
    if resolution is None:
        resolution = 20 if sensor.name.startswith("sentinel") else 30
    assets = [sensor.asset_for(r) for r in roles] + [sensor.qa_asset]
    ds = odc.stac.load(
        items,
        bands=assets,
        bbox=bbox,
        resolution=resolution,
        crs=crs or "utm",
        groupby="solar_day",
    )
    per_item = [
        {r: scale_offset(it.to_dict()["assets"], sensor, r) for r in roles} for it in items
    ]
    if any(p != per_item[0] for p in per_item[1:]):
        # e.g. Sentinel-2 scenes on both sides of the Jan 2022 processing
        # baseline change (offset 0 vs -0.1). One scale/offset per call
        # keeps the math honest, so load those groups separately.
        raise ValueError(
            "items use different reflectance scale/offset; call load_bands "
            "separately for each group (e.g. pre-fire vs post-fire)"
        )
    bands = {}
    for role in roles:
        scale, offset = per_item[0][role]
        dn = ds[sensor.asset_for(role)].values
        refl = dn.astype("float32") * scale + offset
        refl[dn == 0] = np.nan  # 0 is nodata in both products
        bands[role] = refl
    qa = ds[sensor.qa_asset].values
    return bands, qa, ds.odc.geobox
