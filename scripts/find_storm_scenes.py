"""List Sentinel-2 and Landsat scenes before and after a storm over a fire.

This is the go/no-go check for the Wapiti demo: are there clear enough
scenes on both sides of the late-August-2025 storm to see where debris
flows ran?

    pip install -e ".[imagery]"            # plus ".[gis]" for --perimeter
    python scripts/find_storm_scenes.py \\
        --storm-date 2025-08-27 \\
        --bbox -115.6 43.9 -115.0 44.4 \\
        --out wapiti_scenes.csv

--bbox is min_lon min_lat max_lon max_lat (WGS84). Or pass --perimeter
with the fire perimeter (shapefile / GeoJSON / geodatabase layer from
the RECOVER package) and the bbox is taken from it. The bbox above is
only an illustration: take the real one from the Wapiti perimeter.

Storm date: SH-21 closed by mudslides on the Wapiti burn scar, reported
Aug 27, 2025. Confirm the exact date from rainfall records.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from afterburn_watch.ingest import find_scenes, storm_windows, summarize_scenes


def bbox_from_perimeter(path: str, layer: str | None) -> list[float]:
    import geopandas as gpd  # gis extra

    gdf = gpd.read_file(path, layer=layer).to_crs(4326)
    return [float(v) for v in gdf.total_bounds]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storm-date", required=True, type=date.fromisoformat)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--perimeter", help="fire perimeter file (needs the gis extra)")
    ap.add_argument("--layer", help="layer name inside a geodatabase")
    ap.add_argument("--before-days", type=int, default=60)
    ap.add_argument("--after-days", type=int, default=45)
    ap.add_argument("--max-cloud", type=float, default=40.0, help="scene-level cloud cover %% ceiling")
    ap.add_argument("--sensors", nargs="+", default=["sentinel2", "landsat"])
    ap.add_argument("--out", help="write the combined table to this CSV")
    args = ap.parse_args()

    bbox = args.bbox or bbox_from_perimeter(args.perimeter, args.layer)
    before, after = storm_windows(args.storm_date, args.before_days, args.after_days)
    print(f"bbox {bbox}\nbefore {before[0]} .. {before[1]}   after {after[0]} .. {after[1]}\n")

    tables = []
    for sensor in args.sensors:
        for label, (start, end) in (("before", before), ("after", after)):
            items = find_scenes(sensor, bbox, start, end, max_cloud_cover=args.max_cloud)
            t = summarize_scenes(items).assign(sensor=sensor, window=label)
            tables.append(t)
            print(f"{sensor:9s} {label:6s}: {len(t):3d} scenes"
                  + (f", clearest {t['cloud_cover'].min():.0f}% cloud" if len(t) else ""))

    all_scenes = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if len(all_scenes):
        print()
        print(all_scenes.sort_values(["sensor", "datetime"]).to_string(index=False))
    if args.out:
        all_scenes.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    # Scene-level cloud % is only a first filter: smoke and cloud over the
    # burn itself still need checking with masking.py on the loaded pixels.


if __name__ == "__main__":
    main()
