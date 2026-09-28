# Progress Log

Track builder/research steps here (migrated from the team's
`Afterburn_Watch_Progress.xlsx`). Update with each PR.

## Sept 23-28 -- Wapiti data, a real storm, and the demo storyline

- **Wapiti package (Keith, Sept 23):** full RECOVER package + LiDAR DEM,
  `WapitiFire_2024_IDBOF_000683.zip`, 36.1 GB, at
  https://giscenter-sl.isu.edu/AOC/TEMP/. Download it straight to the
  team's shared directory on Lemhi.
- **Troy in ArcGIS Online (Sept 25):** Boise State organization, 5,000
  credits + desktop software. Next steps: Wapiti before/after via the
  API, then CCDC change detection.
- **A real, dated test storm:** in late August 2025, heavy rain on the
  Wapiti burn scar triggered mudslides that closed SH-21 (MP 93-105),
  reported Aug 27, 2025.
- **USGS already assessed Wapiti:** per the BAER report (Oct 2024); 2024
  assessments are in data release doi:10.5066/P13GTMAX. Its output has
  likelihood *classes* (`BP_Legend`), no T, F, S. So it can be scored
  but not recalibrated.
- **NASA Global Landslide Catalog:** parked (2016 one-time export,
  news-based, not post-fire).
- **White paper v0.3** (Ashraf / Troy): repo README and spec aligned
  with its roles and framing.

Code added:

- [x] `recover.py` -- `catalog_zip` (list a package without extracting 36 GB), `extract_members` (only severity/DEM rasters), `find_rasters`, better `find_dnbr_raster`
- [x] `usgs_assessment.py` -- read the official USGS assessment, score it hit/miss/false alarm against observations, spatial join of observations to USGS basins (`gis` extra)
- [x] `ingest.storm_windows` + `summarize_scenes`, and `scripts/find_storm_scenes.py` -- scene search before/after a storm
- [x] 56 unit tests

## Sept 22 -- repo rebuilt around the whiteboard loop

The Sept 18-19 whiteboards define the design: Sentinel-2 + Landsat ->
Model 1 identifies actual debris flows -> observations -> recalibrated
DF prediction -> risk map -> repeat. See `docs/architecture.md`.

- [x] `sensors.py`, `masking.py`, `indices.py`, `ingest.py` (not yet run live)
- [x] `observations.py`, `identify.py`, `predict.py`, `risk.py`, `loop.py`
- [x] `scripts/demo_synthetic_loop.py`, `app.py`
- `pfdf` moved to an optional extra (`.[pfdf]`)

## Next up (roadmap to Oct 13 -- see `docs/technical-spec.md` section 10)

**Sept 29 - Oct 2**
- [ ] Download Keith's Wapiti zip on Lemhi; run `recover.catalog_zip` and fill in real file names (is severity a GeoTIFF or only a web service?)
- [ ] Download `USGS_RG_PFDF_Inventory_v1.csv` + README (doi:10.5066/P1R9VXC4) and confirm the column mapping in `observations.load_inventory`
- [ ] Download the official USGS Wapiti assessment (2024 data release) and load it with `usgs_assessment.load_assessment`
- [ ] Run `scripts/find_storm_scenes.py` around the Aug 2025 storm (Sentinel-2 + Landsat)
- [ ] **Go / no-go:** are the Aug 2025 debris-flow scars visible in the imagery?
- [ ] Align with Troy: storm window, CCDC output format, ArcGIS Online layer

**Oct 5 - 7**
- [ ] Wapiti basins with T, F, S (pfdf on Lemhi) or the USGS basins for scoring only
- [ ] Hand-label a few positive / negative basins
- [ ] Train Model 1 on Sentinel-2-era inventory fires (or a simple detector as fallback)
- [ ] One loop iteration on Wapiti; GeoJSON -> ArcGIS Online + Streamlit

**Oct 8 - 9**
- [ ] Code freeze (tag), update the "what is real" table, screenshots + 3 visuals for Ashraf

**Also open**
- [ ] Per-basin storm rainfall: MRMS or gauges (white paper open item)
- [ ] Verify `STALEY2017_M1` against `pfdf.models.staley2017.M1.parameters()`
- [ ] Check SCL / QA_PIXEL masks over the Wapiti burn scar (burn scars can land in SCL class 2/3)

## Environment & tooling

- [x] Environment setup
- [x] Ran `pfdf`'s local test suite -- 8 local tests failed, apparently NumPy
  type issues in dependencies rather than `pfdf` logic. Considered safe to
  ignore.
- `pfdf` is **not on PyPI**: install with
  `--extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple`
  (see README). Requires Python >= 3.11. The full `pfdf` install hasn't
  been verified end-to-end on a clean machine yet -- please confirm on
  yours and note it here.

## `pfdf` tutorial walkthrough (`pfdf-main/tutorials/`)

- [x] 01 -- Start Here
- [x] 02 -- Raster Intro
- [x] 03 -- Download Data
- [x] 04 -- Preprocessing
- [ ] 05 -- Hazard Assessment (end-to-end; needed for basin T, F, S)
- [ ] 06-11 -- Raster properties/factories/metadata, parallel basins, parameter sweep

## Decisions

- **Sept 11:** RECOVER packages are the authoritative dNBR/soils source for
  the prediction inputs. Don't produce a competing dNBR product.
- **Sept 18-19:** pivot to the satellite feedback loop.
- **Sept 23:** white paper stays an outline of approach and impact.
- **Sept 25:** Troy on ArcGIS Online + CCDC; Map Risk goes to ArcGIS Online.
- **Sept 28:** demo built around the Aug 2025 SH-21 mudslides on Wapiti;
  score the official USGS assessment first. Full log in
  `docs/technical-spec.md` section 9.

## Notes / open ideas

- 3D visualization via Blender + [BlenderGIS](https://github.com/domlysz/BlenderGIS/)
  with OpenStreetMap basemaps.
- HLS (Harmonized Landsat Sentinel-2) as a single input for Model 1, with
  more frequent clear views.
- Personal tracking tabs: team sheet
  https://docs.google.com/spreadsheets/d/1bt3khGChtJN0ZtciiSZbD0SmtfXGdAmyLdU0RPFgcLI/edit
