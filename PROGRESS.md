# Progress Log

Track builder/research steps here (migrated from the team's
`Afterburn_Watch_Progress.xlsx`). Update with each PR.

## Sept 22 -- repo rebuilt around the whiteboard loop

The Sept 18-19 whiteboards define the design: Sentinel-2 + Landsat ->
Model 1 identifies actual debris flows -> observations -> recalibrated
DF prediction -> risk map -> repeat. See `docs/architecture.md`.

- [x] `sensors.py` -- Sentinel-2 (B08/B12) and Landsat 8/9 (B5/B7) band maps by role
- [x] `masking.py` -- SCL, QA_PIXEL, aerosol/smoke masks, multi-scene composite
- [x] `indices.py` -- NBR, dNBR, NDVI, RdNBR (`severity.py` keeps BAER classes)
- [x] `ingest.py` -- STAC search + loading for both sensors (**not yet run live**)
- [x] `observations.py` -- DF Actual schema, USGS inventory loader, dedupe, trusted gate
- [x] `identify.py` -- Model 1 (random forest, grouped CV by fire), basin positives/negatives, sensor fusion
- [x] `predict.py` -- USGS M1 (Staley 2017) + MAP recalibration
- [x] `risk.py` -- likelihood classes, hit/miss/false alarm, GeoJSON
- [x] `loop.py` -- one-iteration orchestrator with guardrails, save/load state
- [x] `scripts/demo_synthetic_loop.py` -- whole loop on synthetic data
- [x] `app.py` -- Streamlit view of the loop
- [x] 41 unit tests
- `pfdf` moved to an optional extra (`.[pfdf]`), so the core installs and tests run anywhere

## Next up

- [ ] Download `USGS_RG_PFDF_Inventory_v1.csv` + README (doi:10.5066/P1R9VXC4) and confirm the column mapping in `observations.load_inventory`
- [ ] First live Sentinel-2 + Landsat pull over the Wapiti fire (`ingest.find_scenes` / `load_bands`)
- [ ] Check SCL / QA_PIXEL masks over the Wapiti burn scar (burn scars can land in SCL class 2/3)
- [ ] Train Model 1 on real inventory sites (grouped CV by fire)
- [ ] Wire basin T, F, S from wildcat/pfdf runs on Lemhi
- [ ] Get per-basin storm rainfall from MRMS (coordinate with Debris Flow Hunters)
- [ ] Cross-check our dNBR against RECOVER's for Wapiti
- [ ] Verify `STALEY2017_M1` against `pfdf.models.staley2017.M1.parameters()`
- [ ] Push the risk layer to ArcGIS Online

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
- **Sept 18-19:** pivot to the satellite feedback loop (see
  `docs/technical-spec.md` section 8).

## Notes / open ideas

- 3D visualization via Blender + [BlenderGIS](https://github.com/domlysz/BlenderGIS/)
  with OpenStreetMap basemaps.
- HLS (Harmonized Landsat Sentinel-2) as a single input for Model 1, with
  more frequent clear views.
- Personal tracking tabs: team sheet
  https://docs.google.com/spreadsheets/d/1bt3khGChtJN0ZtciiSZbD0SmtfXGdAmyLdU0RPFgcLI/edit
