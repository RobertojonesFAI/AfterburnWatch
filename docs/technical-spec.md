# Afterburn Watch -- Technical Spec

Living reference. The box-by-box mapping from the whiteboards to the code
is in [`architecture.md`](architecture.md). The team's narrative is the
white paper (v0.3, led by Ashraf). This spec is the technical companion
to it.

## 1. Mission

Post-wildfire debris flows are an acute hydro-geomorphic threat across
the Western United States. They can damage critical infrastructure,
water resources and communities within minutes of intense rainfall.

Afterburn Watch is a **satellite-driven feedback loop** for post-fire
debris-flow risk:

1. identify debris flows that actually happened, from Sentinel-2 and
   Landsat imagery (**Model 1**);
2. add them to the observation record (**Actual DF Observations**);
3. score the official USGS prediction against them, then recalibrate the
   USGS likelihood model (**DF Prediction**);
4. publish the updated risk map (**Map Risk**), and repeat after the next
   storm.

Theory of change (white paper): if predictions are compared against
actual outcomes, *including* predicted debris flows that never happened,
assessments get measurably more accurate across fires. Today agencies
only learn from confirmed events; over-predictions vanish once the report
is filed.

## 2. Hackathon context

- **Event:** R-CON 2026 Innovation Village, Boise Centre, Oct 12-15, 2026.
- **Live Showcase:** October 13, 2026 (10 min pitch + 5-10 min Q&A).
  Judging emphasizes usability and government adoption potential. A
  finished product isn't required; showing the concept is feasible is.
  The team agreed on Sept 23 that the white paper stays an outline of the
  approach and its impact, not a complete report.
- **Challenge:** 01 (Wildfire Resilience), Problem Statement 2.
- **Case study:** 2024 Wapiti Fire, central Idaho (section 5).

## 3. Team & roles

| Member | Role | Focus |
|---|---|---|
| Troy Jenks | Technical build lead | Debris-flow "actuals" data, imagery access, ArcGIS Online + CCDC change detection |
| Roberto Jones | Data integration | Satellite imagery into the pipeline: sensors, masking, Model 1, the loop, this repo |
| Md Ashrafuzzaman (Ashraf) | Strategy | Feasibility and agency integration; white paper and presentation lead |

Advisors: Dr. Keith Weber (ISU GIS TReC), Dr. Ashley Bosa (BSU
Resilience Institute). Lemhi HPC support: Michael Ennis (ISU).

## 4. Architecture

See [`architecture.md`](architecture.md). In short:

| Stage | Module |
|---|---|
| Sentinel-2 / Landsat ingest, cloud & smoke masking | `sensors.py`, `ingest.py`, `masking.py` |
| Spectral indices, burn severity | `indices.py`, `severity.py` |
| Authoritative inputs (RECOVER package) | `recover.py` |
| DF Actual (USGS inventory + new observations) | `observations.py` |
| Model 1 -- DF identification, single-sat params | `identify.py` |
| DF Prediction -- official USGS assessment as baseline | `usgs_assessment.py` |
| DF Prediction -- USGS M1 + recalibration | `predict.py` |
| Map Risk | `risk.py` |
| The loop | `loop.py` |

## 5. Case study: Wapiti Fire

- **Fire:** 2024, central Idaho, ~129,000 acres. The BAER report (Oct 2024)
  rates 42% of the area at high (10%) or moderate (32%) soil burn
  severity, with the largest change in the west and south of the fire.
  SH-21 ("Avalanche Alley") and the Grandjean Road (FS 524) were flagged
  at risk.
- **USGS assessment exists:** BAER passed the soil burn severity to the
  USGS Landslide Hazards Program for likelihood/volume modeling. The 2024
  assessments are in data release doi:10.5066/P13GTMAX. There is also a
  USGS dashboard:
  https://www.arcgis.com/apps/dashboards/c09fa874362e48a9afe79432f2efe6fe
- **Test event:** in late August 2025, heavy rain on the burn scar
  triggered mudslides that closed SH-21 between MP 93 and 105 (north of
  Grandjean to Banner Summit), reported Aug 27, 2025 (Idaho News 6 /
  KBOI, KIVI). This gives the loop a real, dated storm on an Idaho fire.
- **Demo storyline:** USGS prediction per basin -> Aug 2025 storm ->
  Sentinel-2 / Landsat before and after -> which basins actually ran ->
  hit / miss / false-alarm map in ArcGIS Online -> what recalibration
  would change.
- **Imagery windows:** the loop needs before/after the **storm** (Aug
  2025), not only before/after the fire (2024).
  `ingest.storm_windows` and `scripts/find_storm_scenes.py` handle this.

## 6. Domain notes (Keith Weber briefing, Sept 18)

- **Who uses this:** Forest Service, BLM, National Park Service. USACE and
  the National Weather Service are supporting consumers. The USGS
  Landslide Hazards Program produces a likelihood and a volume product
  per fire and hands both to the land manager.
- **Burn severity:** dNBR measures how much vegetation was consumed. Fire
  *intensity* (heat released) is more accurate, but thermal sensors give
  only intermittent data. Field checks of soil hydrophobicity confirm
  burn intensity.
- **Clouds and smoke:** imagery must be cloud-free and smoke-free. Smoke
  from *other* nearby fires can obscure a burn for months. Landsat's QA
  bands are good at reporting what's in a pixel.
- **Rainfall thresholds:** debris flows can start at intensities as low as
  ~0.25 in/hr, depending on soil type, slope, burn severity/intensity and
  hydrophobicity. The thresholds were trained on California, and Idaho
  soils differ. Silica sand increases hydrophobicity. Fine ash seals the
  surface, but gentle rain can break that crust.
- **Sensors:** Sentinel-2 has the better spatial resolution, Landsat the
  better spectral resolution, and Landsat 9 is preferred over 8.
- **LiDAR:** Idaho has statewide LiDAR, which is a better input than the
  National Elevation Dataset most assessments use. Keith included a LiDAR
  DEM with the Wapiti package. New *post-event* LiDAR is set aside (slow
  and expensive).

## 7. Formula reference

| Quantity | Formula | Notes |
|---|---|---|
| NBR | `(NIR - SWIR2) / (NIR + SWIR2)` | Sentinel-2: `(B08 - B12)/(B08 + B12)`. Landsat 8/9: `(B5 - B7)/(B5 + B7)`. Pick bands by wavelength role, never by number (Keith, Sept 18). |
| dNBR | `(NBR_pre - NBR_post) * 1000` | Burn severity (pre-fire -> post-fire). The same form across a storm is Model 1's `event_dnbr`. |
| NDVI | `(NIR - Red) / (NIR + Red)` | |
| RdNBR | `dNBR / sqrt(abs(NBR_pre))` | Miller & Thode 2007 |
| BAER classes | <100, 100-270, 270-660, >=660 | Unburned/low, low, moderate, high |
| USGS M1 | `p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))` | Staley et al. 2017. 15-min: B=-3.63, Ct=0.41, Cf=0.67, Cs=0.70. Cross-check against `pfdf.models.staley2017.M1`. |
| Rainfall threshold | `R_p = (logit(p) - B) / (Ct*T + Cf*F + Cs*S)` | Rainfall at which likelihood reaches p |
| Recalibration | `min -loglik(theta) + (lambda/2) * abs(theta - theta_Staley)^2` | MAP logistic regression, prior at the published values. Default `lambda` = 2. |
| USGS likelihood classes | `BP_Legend` "0-20%" ... "80-100%" | Scored at the class midpoint by default; low/high ends as a sensitivity check |

T = proportion of upslope area burned at moderate/high severity on slopes
>= 23 degrees; F = mean dNBR/1000 upslope; S = mean soil KF-factor
upslope; R = peak rainfall accumulation (mm) over the duration.

## 8. Data sources

| Data | Source | Used for | Status |
|---|---|---|---|
| Observed debris flows (0/1) | USGS Runoff-Generated Postfire Debris-Flow Inventory, doi:10.5066/P1R9VXC4 (8,981 observations, 1,140 debris flows, 80 fires; none in Idaho) | Model 1 labels; seed of the observation record | Columns to confirm |
| Wapiti RECOVER package + LiDAR DEM | Keith Weber, https://giscenter-sl.isu.edu/AOC/TEMP/ `WapitiFire_2024_IDBOF_000683.zip` (36.1 GB) | Severity, soils, DEM -> T, F, S | To download on Lemhi and catalog (`recover.catalog_zip`) |
| RECOVER package layout | https://fsapps.nwcg.gov/RECOVER/GettingFamiliarWithRECOVER.pdf | File geodatabase (perimeter, Fires1950_Present, Soils_SSURGO, SMA, WBD/HU12) + ~19 GeoTIFFs (NDVI_Median, LANDFIRE EVT/BPS/EVC/FVT, DEM, slope) + .lyrx, HTML report, toolbox | Severity may be a web service, not a TIF -- confirm |
| Official USGS assessment | 2024 data release, doi:10.5066/P13GTMAX | Baseline prediction to score (`usgs_assessment.py`) | To download |
| USGS output schema | BigRockFire example (team's `Resources/Source Code/PostFireDebrisFlowOutput/`) | `BP_Legend`, `BV_Legend`, `BCH_Legend`, `fire_id`, `assessment`, `version`, `start_date` -- classes only, **no T, F, S** | Known |
| Sentinel-2 L2A | Element84 Earth Search (STAC, public) | Model 1 features, dNBR | Code written, not run live |
| Landsat 8/9 C2 L2 | Microsoft Planetary Computer (STAC, signed URLs) | Model 1 features, dNBR, cross-sensor confirmation | Code written, not run live |
| ArcGIS Online | Boise State organization (Troy; 5,000 credits + desktop software) | Map Risk destination; CCDC change detection | Troy connecting (~Sept 29) |
| Per-basin storm rainfall | MRMS (Debris Flow Hunters) or gauges | R for each observation | Source still to settle (white paper open item) |
| NASA Global Landslide Catalog | https://data.nasa.gov/dataset/global-landslide-catalog-export | Low priority: one-time 2016 export, news-based, coarse locations, rain-triggered (not post-fire), predates most of Sentinel-2 | Parked |

## 9. Decision log

- **Sept 11 (kickoff):** don't *produce* a competing authoritative dNBR.
  RECOVER remains the authoritative reference for prediction inputs.
- **Sept 18-19 (pivot):** there is no baseline for how long severity data
  takes to produce, so "faster severity maps" is not the pitch. Adopt the
  whiteboard feedback loop: Sentinel-2 and Landsat identify debris flows
  that actually occurred, and those observations feed back into the
  prediction. We still compute dNBR ourselves because Model 1 needs it.
- **Sept 23:** Keith shared the Wapiti package (RECOVER + LiDAR DEM). The
  team aligned on direction, sourcing and outline. The white paper stays
  an outline of approach and impact.
- **Sept 25:** Troy is working in ArcGIS Online (Boise State org). Next
  steps: view Wapiti before/after via the API, then CCDC change detection.
  This gives Map Risk a real destination and a possible second detector.
- **Sept 28:** demo storyline set around the Aug 2025 SH-21 mudslides.
  The official USGS Wapiti assessment is the baseline to score. The NASA
  Global Landslide Catalog is parked (low priority).

## 10. Roadmap to Oct 13

| Dates | Work |
|---|---|
| Sept 29 - Oct 2 | Download Keith's zip directly on Lemhi and catalog it (`recover.catalog_zip`). Get the USGS inventory CSV + README and confirm columns. Get the official USGS Wapiti layer (2024 data release). First live Sentinel-2 / Landsat scene search around the storm (`scripts/find_storm_scenes.py`). **Go / no-go:** are the Aug 2025 debris-flow scars visible? Align with Troy on the storm window, CCDC output format and the ArcGIS Online layer. |
| Oct 5 - 7 | Wapiti basins (USGS layer, or pfdf on Lemhi for T, F, S). Hand-label a few positive/negative basins. Model 1 trained on Sentinel-2-era inventory fires (or a simple detector as fallback). One loop iteration on Wapiti. GeoJSON -> ArcGIS Online + Streamlit. |
| Oct 8 - 9 | Code freeze (tag). Update the "what is real" table. Screenshots and 3 visuals for Ashraf. |
| Oct 12 - 13 | Rehearsal and showcase. |

## 11. Open questions

- **Wapiti ground truth.** White paper v0.3 says the ground truth will come
  from the RECOVER package. But per the RECOVER tutorial, the package
  holds fire and terrain layers (severity, soils, DEM), not observed
  debris flows. For Wapiti, ground truth likely has to come from the
  satellite identification, checked against the Aug 2025 SH-21 reports
  (and any field/ITD records). Troy's call.
- Is Wapiti's burn severity in the package as a GeoTIFF, or only as a web
  service?
- The official USGS layer has classes but no T, F, S. Recalibration
  needs our own pfdf run on Lemhi (LiDAR DEM + severity + soil KF).
- The USGS inventory's exact columns: coordinates for every observation?
  T/F/S/R? (`observations.load_inventory` fails loudly until mapped.)
- The exact storm date(s) and per-basin rainfall for Aug 2025: MRMS or
  gauges?
- CCDC output format, so it can be fused with the two sensors.
- Pixel labels for Model 1: the inventory is basin-level, so training
  labels a basin's channel pixels with its outcome. That is noisy.
- Trusted-gate threshold (0.9) and `prior_strength` (2) were tuned only
  on synthetic data.
