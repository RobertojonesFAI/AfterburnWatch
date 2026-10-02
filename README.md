# Afterburn Watch

**Afterburn Watch empowers emergency managers to evaluate post-fire risks within hours rather than weeks.**

Built for the R-CON 2026 Innovation Village hackathon (Boise Centre,
Oct 12-15, 2026) -- Challenge 01: Wildfire Resilience, Problem Statement 2
(rapid post-fire intelligence for emergency managers, utilities, and
recovery teams). Live Showcase: **October 13, 2026**.

## The problem

After a wildfire burns steep terrain, the risk of debris flows spikes for
months. A 147-year inventory (1879-2025) found 324 wildfires in the
western US that led to 1,062 damaging post-fire flow events. Losses over
the last decade average about $391 million a year, and deaths have gone
from about one a year historically to nearly six a year
(Thomas, Kostelnik & Bombard, 2026).

USGS publishes a likelihood and volume assessment for each fire, and land
managers (Forest Service, BLM, National Park Service; USACE and the
National Weather Service downstream) act on it. But each assessment is a
one-shot: nothing records whether the predicted debris flows actually
happened. The observations the models learn from are scarce: the USGS
inventory has 1,140 confirmed debris flows across 80 burned areas, and
**none of them are in Idaho**.

## The idea: a feedback loop that learns from what actually happened

Afterburn Watch adds a memory to the existing USGS tools, using
Sentinel-2 and Landsat imagery. It studies predicted debris flows that
never occurred, alongside the ones that did:

```
Identify by satellite  ->  Actual DF observations  ->  DF prediction  ->  Map risk
        ^                                                                    |
        +-------------------------- next storm ------------------------------+
```

1. **Identify by satellite (Model 1).** A classifier trained on the USGS
   inventory ("DF Actual") learns what a fresh debris flow looks like in
   one sensor's before/after imagery. After a storm it scans cloud-masked
   Sentinel-2 and Landsat scenes and reports, per basin, "debris flow",
   "clearly nothing" or "couldn't see".
2. **Actual observations.** Those calls join the observation inventory.
   Sentinel-2 and Landsat are fused: agreement raises confidence,
   disagreement goes to a person.
3. **DF prediction.** First, the official USGS assessment is scored
   against what happened. Then the USGS M1 likelihood model (Staley et al.
   2017) is recalibrated from trusted observations, per region if wanted.
   An update is kept only if it predicts held-out fires at least as well.
4. **Map risk.** Likelihood and rainfall thresholds per basin, with every
   observed basin marked hit / miss / false alarm, exported as GeoJSON
   for ArcGIS Online or the Streamlit app.

The design comes straight from the team whiteboards. See
[`docs/architecture.md`](docs/architecture.md) for the photos and a
box-by-box mapping to the code.

## Case study: the Wapiti Fire (2024, central Idaho)

- ~129,000 acres, recommended by Keith Weber as the case study. The BAER
  report (Oct 2024) found 42% of the area at high (10%) or moderate (32%)
  soil burn severity and flagged SH-21 ("Avalanche Alley") and the
  Grandjean Road as at risk. The soil burn severity went to the USGS
  Landslide Hazards Program for a debris-flow assessment (2024 data
  release, doi:10.5066/P13GTMAX).
- **A real, dated test event:** in late August 2025, heavy rain on the
  burn scar triggered mudslides that closed SH-21 between mileposts 93
  and 105 (north of Grandjean to Banner Summit), reported Aug 27, 2025.
- Demo storyline: the USGS prediction per basin -> the Aug 2025 storm ->
  Sentinel-2 / Landsat before and after -> which basins actually ran ->
  hit / miss / false-alarm map in ArcGIS Online -> what recalibration
  would change.
- Data: Keith shared the full RECOVER package plus the LiDAR DEM
  (36.1 GB) for processing on ISU's Lemhi HPC cluster
  ([`recover.py`](src/afterburn_watch/recover.py) catalogs it without
  extracting everything).

## Try it (no data or network needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,gis]"

pytest                                   # 56 tests
python scripts/demo_synthetic_loop.py    # whole loop on SYNTHETIC data
streamlit run src/afterburn_watch/app.py # look at the result
```

The demo uses **synthetic** scenes and observations
([`synthetic.py`](src/afterburn_watch/synthetic.py)). It shows the
mechanics working together, not real-world skill. In it, a made-up "Idaho"
region behaves differently from the published model, and after 8 storms
the recalibrated coefficients land close to that region's (made-up) truth.

With network access, check which scenes exist around the Wapiti storm:

```bash
pip install -e ".[imagery,gis]"
python scripts/find_storm_scenes.py --storm-date 2025-08-27 \
    --perimeter <Wapiti perimeter from the RECOVER package> --out wapiti_scenes.csv
```

For USGS `pfdf` (needed to compute T, F, S per basin):

```bash
# pfdf is NOT on PyPI -- it's on USGS's own package registry:
pip install -e ".[pfdf]" \
  --extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
```

## Repo structure

```
src/afterburn_watch/
  sensors.py          # Sentinel-2 & Landsat band maps (by role, never by band number)
  ingest.py           # STAC search + loading for both sensors, storm windows
  masking.py          # cloud / shadow / smoke masks, multi-scene composite
  indices.py          # NBR, dNBR, NDVI, RdNBR
  severity.py         # BAER burn-severity classes
  recover.py          # NASA RECOVER packages: catalog, selective extract
  usgs_assessment.py  # official USGS assessment as the baseline prediction
  observations.py     # DF Actual: the observation inventory
  identify.py         # Model 1: satellite debris-flow identification
  predict.py          # DF Prediction: USGS M1 + recalibration
  risk.py             # Map Risk: likelihood classes, outcomes, GeoJSON
  loop.py             # the feedback loop, one iteration at a time
  synthetic.py        # synthetic data for tests and the demo
  app.py              # Streamlit view of the loop
scripts/
  demo_synthetic_loop.py
  find_storm_scenes.py
docs/
  architecture.md     # whiteboards -> code
  technical-spec.md
  img/                # whiteboard photos, framework slide
tests/
```

## Where it plugs into the Project Framework

- **RECOVER packages** (USFS server) provide the authoritative severity,
  soils and terrain.
- **pfdf/wildcat on Lemhi HPC** turn those into per-basin T, F, S and the
  likelihood/volume models.
- **This package** adds the satellite loop on top.
- **The risk layer** goes to ArcGIS Online (Boise State organization) for
  the hazard assessment package and ESRI Dashboard.

Troy is also testing CCDC change detection in ArcGIS, which can join as a
second detector alongside the two sensors. Details in
[`docs/architecture.md`](docs/architecture.md#3-where-it-sits-in-the-project-framework).

## Team (Afterburn Watch)

| Name | Role |
|---|---|
| Troy Jenks | PhD student, Computing (AI emphasis), Boise State. Technical build lead; debris-flow "actuals" data and imagery access. |
| Roberto Jones | Undergraduate, Computer Science, Boise State. Data integration; connecting satellite imagery into the pipeline. |
| Md Ashrafuzzaman (Ashraf) | MS student, Civil and Environmental Engineering, Idaho State. Feasibility and agency-integration strategy; white paper and presentation lead. |

Advisors: Dr. Keith Weber (GIS Training and Research Center, Idaho State
University), Dr. Ashley Bosa (Resilience Institute, Boise State
University).

## Key references

- Graber, A.P., Gorr, A.N., Kean, J.W., Kostelnik, J., Rengers, F.K.,
  Selander, B.D., Thomas, M.A., 2026. *USGS runoff-generated postfire
  debris-flow inventory.* USGS data release, doi:10.5066/P1R9VXC4.
- USGS, *2024 post-wildfire debris-flow hazard assessments.* USGS data
  release, doi:10.5066/P13GTMAX.
- Thomas, M.A., Kostelnik, J. & Bombard, C.W., 2026. *Mounting damage and
  losses in the United States from post-wildfire flash floods and debris
  flows.* Communications Earth & Environment,
  doi:10.1038/s43247-026-03988-w.
- Staley, D.M. et al., 2017. *Prediction of spatially explicit rainfall
  intensity-duration thresholds for post-fire debris-flow generation in
  the western United States.* Geomorphology, 278, 149-162.
- Graber, A. et al., 2026. *Regional Models for Postfire Debris-Flow
  Likelihood and Rainfall Thresholds.* Earth Surface Processes and
  Landforms.
- Gartner, J.E. et al., 2014. *Empirical models for predicting volumes of
  sediment deposited by debris flows and sediment-laden floods in the
  transverse ranges of southern California.* Engineering Geology, 176, 45-56.
- Cannon, S.H. et al., 2010. *Predicting the probability and volume of
  postwildfire debris flows in the intermountain western United States.*
  GSA Bulletin, 122(1-2), 127-144.
- Key, C.H. & Benson, N.C. *Landscape Assessment (LA): sampling and
  analysis methods* (FIREMON) -- NBR / dNBR.
- Miller, J.D. & Thode, A.E., 2007. *Quantifying burn severity in a
  heterogeneous landscape with a relative version of the delta Normalized
  Burn Ratio (dNBR).* Remote Sensing of Environment, 109, 66-80.
- Zhou et al., 2025. *Multi-temporal landslide inventory mapping after
  wildfire and implications for postfire debris flow.* (Shared Articles
  folder.)
- Wapiti Fire Burned Area Report (BAER), Oct 2024.

## Acknowledgments

- [`pfdf`](https://code.usgs.gov/ghsc/lhp/pfdf) -- USGS post-fire
  debris-flow hazard assessment library (King, J. et al.), GPL-3.0. An
  optional dependency, not redistributed here.
- [NASA RECOVER](https://giscenter.isu.edu/research/Techpg/nasa_RECOVER2/index.htm) --
  post-fire decision-support data packages, ISU GIS Center.
- Sentinel-2 L2A via [Element84 Earth Search](https://earth-search.aws.element84.com/v1);
  Landsat C2 L2 via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).

## License

TBD -- to be decided before any public release. `pfdf` is GPL-3.0; confirm
license compatibility before distributing anything that bundles it.
