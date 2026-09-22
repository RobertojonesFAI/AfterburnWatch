# Afterburn Watch

**Afterburn Watch empowers emergency managers to evaluate post-fire risks within hours rather than weeks.**

Built for the R-CON 2026 Innovation Village hackathon (Boise, Idaho) --
Challenge 01: Wildfire Resilience, Problem Statement 2 (rapid post-fire
intelligence for emergency managers, utilities, and recovery teams).
Live Showcase: **October 13, 2026**.

## The idea: a feedback loop that learns from what actually happened

USGS post-fire debris-flow models predict where debris flows are *likely*
after a fire. But the observations those models learn from are scarce:
the USGS inventory has 1,140 confirmed debris flows across 80 burned
areas, and **none of them are in Idaho**. After every storm, nobody
systematically checks where debris flows actually ran and feeds that
back into the model.

Afterburn Watch closes that loop with Sentinel-2 and Landsat imagery:

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
3. **DF prediction.** The USGS M1 likelihood model (Staley et al. 2017) is
   recalibrated from trusted observations, per region if wanted, and an
   update is kept only if it predicts held-out fires at least as well.
4. **Map risk.** Updated likelihood and rainfall thresholds per basin, with
   every observed basin marked hit / miss / false alarm, exported as
   GeoJSON for ArcGIS Online or the Streamlit app.

The design comes straight from the team whiteboards. See
[`docs/architecture.md`](docs/architecture.md) for the photos and a
box-by-box mapping to the code.

## Try it (no data or network needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"

pytest                                   # 41 tests
python scripts/demo_synthetic_loop.py    # whole loop on SYNTHETIC data
streamlit run src/afterburn_watch/app.py # look at the result
```

The demo uses **synthetic** scenes and observations
([`synthetic.py`](src/afterburn_watch/synthetic.py)). It shows the
mechanics working together, not real-world skill. In it, a made-up "Idaho"
region behaves differently from the published model, and after 8 storms
the recalibrated coefficients land close to that region's (made-up) truth.

For real imagery and USGS `pfdf`:

```bash
pip install -e ".[imagery]"
# pfdf is NOT on PyPI -- it's on USGS's own package registry:
pip install -e ".[pfdf]" \
  --extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
```

## Repo structure

```
src/afterburn_watch/
  sensors.py       # Sentinel-2 & Landsat band maps (by role, never by band number)
  ingest.py        # STAC search + loading for both sensors
  masking.py       # cloud / shadow / smoke masks, multi-scene composite
  indices.py       # NBR, dNBR, NDVI, RdNBR
  severity.py      # BAER burn-severity classes
  recover.py       # NASA RECOVER package (authoritative dNBR, soils)
  observations.py  # DF Actual: the observation inventory
  identify.py      # Model 1: satellite debris-flow identification
  predict.py       # DF Prediction: USGS M1 + recalibration
  risk.py          # Map Risk: likelihood classes, outcomes, GeoJSON
  loop.py          # the feedback loop, one iteration at a time
  synthetic.py     # synthetic data for tests and the demo
  app.py           # Streamlit view of the loop
scripts/
  demo_synthetic_loop.py
docs/
  architecture.md     # whiteboards -> code
  technical-spec.md
  img/                # whiteboard photos, framework slide
tests/
```

## Where it plugs into the Project Framework

RECOVER packages (USFS server) provide the authoritative dNBR and soils.
wildcat/pfdf on Lemhi HPC turn those into per-basin T, F, S. This package
adds the satellite loop on top. The resulting risk layer goes to the ArcGIS
Online hazard assessment package / ESRI Dashboard. Details in
[`docs/architecture.md`](docs/architecture.md#3-where-it-sits-in-the-project-framework).

## Team (Afterburn Watch)

| Name | Role |
|---|---|
| Roberto Jones | Researcher & Builder |
| Troy Jenks | Builder |
| Ashraf Md | Strategist |

Advisors: Keith Weber (GIS TReC, Idaho State University), Ashley Bosa
(Resilience Institute, Boise State University).

## Key references

- Graber, A.P., Gorr, A.N., Kean, J.W., Kostelnik, J., Rengers, F.K.,
  Selander, B.D., Thomas, M.A., 2026. *USGS runoff-generated postfire
  debris-flow inventory.* USGS data release, doi:10.5066/P1R9VXC4.
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
