# Afterburn Watch -- Technical Spec

Living reference. The box-by-box mapping from the whiteboards to the code
is in [`architecture.md`](architecture.md).

## 1. Mission

Post-wildfire debris flows are an acute hydro-geomorphic threat across
the Western United States. They can damage critical infrastructure,
water resources and communities within minutes of intense rainfall.

Afterburn Watch is a **satellite-driven feedback loop** for post-fire
debris-flow risk:

1. identify debris flows that actually happened, from Sentinel-2 and
   Landsat imagery (**Model 1**);
2. add them to the observation record (**Actual DF Observations**);
3. recalibrate the USGS likelihood model with them (**DF Prediction**);
4. publish the updated risk map (**Map Risk**), and repeat after the next
   storm.

The goal for emergency managers: a risk picture that is available in
hours and that improves with every storm, instead of a one-shot forecast
that is never checked against what happened.

## 2. Hackathon context

- **Event:** R-CON 2026 Innovation Village, Boise Centre, Boise, Idaho.
- **Live Showcase:** October 13, 2026 (10 min pitch + 5-10 min Q&A).
  Judging emphasizes usability and government adoption potential. A
  finished product isn't required; showing the concept is feasible is.
- **Challenge:** 01 (Wildfire Resilience), Problem Statement 2.
- **Benchmark fire:** 2024 Wapiti Fire, central Idaho.

## 3. Team & roles

| Member | Role | Focus |
|---|---|---|
| Roberto Jones | Researcher + Builder | Sensor choice, masking, Model 1, loop |
| Troy Jenks | Builder | Pipeline, Sentinel data selection, satellite DF marking (his kickoff idea) |
| Ashraf Md | Strategist | Stakeholder fit (USFS, BLM, NPS; USGS, USACE, NWS) |
| Storyteller / PM (shared) | -- | Whitepaper and pitch deck |

## 4. Architecture

See [`architecture.md`](architecture.md). In short:

| Stage | Module |
|---|---|
| Sentinel-2 / Landsat ingest, cloud & smoke masking | `sensors.py`, `ingest.py`, `masking.py` |
| Spectral indices, burn severity | `indices.py`, `severity.py`, `recover.py` |
| DF Actual (USGS inventory + new observations) | `observations.py` |
| Model 1 -- DF identification, single-sat params | `identify.py` |
| DF Prediction -- USGS M1 + recalibration | `predict.py` |
| Map Risk | `risk.py` |
| The loop | `loop.py` |

## 5. Domain notes (Keith Weber briefing, Sept 18)

- **Who uses this:** primary consumers are the Forest Service, BLM and the
  National Park Service. Supporting agencies include the USGS Landslide
  Hazards Program, which produces two products per fire (debris-flow
  likelihood and estimated volume) and hands them to the assigned land
  manager. Fires enter a USGS queue, and results are posted online and
  delivered to the fire manager.
- **Burn severity:** dNBR measures how much vegetation was consumed. Fire
  *intensity* models (heat released) are more accurate, but thermal
  sensors give only intermittent data. Field checks of soil
  hydrophobicity confirm burn intensity.
- **Clouds and smoke:** imagery must be cloud-free and smoke-free. A fire
  that has stopped smoking can still be obscured by *other* nearby fires.
  In one case, clear post-fire imagery wasn't available for several
  months. Landsat's QA bands are good at reporting what's in a pixel.
- **Rainfall thresholds:** debris flows can start at intensities as low as
  ~0.25 in/hr, depending on soil type, slope, burn severity/intensity and
  hydrophobicity. The thresholds were trained on California, and Idaho
  soils differ. Would region-specific models be better? Silica sand
  increases hydrophobicity. Fine ash seals the surface, but gentle rain can
  break that crust and let water soak in instead of running off.
- **Sensors:** Sentinel-2 has the better spatial resolution, Landsat the
  better spectral resolution, and Landsat 9 is preferred over 8. A recent
  ISU thesis compared them for aspen/whitebark pine detection:
  https://giscenter.isu.edu/pdf/PDF_BLM_Aspen/AspenLandsat_Sentinel2_Comparison.pdf
- **LiDAR:** set aside. It isn't wall-to-wall available and is expensive
  and slow to acquire after an event (months, sometimes 2-4 years).
  Reference: https://giscenter.isu.edu/pdf/PDF_FEMA_DOS/LidarGuidanceIdaho.pdf

## 6. Formula reference

| Quantity | Formula | Notes |
|---|---|---|
| NBR | `(NIR - SWIR2) / (NIR + SWIR2)` | Sentinel-2: `(B08 - B12)/(B08 + B12)`. Landsat 8/9: `(B5 - B7)/(B5 + B7)`. Always pick bands by wavelength role, never by number (Keith, Sept 18). |
| dNBR | `(NBR_pre - NBR_post) * 1000` | Burn severity (pre-fire -> post-fire). The same form across a storm is Model 1's `event_dnbr`. |
| NDVI | `(NIR - Red) / (NIR + Red)` | |
| RdNBR | `dNBR / sqrt(abs(NBR_pre))` | Miller & Thode 2007 |
| BAER classes | <100, 100-270, 270-660, >=660 | Unburned/low, low, moderate, high |
| USGS M1 | `p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))` | Staley et al. 2017. 15-min: B=-3.63, Ct=0.41, Cf=0.67, Cs=0.70. Cross-check against `pfdf.models.staley2017.M1`. |
| Rainfall threshold | `R_p = (logit(p) - B) / (Ct*T + Cf*F + Cs*S)` | Rainfall at which likelihood reaches p |
| Recalibration | `min -loglik(theta) + (lambda/2) * abs(theta - theta_Staley)^2` | MAP logistic regression, prior at the published values. Default `lambda` = 2. |

T = proportion of upslope area burned at moderate/high severity on slopes
>= 23 degrees; F = mean dNBR/1000 upslope; S = mean soil KF-factor
upslope; R = peak rainfall accumulation (mm) over the duration.

## 7. Data sources

| Data | Source | Used for |
|---|---|---|
| Observed debris flows (0/1) | USGS Runoff-Generated Postfire Debris-Flow Inventory, doi:10.5066/P1R9VXC4 | Model 1 labels; seed of the observation record |
| Sentinel-2 L2A | Element84 Earth Search (STAC, public) | Model 1 features, dNBR |
| Landsat 8/9 C2 L2 | Microsoft Planetary Computer (STAC, signed URLs) | Model 1 features, dNBR, cross-sensor confirmation |
| dNBR, perimeter, soils | NASA RECOVER package | Authoritative inputs to T, F, S; cross-check of our dNBR |
| Basin T, F, S | wildcat / pfdf on Lemhi HPC | DF Prediction inputs |
| Per-basin storm rainfall | MRMS (Debris Flow Hunters team) | R for each observation |

## 8. Decision log

**Sept 11 (kickoff):** don't *produce* a new authoritative dNBR product.
RECOVER packages already ship the dNBR federal stakeholders trust, so it
remains the authoritative reference for the prediction inputs.

**Sept 18-19 (pivot):** there is no reliable baseline for how long
severity data takes to produce: clouds, smoke from other fires and
provider queues all add delay. So "faster severity maps" is not the pitch.
The team adopted the whiteboard design instead, a feedback loop in which
Sentinel-2 and Landsat identify debris flows that actually occurred and
those observations recalibrate the prediction. This builds on Troy's
kickoff idea of studying predicted debris flows that never happened, and
on his analysis that satellite imagery could mark debris flows at a scale
the current inventory can't reach. We still compute dNBR from Sentinel-2
and Landsat ourselves (Keith's Sept 18 email explains how), because Model
1 needs it as a feature and it fills in where RECOVER has no package yet.

## 9. Roadmap to Oct 13

| Week | Build | Research | Strategy / story |
|---|---|---|---|
| Sept 21 | Loop architecture in repo (done). Download the USGS inventory, confirm column mapping. First live Sentinel-2 + Landsat pull over Wapiti. | Pick pre/post windows for inventory fires; check SCL/QA masks over burn scars. | Draft the loop story for the whitepaper. |
| Sept 28 | Train Model 1 on real inventory sites x real scenes, grouped CV by fire. Wire wildcat/pfdf T, F, S. | Landsat vs Sentinel-2 skill; HLS yes/no. | Stakeholder slides (USFS/BLM/NPS). |
| Oct 5 | Run the loop on Wapiti with real observations. Push the risk layer to ArcGIS Online. | Review results with Keith. | Deck + Q&A prep. |
| Oct 13 | Demo. | | Present. |

## 10. Open questions

- The USGS inventory's exact columns: do they include coordinates for
  every observation, plus T/F/S/R? (`observations.load_inventory` fails
  loudly until mapped.)
- Pixel labels for Model 1: the inventory is basin-level, so training
  labels a basin's channel pixels with its outcome. That is noisy. Are
  mapped flow paths available for some fires?
- Which post-storm window gives the clearest debris-flow signal before
  vegetation regrowth or cleanup hides it?
- How to get per-basin rainfall (R) for each observation from MRMS, and
  in what format.
- Trusted-gate threshold (0.9) and `prior_strength` (2) were tuned only
  on synthetic data.
