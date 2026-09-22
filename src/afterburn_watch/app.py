"""Streamlit view of the feedback loop -- entry point.

    python scripts/demo_synthetic_loop.py          # writes demo_output/
    streamlit run src/afterburn_watch/app.py

Reads a saved loop state (loop.save_state) and the latest Map Risk GeoJSON
and shows the red whiteboard as it runs: the observations collected so
far, how the DF Prediction coefficients moved, and the risk map with
hits / misses / false alarms.

The Project Framework's production front end is the ESRI Dashboard on
ArcGIS Online. This app is for local development and the hackathon demo;
the same GeoJSON can be uploaded to ArcGIS Online.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from afterburn_watch.loop import load_state

OUTCOME_COLORS = {
    "hit": "#1a9850",
    "correct_negative": "#91cf60",
    "false_alarm": "#fc8d59",
    "miss": "#d73027",
    None: "#999999",
}
CLASS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

st.set_page_config(page_title="Afterburn Watch", layout="wide")
st.title("Afterburn Watch")
st.caption("Identify by satellite -> actual observations -> DF prediction -> map risk, and repeat.")

out_dir = Path(st.sidebar.text_input("Output folder", "demo_output"))
state_dir = out_dir / "loop_state"
if not (state_dir / "history.json").exists():
    st.info(f"No loop state in {state_dir}/ yet. Run `python scripts/demo_synthetic_loop.py` first.")
    st.stop()

state = load_state(state_dir)
if any("synthetic" in str(r) for r in state.inventory["region"].dropna().unique()):
    st.warning("Showing SYNTHETIC demo data -- not real observations.")

# --- Actual DF Observations -------------------------------------------------
inv = state.inventory
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", len(inv))
c2.metric("Debris flows", int(inv["response"].sum()))
c3.metric("From satellites", int(inv["source"].astype(str).str.startswith("sat_").sum()))
c4.metric("Loop iterations", len(state.history))

# --- DF Prediction: coefficient history --------------------------------------
st.subheader("DF Prediction -- USGS M1 coefficients over the loop")
rows = []
for h in state.history:
    after = h["prediction"]["coefficients_after"]
    rows.append({"iteration": h["iteration"], **{k: after[k] for k in ("B", "Ct", "Cf", "Cs")},
                 "accepted": h["prediction"]["accepted"], "pool": h["prediction"].get("pool")})
hist = pd.DataFrame(rows)
if len(hist):
    st.line_chart(hist.set_index("iteration")[["B", "Ct", "Cf", "Cs"]])
    st.dataframe(hist, hide_index=True)
st.caption(f"Base: published Staley et al. 2017 ({state.base.duration_min}-min). "
           f"Current source: {state.coefficients.source}")

# --- Map Risk -------------------------------------------------------------------
st.subheader("Map Risk -- latest storm")
gj_path = out_dir / "map_risk_latest.geojson"
if gj_path.exists():
    import folium
    from streamlit_folium import st_folium

    gj = json.loads(gj_path.read_text())
    color_by = st.radio("Color by", ["likelihood class", "observed outcome"], horizontal=True)
    lats = [f["geometry"]["coordinates"][1] for f in gj["features"]]
    lons = [f["geometry"]["coordinates"][0] for f in gj["features"]]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=13, tiles="OpenStreetMap")
    for f in gj["features"]:
        p = f["properties"]
        if color_by == "observed outcome":
            color = OUTCOME_COLORS.get(p.get("outcome"), "#999999")
        else:
            color = CLASS_COLORS[p["likelihood_class"]] if p.get("likelihood_class", -1) >= 0 else "#999999"
        lon, lat = f["geometry"]["coordinates"]
        folium.CircleMarker(
            [lat, lon], radius=7, color=color, fill=True, fill_opacity=0.85,
            tooltip=(f"basin {p.get('basin_id')}: likelihood {p.get('likelihood', 0):.0%} "
                     f"({p.get('likelihood_label')}), observed: {p.get('outcome') or 'n/a'}"),
        ).add_to(m)
    st_folium(m, height=520, use_container_width=True)
    counts = pd.Series([f["properties"].get("outcome") for f in gj["features"]]).value_counts()
    st.write(counts.rename("basins"))
else:
    st.info("No risk map yet.")
