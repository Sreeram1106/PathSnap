"""PathSnap evaluator UI  (runs on LAPTOP 2).

    streamlit run ui/app.py --server.port 8501

Enter Laptop 1's API address in the sidebar (e.g. http://192.168.1.20:8000).
The UI never computes predictions itself: it only calls the API.
"""
import os

import folium
import requests
import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")
st.title("PathSnap — GPS map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")).rstrip("/")
try:
    h = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected — {h.get('n_trips')} trips ({h.get('evaluation_partition')})")
    st.sidebar.caption(f"Model: {h.get('model')}")
    st.sidebar.json(h.get("decoder", {}), expanded=False)
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"API not reachable: {e}")
    st.stop()


@st.cache_data(ttl=60)
def get_trips(url):
    return requests.get(f"{url}/trips", timeout=60).json()


data = get_trips(api)
trips = data["trips"]
st.subheader(f"Evaluation trips ({len(trips)})")
labels = [f"{t['trip_id']}  ·  {t['n_points']} GPS points  ·  {t['duration_s']/60:.1f} min" for t in trips]
ids = [t["trip_id"] for t in trips]
qp_trip = st.query_params.get("trip")  # optional deep link: ?trip=<id>&run=1
idx = st.radio("Select a trip", range(len(trips)), format_func=lambda i: labels[i], horizontal=False,
               index=ids.index(qp_trip) if qp_trip in ids else 0)
trip_id = trips[idx]["trip_id"]
auto = st.query_params.get("run") == "1" and qp_trip == trip_id and "result" not in st.session_state

if st.button(f"Run map-matching for trip {trip_id}", type="primary") or auto:
    with st.spinner("API is running candidate generation → classifier → HMM/Viterbi ..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    m = res.get("metrics") or {}
    st.subheader(f"Results for trip {res['trip_id']}")
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", f"{m.get('top1_accuracy_hmm', float('nan')):.2%}",
                f"{m.get('top1_accuracy_hmm', 0) - m.get('top1_accuracy_classifier_only', 0):+.2%} vs classifier-only")
    c[1].metric("Classifier-only accuracy", f"{m.get('top1_accuracy_classifier_only', float('nan')):.2%}")
    c[2].metric("Route continuity", f"{m.get('route_continuity_hmm', float('nan')):.2%}")
    c[3].metric("Route F1 (length-weighted)", f"{m.get('route_f1', float('nan')):.2%}")
    c[4].metric("API inference time", f"{res['timings_s']['inference_total']:.2f} s")

    gps = res["gps_points"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap", control_scale=True,
                      width="100%", height=590)
    g1 = folium.FeatureGroup("Raw GPS observations (blue)")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(g1)
    g1.add_to(fmap)
    if res["predicted_route"]:
        folium.PolyLine(res["predicted_route"], color="#d62728", weight=5, opacity=0.85,
                        tooltip="Predicted route (HMM/Viterbi)").add_to(folium.FeatureGroup("Predicted route (red)").add_to(fmap))
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    html(fmap.get_root().render(), height=620)

    with st.expander("All metrics returned by the API"):
        st.json(m)
        st.json(res["timings_s"])
    st.caption(f"{res['n_points']} GPS points · {res['n_candidates']} candidate road segments scored · "
               f"{res['hmm_chain_breaks']} HMM chain breaks · {len(res['predicted_route_arc_ids'])} arcs in predicted route")
