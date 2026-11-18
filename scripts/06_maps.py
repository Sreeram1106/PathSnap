"""Folium maps of the final pipeline on student-test trips:
raw GPS, road network around the track, reference route, classifier-only matches
and the HMM/Viterbi predicted route, each in its own toggleable layer."""
import sys
from pathlib import Path

import folium
import numpy as np
from branca.element import Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import RESULTS
from pathsnap.data_io import list_trips, load_trip
from pathsnap.pipeline import PathSnapPredictor, evaluate

MAPS = RESULTS / "maps"
MAPS.mkdir(parents=True, exist_ok=True)

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:10px 14px;
border:1px solid #999;border-radius:6px;font:13px sans-serif;line-height:1.6">
<b>Trip {tid}</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS observations<br>
<span style="color:#888">&#9472;&#9472;</span> Road network (within 150 m)<br>
<span style="color:#2ca02c">&#9472;&#9472;</span> Reference route<br>
<span style="color:#ff7f0e">&#9679;</span> Classifier-only match (wrong arcs)<br>
<span style="color:#d62728">&#9472;&#9472;</span> Predicted route (HMM/Viterbi)<br>
Top-1: classifier {clf:.1%} &rarr; HMM {hmm:.1%}</div>"""


def seg(net, proj, a):
    ll = proj.inv(np.vstack([net.A[a], net.B[a]]))
    return [[ll[0, 1], ll[0, 0]], [ll[1, 1], ll[1, 0]]]


def make_map(pred_obj, tid, part="student_test"):
    trip = load_trip(tid, part, with_route=False)
    pred = pred_obj.predict(trip)
    ref = load_trip(tid, part).route
    met = evaluate(pred, ref)
    net, proj, tr = pred["net"], pred["proj"], trip.track
    _, near, _, _ = net.candidates(pred["xy"], 150.0, 50)
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, control_scale=True)
    g = folium.FeatureGroup("Road network (150 m)", show=True)
    for a in np.unique(near):
        folium.PolyLine(seg(net, proj, a), color="#888888", weight=2, opacity=0.5).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Reference route")
    for a in ref:
        folium.PolyLine(seg(net, proj, a), color="#2ca02c", weight=7, opacity=0.7).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Predicted route (HMM/Viterbi)")
    for a in pred["route"]:
        folium.PolyLine(seg(net, proj, a), color="#d62728", weight=3, opacity=0.95).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Raw GPS observations")
    step = max(1, len(tr) // 2500)
    for lo, la, t in tr[::step]:
        folium.CircleMarker([la, lo], radius=2, color="#1f77b4", weight=0, fill=True, fill_opacity=0.9,
                            tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Classifier-only wrong matches", show=False)
    from pathsnap.labels import align_route
    ref_obs, _, _ = align_route(pred["xy"], net, ref)
    wrong = np.flatnonzero(pred["clf_arcs"] != ref_obs)
    for i in wrong[::max(1, len(wrong) // 1500)]:
        a = pred["clf_arcs"][i]
        mid = proj.inv(((net.A[a] + net.B[a]) / 2)[None])[0]
        folium.CircleMarker([mid[1], mid[0]], radius=3, color="#ff7f0e", fill=True, weight=0, fill_opacity=0.9).add_to(g)
    g.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="black")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element(LEGEND.format(tid=tid, clf=met["top1_accuracy_classifier_only"],
                                                      hmm=met["top1_accuracy_hmm"])))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(MAPS / f"student_test_{tid}.html"))
    print(tid, {k: round(v, 4) for k, v in met.items()})


if __name__ == "__main__":
    p = PathSnapPredictor()
    for tid in (sys.argv[1:] or list_trips("student_test")):
        make_map(p, tid)
