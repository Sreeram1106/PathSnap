"""Dataset exploration: inventory, data-quality checks, plots and Folium maps.

Partitions used: train (80 trips) and student_test (10 trips). The evaluator
partition is not available to us and is never touched.
Outputs -> results/eda/ and results/maps/
"""
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import DATA_RAW, RESULTS, PROCESSED, CAND_RADIUS_M
from pathsnap.data_io import list_trips, load_trip, _read
from pathsnap.geo import LocalProjection, haversine
from pathsnap.network import RoadNetwork

OUT = RESULTS / "eda"
MAPS = RESULTS / "maps"
OUT.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)


def inspect(tid, part):
    d = DATA_RAW / part / tid
    raw = _read(d / f"{tid}.track")
    trip = load_trip(tid, part)
    tr = trip.track
    dt = np.diff(tr[:, 2])
    step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    n = len(trip.nodes)
    g = coo_matrix((np.ones(len(trip.arcs)), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(n, n))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=n)
    rt = trip.route
    proj = LocalProjection.for_track(tr)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    pair = set(map(tuple, trip.arcs))
    two_way = np.mean([(b, a) in pair for a, b in trip.arcs[:5000]])
    ext_x = haversine(tr[:, 0].min(), tr[:, 1].mean(), tr[:, 0].max(), tr[:, 1].mean())
    ext_y = haversine(tr[:, 0].mean(), tr[:, 1].min(), tr[:, 0].mean(), tr[:, 1].max())
    return {
        "partition": part, "trip_id": tid,
        **{f"size_kb_{ext}": (d / f"{tid}.{ext}").stat().st_size / 1024 for ext in ("track", "nodes", "arcs", "route")},
        "raw_points": len(raw), "clean_points": len(tr), "invalid_dropped": trip.cleaning["invalid_dropped"],
        "duplicate_time_dropped": trip.cleaning["duplicate_time_dropped"], "out_of_order": trip.cleaning["out_of_order"],
        "duration_min": (tr[-1, 2] - tr[0, 2]) / 60, "dt_median": np.median(dt), "frac_dt_1s": np.mean(dt == 1),
        "dt_max": dt.max(), "n_gaps_gt5s": int((dt > 5).sum()), "n_gaps_gt60s": int((dt > 60).sum()),
        "track_km": step.sum() / 1000, "max_step_m": step.max(),
        "mean_speed_kmh": 3.6 * step.sum() / max(tr[-1, 2] - tr[0, 2], 1),
        "lon_center": np.median(tr[:, 0]), "lat_center": np.median(tr[:, 1]),
        "extent_ew_km": ext_x / 1000, "extent_ns_km": ext_y / 1000,
        "n_nodes": n, "n_arcs": len(trip.arcs), "network_km": net.length.sum() / 1000,
        "weak_components": n_weak, "largest_weak_share": np.bincount(lab_w).max() / n,
        "strong_components": n_strong, "largest_strong_share": np.bincount(lab_s).max() / n,
        "isolated_nodes": int((deg == 0).sum()), "two_way_share": two_way,
        "route_arcs": len(rt), "route_unique_arcs": len(set(rt)), "route_km": net.length[rt].sum() / 1000,
        "route_connected_share": np.mean(trip.arcs[rt[:-1], 1] == trip.arcs[rt[1:], 0]),
        "route_repeats": len(rt) - len(set(rt)),
    }


def eda_map(tid, part, fname):
    trip = load_trip(tid, part)
    tr = trip.track
    proj = LocalProjection.for_track(tr)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.fwd(tr[:, :2])
    net.build_index(xy, CAND_RADIUS_M)
    _, near, _, _ = net.candidates(xy, 150.0, 50)
    near = np.unique(near)
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, control_scale=True)
    g_net = folium.FeatureGroup(f"Road network within 150 m ({len(near)} arcs)")
    for a in near:
        u, v = trip.arcs[a]
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color="#888888", weight=2, opacity=0.6).add_to(g_net)
    g_net.add_to(m)
    g_ref = folium.FeatureGroup("Reference route (.route)")
    for a in trip.route:
        u, v = trip.arcs[a]
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color="#2ca02c", weight=5, opacity=0.8).add_to(g_ref)
    g_ref.add_to(m)
    g_gps = folium.FeatureGroup("Raw GPS observations")
    for lo, la, t in tr[:: max(1, len(tr) // 2000)]:
        folium.CircleMarker([la, lo], radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=0.9,
                            tooltip=f"t={t:.0f}s").add_to(g_gps)
    g_gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(MAPS / fname))


def main():
    jobs = [(t, p) for p in ("train", "student_test") for t in list_trips(p)]
    rows = Parallel(n_jobs=4, verbose=2)(delayed(inspect)(t, p) for t, p in jobs)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "trip_inventory.csv", index=False)
    summ = inv.groupby("partition").agg(trips=("trip_id", "size"), gps_points=("clean_points", "sum"),
                                        nodes=("n_nodes", "sum"), arcs=("n_arcs", "sum"),
                                        route_arcs=("route_arcs", "sum"), invalid=("invalid_dropped", "sum"),
                                        gaps_gt60s=("n_gaps_gt60s", "sum"))
    summ.to_csv(OUT / "partition_summary.csv")
    desc = inv.drop(columns=["trip_id", "partition"]).describe().T
    desc.to_csv(OUT / "summary_statistics.csv")
    print(summ.to_string())

    # ---- plots (train + student_test, coloured by partition)
    col = inv.partition.map({"train": "C0", "student_test": "C1"})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist([inv[inv.partition == p].clean_points for p in ("train", "student_test")], bins=30,
                  stacked=True, label=["train", "student_test"]); ax[0, 0].set_title("GPS observations per trip")
    ax[0, 0].legend()
    tr_all = []
    for t, p in jobs:
        tr_all.append(np.diff(load_trip(t, p, with_route=False).track[:, 2]))
    dts = np.concatenate(tr_all)
    ax[0, 1].hist(np.clip(dts, 0, 30), bins=30, log=True); ax[0, 1].set_title("Sampling interval (s, clipped at 30, log count)")
    ax[0, 2].scatter(inv.lon_center, inv.lat_center, c=col, s=25); ax[0, 2].set_title("Trip locations (lon, lat)")
    ax[0, 2].set_xlabel("longitude"); ax[0, 2].set_ylabel("latitude")
    ax[1, 0].scatter(inv.n_arcs, inv.clean_points, c=col, s=20); ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("Road arcs vs GPS points per trip"); ax[1, 0].set_xlabel("arcs"); ax[1, 0].set_ylabel("GPS points")
    ax[1, 1].hist(inv.mean_speed_kmh, bins=25); ax[1, 1].set_title("Mean trip speed (km/h)")
    ax[1, 2].hist(inv.largest_strong_share, bins=25); ax[1, 2].set_title("Share of nodes in largest strongly-connected component")
    fig.tight_layout(); fig.savefig(OUT / "eda_overview.png", dpi=130); plt.close(fig)

    obs = pd.read_parquet(PROCESSED / "train_obs.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.clip(obs.ref_dist, 0, 200), bins=80, log=True)
    ax[0].axvline(100, color="r", ls="--", label="label reliability cut (100 m)"); ax[0].legend()
    ax[0].set_title("Train: GPS distance to aligned reference arc (m)")
    sw = pd.read_csv(OUT / "radius_sweep.csv")
    ax[1].plot(sw.radius_m, sw.miss_rate_ignoring_cap, "o-"); ax[1].axvline(CAND_RADIUS_M, color="r", ls="--")
    ax[1].set_title("Train: candidate miss rate vs search radius"); ax[1].set_xlabel("radius (m)")
    fig.tight_layout(); fig.savefig(OUT / "gps_noise_and_radius.png", dpi=130); plt.close(fig)

    # one trajectory with its road network (static) for the report
    trip = load_trip("00000018", "train")
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = trip.track[:, :2].min(0) - 0.003, trip.track[:, :2].max(0) + 0.003
    nd = trip.nodes
    keep = ((nd[trip.arcs[:, 0]] >= lo) & (nd[trip.arcs[:, 0]] <= hi)).all(1)
    for u, v in trip.arcs[keep][::1]:
        ax.plot(nd[[u, v], 0], nd[[u, v], 1], color="0.75", lw=0.4)
    for a in trip.route:
        u, v = trip.arcs[a]; ax.plot(nd[[u, v], 0], nd[[u, v], 1], color="g", lw=2)
    ax.plot(trip.track[:, 0], trip.track[:, 1], ".", ms=2, color="C0")
    ax.set_title("Train trip 00000018: road network (grey), reference route (green), GPS (blue)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); fig.tight_layout()
    fig.savefig(OUT / "trip_00000018_network.png", dpi=130); plt.close(fig)

    # ---- data-quality notes
    bad = obs.groupby("trip_id").ref_dist.apply(lambda s: (s > 100).mean())
    notes = [
        "# Data-quality record (generated by scripts/01_eda.py)",
        f"* Partitions: train {int((inv.partition=='train').sum())} trips, student_test {int((inv.partition=='student_test').sum())} trips; evaluator_test not provided to students.",
        f"* Invalid / NaN coordinates removed: {int(inv.invalid_dropped.sum())}; duplicate timestamps removed: {int(inv.duplicate_time_dropped.sum())}; out-of-order timestamps: {int(inv.out_of_order.sum())}.",
        f"* Sampling: median interval 1 s; {np.mean(dts == 1):.1%} of all steps are exactly 1 s; {int((dts > 60).sum())} gaps longer than 60 s; longest gap {dts.max():.0f} s.",
        f"* Trips are spread over {inv.lon_center.round(0).nunique()} distinct longitudes (worldwide) -> a per-trip local metric projection is used.",
        f"* Every .arcs file only references valid node rows; {int(inv.isolated_nodes.sum())} isolated nodes in total; networks are mostly one large component (median largest strongly-connected share {inv.largest_strong_share.median():.2f}).",
        f"* Reference routes are almost always connected (mean share of consecutive arcs that share a node: {inv.route_connected_share.mean():.4f}).",
        f"* Label alignment: {((obs.ref_dist > 100).mean()):.1%} of training observations are > 100 m from their aligned reference arc; trips with > 50 % such points: {', '.join(bad[bad > 0.5].index)} (GPS appears systematically offset from the route). These observations are excluded from training labels.",
        f"* Candidate radius {CAND_RADIUS_M:.0f} m chosen from the radius sweep (miss rate flattens after ~40-50 m while candidates per point keep growing).",
    ]
    (OUT / "data_quality.md").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))

    eda_map("00000018", "train", "eda_train_00000018.html")
    eda_map("00000054", "train", "eda_train_00000054_offset_anomaly.html")


if __name__ == "__main__":
    main()
