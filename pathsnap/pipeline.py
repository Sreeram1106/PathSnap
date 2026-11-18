"""End-to-end inference (no ground truth) + separate server-side evaluation."""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .config import MODELS, FEATURES
from .data_io import Trip
from .features import prepare_trip, build_candidates
from .hmm import viterbi_decode, classifier_only, reconstruct_route
from .labels import align_route
from .metrics import route_continuity, route_overlap


class PathSnapPredictor:
    def __init__(self, model_path: Path = MODELS / "final_model.joblib",
                 decoder_path: Path = MODELS / "decoder_config.json"):
        self.model = joblib.load(model_path)
        self.decoder = json.load(open(decoder_path)) if Path(decoder_path).exists() else {}
        meta = Path(model_path).with_suffix(".json")
        self.meta = json.load(open(meta)) if meta.exists() else {}

    def predict(self, trip: Trip) -> dict:
        """Uses only trip.track, trip.nodes, trip.arcs (route is never read)."""
        timings = {}
        t0 = time.perf_counter()
        proj, net, xy = prepare_trip(trip)
        cands = build_candidates(trip, net, xy)
        timings["features_s"] = time.perf_counter() - t0
        N = len(xy)
        t0 = time.perf_counter()
        prob = self.model.predict_proba(cands[FEATURES].to_numpy(np.float32))[:, 1] if len(cands) else np.zeros(0)
        timings["classifier_s"] = time.perf_counter() - t0
        clf_arcs = classifier_only(cands, prob, N)
        t0 = time.perf_counter()
        hmm_arcs, n_breaks = viterbi_decode(cands, prob, net, xy, trip.track[:, 2], **self.decoder)
        timings["viterbi_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        route = reconstruct_route(hmm_arcs, net)
        timings["route_s"] = time.perf_counter() - t0
        has_cand = np.zeros(N, bool)
        has_cand[cands["obs"].unique()] = True
        return dict(trip_id=trip.trip_id, n_obs=N, net=net, xy=xy, proj=proj,
                    obs_arcs=hmm_arcs, clf_arcs=clf_arcs, route=route, n_breaks=n_breaks,
                    has_candidates=has_cand, n_candidates=len(cands), timings=timings)


def evaluate(pred: dict, ref_route: np.ndarray) -> dict:
    """Server-side scoring against the reference route (never sent to the model)."""
    net, xy = pred["net"], pred["xy"]
    ref_obs, _, ref_d = align_route(xy, net, ref_route)
    hmm, clf = pred["obs_arcs"], pred["clf_arcs"]
    out = {
        "top1_accuracy_hmm": float((hmm == ref_obs).mean()),
        "top1_accuracy_classifier_only": float((clf == ref_obs).mean()),
        "route_continuity_hmm": route_continuity(hmm, net.arcs),
        "route_continuity_classifier_only": route_continuity(clf, net.arcs),
        "candidate_miss_rate": float(1 - pred["has_candidates"].mean()),
    }
    out.update(route_overlap(pred["route"], ref_route, net.length))
    clf_route = reconstruct_route(clf, net)
    out.update({f"{k}_classifier_only": v for k, v in route_overlap(clf_route, ref_route, net.length).items()})
    return out


def route_geometry(pred: dict) -> list[list[float]]:
    """Predicted route as [lat, lon] polyline."""
    net, proj = pred["net"], pred["proj"]
    pts = []
    for a in pred["route"]:
        for xy in (net.A[a], net.B[a]):
            pts.append(xy)
    if not pts:
        return []
    ll = proj.inv(np.array(pts))
    return [[float(la), float(lo)] for lo, la in ll]
