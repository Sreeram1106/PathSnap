"""HMM/Viterbi decoder: tune (beta, emission weight) and compare with classifier-only.

* decoder parameters are chosen on 25 training trips using OUT-OF-FOLD classifier
  probabilities (the classifier never saw those trips), as the guidelines allow
  training + student-test data for decoder optimisation;
* the chosen decoder is then compared with classifier-only on the 10 student-test trips.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import PROCESSED, RESULTS, MODELS, SEED
from pathsnap.data_io import load_trip
from pathsnap.features import prepare_trip
from pathsnap.hmm import viterbi_decode, classifier_only, reconstruct_route
from pathsnap.metrics import route_continuity, route_overlap

OUT = RESULTS / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
GRID = [dict(beta=b, emis_w=w) for b in (2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0) for w in (0.25, 0.5, 1.0, 2.0)]


def run_trip(tid, part, cands, prob_col, obs_df, configs):
    trip = load_trip(tid, part)
    proj, net, xy = prepare_trip(trip)
    cands = cands.sort_values(["obs", "dist"])
    prob = cands[prob_col].to_numpy()
    ref = obs_df.set_index("obs")["ref_arc"].reindex(np.arange(len(xy))).to_numpy()
    rel = obs_df.set_index("obs")["reliable"].reindex(np.arange(len(xy))).fillna(False).to_numpy()
    out = []
    clf = classifier_only(cands, prob, len(xy))
    clf_route = reconstruct_route(clf, net)
    base = {"trip_id": tid, "n_obs": int(rel.sum()),
            "top1_clf": float((clf == ref)[rel].mean()),
            "continuity_clf": route_continuity(clf, net.arcs),
            **{f"{k}_clf": v for k, v in route_overlap(clf_route, trip.route, net.length).items()}}
    lab = cands["label"].to_numpy()
    pick_clf = (cands["arc"].to_numpy() == clf[cands["obs"].to_numpy()]).astype(int)
    base["f1_macro_clf"] = f1_score(lab, pick_clf, average="macro")
    for cfg in configs:
        t0 = time.perf_counter()
        hmm, breaks = viterbi_decode(cands, prob, net, xy, trip.track[:, 2], **cfg)
        dec_s = time.perf_counter() - t0
        route = reconstruct_route(hmm, net)
        pick = (cands["arc"].to_numpy() == hmm[cands["obs"].to_numpy()]).astype(int)
        out.append({**base, **cfg, "top1_hmm": float((hmm == ref)[rel].mean()),
                    "continuity_hmm": route_continuity(hmm, net.arcs), "breaks": breaks,
                    "f1_macro_hmm": f1_score(lab, pick, average="macro"), "decode_s": dec_s,
                    **{f"{k}_hmm": v for k, v in route_overlap(route, trip.route, net.length).items()}})
    return out


def load_part(part):
    full = pd.read_parquet(PROCESSED / f"{part}_cands.parquet", columns=["trip_id", "obs", "arc", "dist", "proj_frac"])
    obs = pd.read_parquet(PROCESSED / f"{part}_obs.parquet")
    obs["reliable"] = obs["ref_dist"] <= 100.0
    if part == "train":
        pr = pd.read_parquet(PROCESSED / "train_oof.parquet").rename(columns={"p_oof": "p"})
    else:
        pr = pd.read_parquet(PROCESSED / "student_test_pred.parquet")
    return full.merge(pr, on=["trip_id", "obs", "arc"]), obs


def main():
    tr, tr_obs = load_part("train")
    rel_share = tr_obs.groupby("trip_id")["reliable"].mean()
    pool = sorted(rel_share[rel_share > 0.95].index)
    rng = np.random.default_rng(SEED)
    tune_trips = sorted(rng.choice(pool, size=min(25, len(pool)), replace=False))
    res = Parallel(n_jobs=6, verbose=5)(
        delayed(run_trip)(t, "train", tr[tr.trip_id == t], "p", tr_obs[tr_obs.trip_id == t], GRID)
        for t in tune_trips)
    grid = pd.DataFrame([r for rr in res for r in rr])
    grid.to_csv(OUT / "decoder_grid_train_trips.csv", index=False)
    w = grid.groupby(["beta", "emis_w"]).apply(lambda g: np.average(g.top1_hmm, weights=g.n_obs)).rename("top1_hmm")
    agg = grid.groupby(["beta", "emis_w"])[["continuity_hmm", "route_f1_hmm", "decode_s"]].mean().join(w)
    agg = agg.reset_index().sort_values("top1_hmm", ascending=False)
    agg.to_csv(OUT / "decoder_grid_summary.csv", index=False)
    print(agg.round(4).to_string())
    best = agg.iloc[0]
    cfg = {"beta": float(best.beta), "emis_w": float(best.emis_w), "gap_s": 60.0, "back_tol": 25.0,
           "cutoff_extra": 200.0}
    json.dump(cfg, open(MODELS / "decoder_config.json", "w"), indent=2)
    clf_ref = np.average(grid.drop_duplicates("trip_id").top1_clf, weights=grid.drop_duplicates("trip_id").n_obs)
    print("tuning trips classifier-only top1:", round(clf_ref, 4), "best decoder:", cfg)

    st, st_obs = load_part("student_test")
    dc = {k: cfg[k] for k in ("beta", "emis_w", "gap_s", "back_tol", "cutoff_extra")}
    res = Parallel(n_jobs=6)(delayed(run_trip)(t, "student_test", st[st.trip_id == t], "p",
                                               st_obs[st_obs.trip_id == t], [dc])
                             for t in sorted(st.trip_id.unique()))
    per = pd.DataFrame([r for rr in res for r in rr])
    per.to_csv(OUT / "student_test_per_trip.csv", index=False)
    tot = {}
    for c in ["top1", "continuity", "route_precision", "route_recall", "route_f1", "f1_macro"]:
        for s in ["clf", "hmm"]:
            col = f"{c}_{s}"
            tot[col] = np.average(per[col], weights=per.n_obs) if c == "top1" else per[col].mean()
    tot["decode_s_mean"] = per.decode_s.mean()
    tot["breaks_total"] = int(per.breaks.sum())
    json.dump(tot, open(OUT / "student_test_summary.json", "w"), indent=2)
    print(per.round(4).to_string())
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
