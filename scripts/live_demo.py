"""Live demo for the review (run in the VS Code terminal).

    python scripts/live_demo.py models          # all 7 trained models + tuned LightGBM + HMM on the 10 student-test trips
    python scripts/live_demo.py trip 00000065   # full pipeline on one trip, opens the Folium map
    python scripts/live_demo.py mlp             # MLP epoch-wise train/validation loss (recorded run) + graph
    python scripts/03b_mlp_history.py           # retrain the MLP live, printing every epoch (~2-3 min)

Everything is computed live from the raw files in data/raw/student_test: candidate
generation, features, prediction with the saved models, Viterbi decoding, scoring.
Nothing is retrained (training all 7 models with 5-fold CV takes ~25 min; use
scripts/03_train_models.py or scripts/models/train_<model>.py for that).
"""
import sys
import time
import warnings
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pathsnap.config import FEATURES, MODELS, RESULTS  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.features import prepare_trip, build_candidates  # noqa: E402
from pathsnap.hmm import viterbi_decode, classifier_only  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import pair_metrics, route_continuity  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor, evaluate  # noqa: E402

ORDER = ["LogisticRegression", "GaussianNB", "KNN", "DecisionTree", "RandomForest", "LightGBM", "MLP"]
FAMILY = {"LogisticRegression": "Linear", "GaussianNB": "Probabilistic", "KNN": "Instance-based",
          "DecisionTree": "Tree", "RandomForest": "Bagging ensemble", "LightGBM": "Boosting", "MLP": "Neural net"}


def line(c="="):
    print(c * 100)


def build_student_test():
    """Candidates + features + reference labels for the 10 student-test trips, built live."""
    trips, rows = {}, []
    t0 = time.time()
    for tid in list_trips("student_test"):
        trip = load_trip(tid, "student_test")
        proj, net, xy = prepare_trip(trip)
        c = build_candidates(trip, net, xy).sort_values(["obs", "dist"]).reset_index(drop=True)
        ref, _, ref_d = align_route(xy, net, trip.route)      # used ONLY for scoring
        c["label"] = (c["arc"].to_numpy() == ref[c["obs"].to_numpy()]).astype(int)
        c.insert(0, "trip_id", tid)
        trips[tid] = (trip, net, xy, ref, ref_d <= 100.0)
        rows.append(c)
        print(f"  trip {tid}: {len(xy):5d} GPS points -> {len(c):6d} candidate road segments")
    df = pd.concat(rows, ignore_index=True)
    print(f"  features built live in {time.time() - t0:.1f} s  ({len(df):,} candidate rows x {len(FEATURES)} features)")
    return df, trips


def top1(df, p, trips):
    n_ok = n_all = 0
    conts = []
    for tid, g in df.assign(p=p).groupby("trip_id", sort=False):
        trip, net, xy, ref, _ = trips[tid]
        pred = classifier_only(g, g["p"].to_numpy(), len(xy))
        n_ok += int((pred == ref).sum()); n_all += len(xy)
        conts.append((route_continuity(pred, net.arcs), len(xy) - 1))
    cont = sum(c * w for c, w in conts) / sum(w for _, w in conts)
    return n_ok / n_all, cont


def run_models():
    line(); print("PathSnap live model comparison on the 10 STUDENT-TEST trips (models trained on the 80 training trips)"); line()
    print("Step 1: candidate generation + feature engineering from the raw .track/.nodes/.arcs files")
    df, trips = build_student_test()
    X, y = df[FEATURES].to_numpy(np.float32), df["label"].to_numpy()
    print("\nStep 2: scoring every candidate with each saved model")
    res = []
    for m in ORDER:
        model = joblib.load(MODELS / "experiments" / f"{m}.joblib")
        t0 = time.time()
        p = model.predict_proba(X)[:, 1]
        dt = time.time() - t0
        pm = pair_metrics(y, p)
        acc, cont = top1(df, p, trips)
        res.append({"Model": m, "Family": FAMILY[m], "Top-1 accuracy": acc, "Macro F1": pm["f1_macro"],
                    "Precision": pm["precision_macro"], "Recall": pm["recall_macro"], "Log loss": pm["log_loss"],
                    "Continuity": cont, "Predict time (s)": dt})
        print(f"  {m:20s} Top-1 = {acc:.4f}   Macro F1 = {pm['f1_macro']:.4f}   log loss = {pm['log_loss']:.4f}   ({dt:.2f} s)")
    print("\nStep 3: Bayes-tuned LightGBM (final model) + HMM/Viterbi decoding")
    pred = PathSnapPredictor()
    p = pred.model.predict_proba(X)[:, 1]
    pm = pair_metrics(y, p)
    acc, cont = top1(df, p, trips)
    res.append({"Model": "LightGBM (Bayes-tuned)", "Family": "Boosting", "Top-1 accuracy": acc, "Macro F1": pm["f1_macro"],
                "Precision": pm["precision_macro"], "Recall": pm["recall_macro"], "Log loss": pm["log_loss"],
                "Continuity": cont, "Predict time (s)": np.nan})
    n_ok = n_all = r_ok_h = r_ok_c = r_all = 0
    conts = []
    t0 = time.time()
    for tid, g in df.assign(p=p).groupby("trip_id", sort=False):
        trip, net, xy, ref, rel = trips[tid]
        h, _ = viterbi_decode(g, g["p"].to_numpy(), net, xy, trip.track[:, 2], **pred.decoder)
        c = classifier_only(g, g["p"].to_numpy(), len(xy))
        n_ok += int((h == ref).sum()); n_all += len(xy)
        r_ok_h += int((h == ref)[rel].sum()); r_ok_c += int((c == ref)[rel].sum()); r_all += int(rel.sum())
        conts.append((route_continuity(h, net.arcs), len(xy) - 1))
    hmm_t = time.time() - t0
    hcont = sum(c * w for c, w in conts) / sum(w for _, w in conts)
    res.append({"Model": "LightGBM tuned + HMM/Viterbi", "Family": "Final system", "Top-1 accuracy": n_ok / n_all,
                "Macro F1": np.nan, "Precision": np.nan, "Recall": np.nan, "Log loss": np.nan,
                "Continuity": hcont, "Predict time (s)": hmm_t})
    print(f"  Viterbi decoding of 10 trips took {hmm_t:.1f} s")

    t = pd.DataFrame(res)
    base = t.iloc[:7].sort_values("Top-1 accuracy", ascending=False)
    t = pd.concat([base, t.iloc[7:]], ignore_index=True)
    t.insert(0, "Rank", [str(i) for i in range(1, 8)] + ["", ""])
    print(); line(); print("RESULTS - student-test, 10 trips, 19,160 GPS points (ranked by Top-1 accuracy)"); line()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(t.to_string(index=False, na_rep="-"))
    line()
    b = base.iloc[0]
    print(f"BEST MODEL: {b.Model}  (Top-1 {b['Top-1 accuracy']:.4f})  -> Bayes-tuned {res[7]['Top-1 accuracy']:.4f}"
          f"  -> with HMM {res[8]['Top-1 accuracy']:.4f}")
    print(f"(Table counts all {n_all:,} GPS points. Report section 5 counts the {r_all:,} points with a reliable reference"
          f" label: classifier {r_ok_c / r_all:.4f} -> HMM {r_ok_h / r_all:.4f}.)")
    line()


def run_mlp():
    h = pd.read_csv(RESULTS / "models" / "mlp_epoch_history.csv")
    line(); print("MLP(64,32) training history - 64 training trips fit, 16 held-out training trips validate"); line()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(h.to_string(index=False))
    b = int(h.val_log_loss.idxmin())
    line("-")
    print(f"Lowest validation loss at epoch {int(h.epoch[b])} ({h.val_log_loss[b]:.4f}); training loss keeps falling to "
          f"{h.train_log_loss.iloc[-1]:.4f} -> the gap that opens after that is mild overfitting, which early stopping limits.")
    print("Re-run the training live (prints every epoch):  python scripts/03b_mlp_history.py")
    line()
    webbrowser.open((RESULTS / "models" / "mlp_epoch_history.png").resolve().as_uri())


def run_trip(tid):
    line(); print(f"PathSnap live run on student-test trip {tid}"); line()
    pred = PathSnapPredictor()
    t0 = time.time()
    trip = load_trip(tid, "student_test", with_route=False)       # the model never sees the route
    out = pred.predict(trip)
    print(f"GPS points            : {out['n_obs']}")
    print(f"Candidate segments    : {out['n_candidates']}")
    for k, v in out["timings"].items():
        print(f"{k:22s}: {v:.3f} s")
    print(f"Total inference       : {time.time() - t0:.2f} s")
    ref = load_trip(tid, "student_test").route                     # scoring only, after prediction
    met = evaluate(out, ref)
    line("-")
    print(f"Top-1 accuracy  classifier only : {met['top1_accuracy_classifier_only']:.4f}")
    print(f"Top-1 accuracy  with HMM/Viterbi: {met['top1_accuracy_hmm']:.4f}")
    print(f"Route continuity (HMM)          : {met['route_continuity_hmm']:.4f}")
    print(f"Route F1, length-weighted (HMM) : {met['route_f1']:.4f}")
    line("-")
    sys.argv = ["06_maps.py"]
    import importlib
    maps = importlib.import_module("06_maps")
    maps.make_map(pred, tid)
    path = RESULTS / "maps" / f"student_test_{tid}.html"
    print(f"Map: {path}")
    webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "trip":
        run_trip(sys.argv[2] if len(sys.argv) > 2 else "00000065")
    elif len(sys.argv) >= 2 and sys.argv[1] == "mlp":
        run_mlp()
    else:
        run_models()
