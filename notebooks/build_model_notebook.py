"""Builds and executes notebooks/model_comparison.ipynb (all 7 models one by one + comparison).

    python notebooks/build_model_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "model_comparison.ipynb"

MODELS = [
    ("LogisticRegression", "Logistic Regression", "Linear / probabilistic",
     "Learns one weight per feature (log-odds). Fast and gives calibrated probabilities, but cannot model feature interactions."),
    ("GaussianNB", "Gaussian Naive Bayes", "Probabilistic",
     "Assumes every feature is independent given the class. Our distance features are strongly correlated, so it double-counts evidence and is over-confident."),
    ("KNN", "K-Nearest Neighbours (k = 31, distance-weighted)", "Instance-based",
     "Labels a candidate by the 31 most similar training candidates. No training, but prediction is slow (it searches the stored rows)."),
    ("DecisionTree", "Decision Tree (depth 12)", "Tree-based",
     "Learns if-then rules such as 'closest candidate AND heading matches'. Interpretable, but a single tree has high variance."),
    ("RandomForest", "Random Forest (100 trees)", "Ensemble - bagging",
     "Averages 100 trees built on random samples, which reduces the variance of a single tree. Robust, but the model is large (10 MB)."),
    ("LightGBM", "LightGBM (gradient boosting)", "Ensemble - boosting",
     "Builds trees one after another, each fixing the errors of the previous ones, and directly minimises log loss. SELECTED model."),
    ("MLP", "MLP neural network (64-32, ReLU, Adam)", "Neural network",
     "Two hidden layers learn non-linear combinations of the features. The only model trained in epochs (see its loss curve)."),
]


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


cells = [
    md("""# PathSnap - comparison of the 7 candidate-scoring models

**Task.** Every GPS point has up to 12 candidate road segments within 50 m. Each model looks at one
(GPS point, candidate road) pair, described by 22 features, and predicts the probability that this is
the correct road. The highest-scoring candidate is the model's answer for that point.

**Protocol (same for every model).**
1. 5-fold **GroupKFold** cross-validation on the **80 training trips**. Folds are split by trip, so no trip is in both training and validation.
2. Refit on all 80 trips and test once on the **10 student-test trips** (19,160 GPS points).

**Metrics.** **Top-1 accuracy** (main metric: is the chosen road correct?), Macro F1, precision, recall, log loss, route continuity.

This notebook loads each trained model and scores the student-test trips **live**, building the features from the raw
`.track/.nodes/.arcs` files. The CV fold scores were recorded during training (`results/models/cv_fold_results.csv`)."""),
    code("""import sys, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from IPython.display import display, Image
from pathsnap.config import FEATURES, MODELS, RESULTS
from pathsnap.data_io import list_trips, load_trip
from pathsnap.features import prepare_trip, build_candidates
from pathsnap.labels import align_route
from pathsnap.hmm import classifier_only, viterbi_decode
from pathsnap.metrics import pair_metrics, route_continuity, confusion
from pathsnap.pipeline import PathSnapPredictor
pd.set_option("display.float_format", "{:.4f}".format)
plt.rcParams["figure.dpi"] = 100"""),
    md("## Step 1 - build the student-test features live from the raw files"),
    code("""t0 = time.time()
trips, rows = {}, []
for tid in list_trips("student_test"):
    trip = load_trip(tid, "student_test")
    proj, net, xy = prepare_trip(trip)
    c = build_candidates(trip, net, xy).sort_values(["obs", "dist"]).reset_index(drop=True)
    ref, _, _ = align_route(xy, net, trip.route)            # reference labels - used only for scoring
    c["label"] = (c["arc"].to_numpy() == ref[c["obs"].to_numpy()]).astype(int)
    c.insert(0, "trip_id", tid)
    trips[tid] = (trip, net, xy, ref)
    rows.append(c)
df = pd.concat(rows, ignore_index=True)
X, y = df[FEATURES].to_numpy(np.float32), df["label"].to_numpy()
print(f"{len(trips)} trips, {sum(len(v[2]) for v in trips.values()):,} GPS points, {len(df):,} candidate rows x {len(FEATURES)} features ({time.time()-t0:.1f} s)")
print("Features:", ", ".join(FEATURES))"""),
    code("""cv = pd.read_csv(RESULTS / "models" / "cv_fold_results.csv")
summary = pd.read_csv(RESULTS / "models" / "model_comparison.csv").set_index("model")
live = []

def top1_and_continuity(p):
    ok = n = 0; cont = []
    for tid, g in df.assign(p=p).groupby("trip_id", sort=False):
        trip, net, xy, ref = trips[tid]
        pred = classifier_only(g, g["p"].to_numpy(), len(xy))
        ok += int((pred == ref).sum()); n += len(xy)
        cont.append((route_continuity(pred, net.arcs), len(xy) - 1))
    return ok / n, sum(a * b for a, b in cont) / sum(b for _, b in cont)

def show_model(name):
    model = joblib.load(MODELS / "experiments" / f"{name}.joblib")
    print("Configuration:", summary.loc[name, "config"])
    print(f"Training rows: {int(summary.loc[name, 'train_rows']):,}   training time: {summary.loc[name, 'fit_time_s']:.1f} s")
    f = cv[cv.model == name][["fold", "f1_macro", "top1_acc", "log_loss", "train_f1_macro"]].rename(columns={
        "f1_macro": "CV Macro F1", "top1_acc": "CV Top-1", "log_loss": "CV log loss", "train_f1_macro": "Train Macro F1"})
    print("\\nCross-validation (5 folds, 80 training trips) - recorded during training:")
    display(pd.concat([f, f.drop(columns="fold").agg(["mean", "std"]).reset_index().rename(columns={"index": "fold"})]).set_index("fold"))
    t0 = time.time(); p = model.predict_proba(X)[:, 1]; dt = time.time() - t0
    m = pair_metrics(y, p); acc, cont = top1_and_continuity(p)
    row = {"Model": name, "Top-1 accuracy": acc, "Macro F1": m["f1_macro"], "Precision": m["precision_macro"],
           "Recall": m["recall_macro"], "Log loss": m["log_loss"], "Route continuity": cont, "Predict time (s)": dt}
    live.append(row)
    print("Student-test (10 trips) - computed live now:")
    display(pd.DataFrame([row]).set_index("Model"))
    cm = confusion(y, p); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\\n({cmn[i, j]:.1%})", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["pred wrong", "pred correct"]); ax.set_yticks([0, 1], ["true wrong", "true correct"])
    ax.set_title(f"{name} - confusion matrix", fontsize=10); plt.tight_layout(); plt.show()"""),
]

for i, (key, title, fam, desc) in enumerate(MODELS, 1):
    cells.append(md(f"## Model {i} - {title}\n**Family:** {fam}  \n{desc}"))
    cells.append(code(f'show_model("{key}")'))
    if key == "MLP":
        cells.append(md("**MLP training history** - training vs validation loss per epoch (64 trips train, 16 held-out trips validate):"))
        cells.append(code("""h = pd.read_csv(RESULTS / "models" / "mlp_epoch_history.csv")
plt.figure(figsize=(6, 3.5)); plt.plot(h.epoch, h.train_log_loss, "o-", label="train"); plt.plot(h.epoch, h.val_log_loss, "s-", label="validation")
plt.xlabel("epoch"); plt.ylabel("log loss"); plt.legend(); plt.title("MLP training history"); plt.tight_layout(); plt.show()
b = h.val_log_loss.idxmin(); print(f"Lowest validation loss at epoch {int(h.epoch[b])}: after that the training loss keeps falling but validation does not (mild overfitting, early stopping limits it).")"""))

cells += [
    md("## Comparison of all 7 models (student-test, ranked by Top-1 accuracy)"),
    code("""comp = pd.DataFrame(live).set_index("Model").sort_values("Top-1 accuracy", ascending=False)
comp.insert(0, "CV Top-1 (mean)", summary.loc[comp.index, "cv_top1_mean"].values)
comp.insert(1, "CV Macro F1 (mean)", summary.loc[comp.index, "cv_f1_macro_mean"].values)
comp.insert(0, "Rank", range(1, len(comp) + 1))
display(comp)"""),
    code("""order = comp.index[::-1]
colors = ["#6baed6" if m == "LightGBM" else "#9aa5b1" for m in order]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
b = ax[0].barh(order, comp.loc[order, "Top-1 accuracy"], color=colors); ax[0].bar_label(b, fmt="%.3f"); ax[0].set_xlim(0.6, 0.95); ax[0].set_title("Student-test Top-1 accuracy (higher = better)")
b = ax[1].barh(order, comp.loc[order, "Macro F1"], color=colors); ax[1].bar_label(b, fmt="%.3f"); ax[1].set_xlim(0.75, 0.97); ax[1].set_title("Student-test Macro F1 (higher = better)")
b = ax[2].barh(order, comp.loc[order, "Log loss"], color=colors); ax[2].bar_label(b, fmt="%.3f"); ax[2].set_xscale("log"); ax[2].set_title("Log loss (lower = better)")
plt.tight_layout(); plt.show()

data = [cv[cv.model == m].f1_macro.values for m in comp.index]
plt.figure(figsize=(9, 3.6)); plt.boxplot(data, tick_labels=list(comp.index)); plt.ylabel("Macro F1 per fold")
plt.title("Cross-validation Macro F1 across the 5 folds"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()"""),
    md("## Selected model: LightGBM - then Bayesian tuning and HMM/Viterbi"),
    code("""pred = PathSnapPredictor()                          # Bayes-tuned LightGBM + decoder settings
p = pred.model.predict_proba(X)[:, 1]
acc_t, cont_t = top1_and_continuity(p)
ok = n = 0
for tid, g in df.assign(p=p).groupby("trip_id", sort=False):
    trip, net, xy, ref = trips[tid]
    h, _ = viterbi_decode(g, g["p"].to_numpy(), net, xy, trip.track[:, 2], **pred.decoder)
    ok += int((h == ref).sum()); n += len(xy)
final = pd.DataFrame({"Top-1 accuracy": [comp.loc["LightGBM", "Top-1 accuracy"], acc_t, ok / n]},
                     index=["LightGBM (default)", "LightGBM (Bayes-tuned, Optuna 25 trials)", "Tuned LightGBM + HMM/Viterbi"])
display(final)
display(Image(filename=str(RESULTS / "plots" / "model_selection_summary.png")))"""),
    md("""### Why LightGBM was selected
* **Best student-test Top-1 accuracy** of the 7 models (and best Macro F1).
* On cross-validation it is **tied with Random Forest** (the difference is far below the fold-to-fold spread), but it is
  faster to predict and **60x smaller** (0.16 MB vs 10 MB).
* It optimises log loss directly, so its probabilities work well as **HMM emission probabilities**.
* After **Bayesian tuning** it beats Random Forest on every metric. With **HMM/Viterbi** decoding, the final system
  reaches about **94%** Top-1 accuracy on the student-test trips.
* Naive Bayes is worst because its independence assumption is broken by our correlated distance features."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "pathsnap", "display_name": "PathSnap (.venv)", "language": "python"},
                                                 "language_info": {"name": "python"}})
NotebookClient(nb, timeout=600, kernel_name="pathsnap", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, NB)
print("written", NB)
