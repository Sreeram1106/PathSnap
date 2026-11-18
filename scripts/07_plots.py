"""Comparison plots generated from the recorded experiment CSVs (no numbers typed in by hand)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import RESULTS

M = RESULTS / "models"
O = RESULTS / "optuna"
H = RESULTS / "hmm"
P = RESULTS / "plots"
P.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(P / name, dpi=150)
    plt.close(fig)


def main():
    sm = pd.read_csv(M / "model_comparison.csv")
    fr = pd.read_csv(M / "cv_fold_results.csv")
    order = sm.sort_values("st_top1_acc")["model"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [fr[fr.model == m].f1_macro.to_numpy() for m in order]
    ax.boxplot(data, tick_labels=order)
    ax.scatter(np.repeat(np.arange(1, len(order) + 1), [len(d) for d in data]), np.concatenate(data), s=12, c="C1", zorder=3)
    ax.set_ylabel("Macro F1 (per fold)"); ax.set_title("5-fold GroupKFold CV Macro F1 (80 training trips)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); save(fig, "cv_macro_f1.png")

    s = sm.set_index("model").loc[order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w / 2, s.st_top1_acc, w, label="Top-1 accuracy (observation)")
    b2 = ax.bar(x + w / 2, s.st_f1_macro, w, label="Macro F1 (candidate pairs)")
    ax.bar_label(b1, fmt="%.3f", fontsize=7); ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, order, rotation=25, ha="right"); ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right")
    ax.set_title("Student-test (10 trips) performance"); save(fig, "student_test_top1_f1.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    b = ax[0].bar(order, s.st_log_loss, color="C2"); ax[0].bar_label(b, fmt="%.3f", fontsize=7)
    ax[0].set_title("Student-test log loss (lower is better)"); ax[0].set_yscale("log")
    b = ax[1].bar(order, s.inference_ms_per_obs, color="C4"); ax[1].bar_label(b, fmt="%.3f", fontsize=7)
    ax[1].set_title("Classifier inference time (ms per GPS observation)"); ax[1].set_yscale("log")
    for a in ax:
        plt.setp(a.get_xticklabels(), rotation=25, ha="right")
    save(fig, "logloss_inference.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w / 2, s.cv_top1_mean, w, yerr=s.cv_top1_std, capsize=3, label="CV Top-1 (mean ± sd)")
    b2 = ax.bar(x + w / 2, s.st_route_continuity, w, label="Student-test route continuity")
    ax.set_xticks(x, order, rotation=25, ha="right"); ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right")
    ax.set_title("Generalisation and route continuity (classifier only)"); save(fig, "cv_top1_continuity.png")

    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for a, m in zip(axs.ravel(), sm.model):
        cm = np.loadtxt(M / f"confusion_{m}.csv", delimiter=",")
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.2f}", ha="center", va="center",
                       color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"]); a.set_title(m, fontsize=9)
    axs.ravel()[-1].axis("off")
    fig.suptitle("Student-test confusion matrices (candidate pairs, threshold 0.5)"); save(fig, "confusion_matrices.png")

    if (O / "tuned_vs_untuned.csv").exists():
        tv = pd.read_csv(O / "tuned_vs_untuned.csv").set_index("model")
        cols = ["cv_f1_macro", "st_top1_acc", "st_f1_macro", "st_route_continuity"]
        fig, ax = plt.subplots(figsize=(8, 4))
        xx = np.arange(len(cols))
        for k, (name, r) in enumerate(tv.iterrows()):
            bb = ax.bar(xx + (k - .5) * w, r[cols], w, label=name); ax.bar_label(bb, fmt="%.4f", fontsize=7)
        ax.set_xticks(xx, ["CV Macro F1", "ST Top-1", "ST Macro F1", "ST continuity"]); ax.set_ylim(0.8, 1.0)
        ax.legend(); ax.set_title("LightGBM: untuned vs Bayesian-tuned"); save(fig, "tuned_vs_untuned.png")
    if (O / "lgbm_learning_curve.csv").exists():
        lc = pd.read_csv(O / "lgbm_learning_curve.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lc.iteration, lc.train_logloss, label="train folds"); ax.plot(lc.iteration, lc.valid_logloss, label="validation fold")
        ax.set_xlabel("boosting round"); ax.set_ylabel("binary log loss"); ax.legend()
        ax.set_title("Tuned LightGBM learning curve (CV fold 0)"); save(fig, "lgbm_learning_curve.png")
    if (H / "student_test_per_trip.csv").exists():
        per = pd.read_csv(H / "student_test_per_trip.csv")
        per["trip_id"] = per.trip_id.astype(str).str.zfill(8)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        xx = np.arange(len(per))
        ax[0].bar(xx - w / 2, per.top1_clf, w, label="classifier only"); ax[0].bar(xx + w / 2, per.top1_hmm, w, label="classifier + HMM")
        ax[0].set_xticks(xx, per.trip_id, rotation=45); ax[0].set_ylim(0.5, 1); ax[0].legend(); ax[0].set_title("Top-1 accuracy per student-test trip")
        ax[1].bar(xx - w / 2, per.continuity_clf, w, label="classifier only"); ax[1].bar(xx + w / 2, per.continuity_hmm, w, label="classifier + HMM")
        ax[1].set_xticks(xx, per.trip_id, rotation=45); ax[1].set_ylim(0.5, 1); ax[1].legend(); ax[1].set_title("Route continuity per student-test trip")
        save(fig, "hmm_vs_classifier.png")
        g = pd.read_csv(H / "decoder_grid_summary.csv")
        pv = g.pivot(index="beta", columns="emis_w", values="top1_hmm")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pv.to_numpy(), cmap="viridis")
        ax.set_xticks(range(pv.shape[1]), pv.columns); ax.set_yticks(range(pv.shape[0]), pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.iloc[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_xlabel("emission weight w"); ax.set_ylabel("transition scale beta (m)")
        ax.set_title("Decoder grid: Top-1 on 25 train trips (OOF)"); fig.colorbar(im); save(fig, "decoder_grid.png")


if __name__ == "__main__":
    main()


def selection_summary():
    """One picture for model selection: the 7 models, tuned LightGBM and tuned LightGBM + HMM."""
    sm = pd.read_csv(M / "model_comparison.csv").sort_values("st_top1_acc")
    tv = pd.read_csv(O / "tuned_vs_untuned.csv").set_index("model")
    hs = json.load(open(H / "student_test_summary.json"))
    tuned = tv.loc["LightGBM (Bayes-tuned)"]
    names = sm.model.tolist() + ["LightGBM\n(Bayes-tuned)", "LightGBM tuned\n+ HMM/Viterbi"]
    st = sm.st_top1_acc.tolist() + [tuned.st_top1_acc, hs["top1_hmm"]]
    cv = sm.cv_top1_mean.tolist() + [tuned.cv_top1, np.nan]
    cvsd = sm.cv_top1_std.tolist() + [np.nan, np.nan]
    colors = ["#9aa5b1"] * len(sm) + ["#1f77b4", "#d62728"]
    colors[names.index("LightGBM")] = "#6baed6"
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    b = ax[0].bar(range(len(names)), st, color=colors)
    ax[0].bar_label(b, fmt="%.3f", fontsize=8)
    ax[0].set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax[0].set_ylim(0.6, 1.0); ax[0].set_ylabel("Top-1 accuracy")
    ax[0].set_title("Student-test Top-1 accuracy (10 trips)")
    n = len(sm) + 1
    b = ax[1].bar(range(n), cv[:n], yerr=cvsd[:n], capsize=3, color=colors[:n])
    ax[1].bar_label(b, fmt="%.3f", fontsize=8, padding=6)
    ax[1].set_xticks(range(n), names[:n], rotation=35, ha="right", fontsize=8)
    ax[1].set_ylim(0.6, 0.95); ax[1].set_ylabel("Top-1 accuracy (mean ± sd)")
    ax[1].set_title("5-fold GroupKFold CV Top-1 (80 training trips)")
    fig.suptitle("Model selection: LightGBM is best on student-test, tied best on CV, and best on both after tuning")
    save(fig, "model_selection_summary.png")


if __name__ == "__main__":
    selection_summary()
