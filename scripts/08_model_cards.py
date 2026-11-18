"""One results folder per model + a comparison index page, generated from the
recorded experiment files (no numbers are typed in by hand).

Output:
  results/models/README.md                      comparison of all models (start here)
  results/models/per_model/NN_<Model>/README.md  everything about one model
  results/models/per_model/NN_<Model>/confusion_matrix.png
  results/models/per_model/NN_<Model>/cv_folds.csv
"""
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))
import report_text as T  # noqa: E402  (shared model descriptions, same text as the report)

M = ROOT / "results" / "models"
O = ROOT / "results" / "optuna"
H = ROOT / "results" / "hmm"
PLOTS = ROOT / "results" / "plots"
OUT = M / "per_model"

SCRIPT = {"LogisticRegression": "train_logistic_regression.py", "GaussianNB": "train_gaussian_nb.py",
          "KNN": "train_knn.py", "DecisionTree": "train_decision_tree.py", "RandomForest": "train_random_forest.py",
          "LightGBM": "train_lightgbm.py", "MLP": "train_mlp.py"}


def strip(html):
    return re.sub(r"<[^>]+>", "", html).replace("&rarr;", "->").strip()


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def confusion_png(cm, title, path):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n({cmn[i, j]:.1%})", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["predicted\nwrong arc (0)", "predicted\ncorrect arc (1)"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    sm = pd.read_csv(M / "model_comparison.csv")
    fr = pd.read_csv(M / "cv_fold_results.csv")
    smi = sm.set_index("model")
    ctx = {"sm": smi}
    rank_st = sm.st_top1_acc.rank(ascending=False).astype(int)
    rank_cv = sm.cv_top1_mean.rank(ascending=False).astype(int)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    links = []

    for i, r in sm.iterrows():
        m = r.model
        d = OUT / f"{i + 1:02d}_{m}"
        d.mkdir()
        info = T.INFO[m]
        f = fr[fr.model == m][["fold", "f1_macro", "top1_acc", "log_loss", "accuracy", "precision_macro",
                               "recall_macro", "train_f1_macro", "fit_s"]].copy()
        f.to_csv(d / "cv_folds.csv", index=False)
        mean = f.drop(columns="fold").mean()
        sd = f.drop(columns="fold").std()
        ftab = pd.concat([f.assign(fold=f.fold.astype(str)),
                          pd.DataFrame([{"fold": "mean", **mean}, {"fold": "std", **sd}])], ignore_index=True)
        ftab.columns = ["Fold", "Macro F1", "Top-1", "Log loss", "Accuracy", "Precision (macro)",
                        "Recall (macro)", "Train Macro F1", "Fit time (s)"]
        cm = np.loadtxt(M / f"confusion_{m}.csv", delimiter=",")
        confusion_png(cm, f"{m}: student-test confusion matrix", d / "confusion_matrix.png")
        tn, fp, fn, tp = cm.ravel().astype(int)
        selected = " (SELECTED MODEL)" if m == "LightGBM" else ""
        st = pd.DataFrame([
            ["Observation-level Top-1 accuracy", r.st_top1_acc], ["Macro F1", r.st_f1_macro],
            ["Accuracy (candidate pairs)", r.st_accuracy], ["Precision (macro)", r.st_precision_macro],
            ["Recall (macro)", r.st_recall_macro], ["Precision (correct-arc class)", r.st_precision_pos],
            ["Recall (correct-arc class)", r.st_recall_pos], ["Log loss", r.st_log_loss],
            ["Route continuity (classifier only)", r.st_route_continuity]], columns=["Metric", "Value"])
        lines = [
            f"# Model {i + 1}: {m}{selected}",
            "",
            f"**Algorithm family:** {r.family}  ",
            f"**Configuration:** {r.config}  ",
            f"**Training rows:** {int(r.train_rows):,} (observation, candidate) pairs from 80 training trips  ",
            f"**Training time:** {r.fit_time_s:.1f} s · **Inference:** {r.inference_ms_per_obs:.4f} ms per GPS observation  ",
            f"**Rank among 7 models:** #{rank_st[i]} on student-test Top-1, #{rank_cv[i]} on CV Top-1",
            "",
            "## Headline results",
            "",
            "| CV Macro F1 (mean ± sd) | CV Top-1 (mean ± sd) | Student-test Top-1 | Student-test Macro F1 | Student-test log loss |",
            "|---|---|---|---|---|",
            f"| {r.cv_f1_macro_mean:.4f} ± {r.cv_f1_macro_std:.4f} | {r.cv_top1_mean:.4f} ± {r.cv_top1_std:.4f} | "
            f"{r.st_top1_acc:.4f} | {r.st_f1_macro:.4f} | {r.st_log_loss:.4f} |",
            "",
            "## Why this model was tried",
            strip(info["why"]),
            "",
            "## Strengths and limitations for PathSnap",
            strip(info["sl"]),
            "",
            "## Hyper-parameters (initial values)",
            strip(info["hp"]),
            "",
            "## Loss / objective",
            strip(info["loss"]),
            "",
            "## Cross-validation: 5-fold GroupKFold over the 80 training trips (grouped by trip)",
            "",
            md_table(ftab),
            "",
            "## Student-test results (10 trips)",
            "",
            md_table(st),
            "",
            "## Confusion matrix (student-test candidate pairs, threshold 0.5)",
            "",
            "![confusion matrix](confusion_matrix.png)",
            "",
            f"TN = {tn:,} wrong candidates correctly rejected · FP = {fp:,} wrong candidates accepted · "
            f"FN = {fn:,} correct arcs missed · TP = {tp:,} correct arcs found.",
            "",
            "## Over- or under-fitting?",
            strip(T.fit_comment(m, r, ctx)),
        ]
        if m == "MLP":
            shutil.copy(M / "mlp_epoch_history.png", d / "training_history.png")
            h = pd.read_csv(M / "mlp_epoch_history.csv")
            h.to_csv(d / "training_history.csv", index=False)
            lines += ["", "## Training history (epoch-wise loss)", "",
                      "64 training trips for fitting, 16 held-out training trips for validation.", "",
                      "![training history](training_history.png)"]
        if m == "LightGBM":
            tv = pd.read_csv(O / "tuned_vs_untuned.csv")
            bp = json.load(open(O / "best_params.json"))
            for fn_ in ("optuna_history.png",):
                shutil.copy(O / fn_, d / fn_)
            for fn_ in ("tuned_vs_untuned.png", "lgbm_learning_curve.png", "model_selection_summary.png"):
                shutil.copy(PLOTS / fn_, d / fn_)
            t2 = tv[["model", "cv_f1_macro", "cv_top1", "cv_log_loss", "st_top1_acc", "st_f1_macro", "st_log_loss",
                     "st_route_continuity", "fit_time_s", "inference_ms_per_obs"]]
            t2.columns = ["Model", "CV Macro F1", "CV Top-1", "CV log loss", "ST Top-1", "ST Macro F1", "ST log loss",
                          "ST continuity", "Fit time (s)", "Inference (ms/obs)"]
            hs = json.load(open(H / "student_test_summary.json"))
            dec = json.load(open(ROOT / "models" / "decoder_config.json"))
            lines += [
                "", "## Why it was selected", strip(T.sec3_selection(ctx)),
                "", "![model selection summary](model_selection_summary.png)",
                "", "## Bayesian hyper-parameter optimisation (Optuna TPE)",
                f"Objective: {bp['objective']}. Trials: {bp['n_trials']}. "
                f"Best 3-fold CV Top-1: {bp['best_value']:.4f}.", "",
                "**Best parameters:** " + ", ".join(f"`{k}` = {v:.4g}" if isinstance(v, float) else f"`{k}` = {v}"
                                                    for k, v in bp["best_params"].items()), "",
                md_table(t2), "",
                "![optuna history](optuna_history.png)", "",
                "![tuned vs untuned](tuned_vs_untuned.png)", "",
                "![learning curve](lgbm_learning_curve.png)", "",
                "## After HMM/Viterbi decoding (final system, 10 student-test trips)", "",
                f"Decoder: beta = {dec['beta']} m, emission weight = {dec['emis_w']}, chain break after {dec['gap_s']:.0f} s gap.", "",
                "| | Top-1 | Route continuity | Route F1 |", "|---|---|---|---|",
                f"| Tuned LightGBM only | {hs['top1_clf']:.4f} | {hs['continuity_clf']:.4f} | {hs['route_f1_clf']:.4f} |",
                f"| Tuned LightGBM + HMM | {hs['top1_hmm']:.4f} | {hs['continuity_hmm']:.4f} | {hs['route_f1_hmm']:.4f} |",
                "", "Final model file: `models/final_model.joblib` (parameters in `models/final_model.json`).",
            ]
        lines += ["", "## Reproduce", f"```bash\npython scripts/models/{SCRIPT[m]}\n```",
                  f"Saved model: `models/experiments/{m}.joblib`. Raw fold results: [`cv_folds.csv`](cv_folds.csv)."]
        (d / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        links.append((i + 1, m, r.family, d.name))

    # ---------------- comparison index
    t = sm.copy()
    t.insert(0, "#", range(1, len(t) + 1))
    t["Model"] = [f"[{m}](per_model/{n})" for _, m, _, n in links]
    t["CV Macro F1"] = t.apply(lambda x: f"{x.cv_f1_macro_mean:.4f} ± {x.cv_f1_macro_std:.4f}", axis=1)
    t["CV Top-1"] = t.apply(lambda x: f"{x.cv_top1_mean:.4f} ± {x.cv_top1_std:.4f}", axis=1)
    tab = t[["#", "Model", "family", "CV Macro F1", "CV Top-1", "st_top1_acc", "st_f1_macro", "st_log_loss",
             "st_route_continuity", "fit_time_s", "inference_ms_per_obs"]].rename(columns={
        "family": "Family", "st_top1_acc": "ST Top-1", "st_f1_macro": "ST Macro F1", "st_log_loss": "ST log loss",
        "st_route_continuity": "ST continuity", "fit_time_s": "Fit (s)", "inference_ms_per_obs": "Inference (ms/obs)"})
    best = sm.sort_values("st_top1_acc", ascending=False).model.tolist()
    idx = [
        "# Model comparison (all 7 models)",
        "",
        "Every model is a binary classifier over (GPS observation, candidate road arc) pairs. All models use the same 22 "
        "features and the same 5-fold GroupKFold over the 80 training trips (grouped by trip). Each model is then refit "
        "on all 80 trips and evaluated once on the 10 student-test (ST) trips. Primary metric: observation-level "
        "**Top-1 accuracy** (the highest-scoring candidate per GPS point is the true arc). Click a model for its full page.",
        "",
        md_table(tab),
        "",
        "![Model selection summary](../plots/model_selection_summary.png)",
        "",
        f"**Ranking by student-test Top-1:** {' > '.join(best)}.  ",
        "**Selected:** LightGBM. It is tied with Random Forest on CV, but it is better on student-test, faster and smaller. "
        "It was then Bayes-tuned and combined with the HMM/Viterbi decoder.",
        "",
        "## Comparison plots",
        "", "![CV Macro F1](../plots/cv_macro_f1.png)",
        "", "![Student-test Top-1 and Macro F1](../plots/student_test_top1_f1.png)",
        "", "![Log loss and inference time](../plots/logloss_inference.png)",
        "", "![CV Top-1 and continuity](../plots/cv_top1_continuity.png)",
        "", "![Confusion matrices](../plots/confusion_matrices.png)",
        "",
        "## Files",
        "- `model_comparison.csv`: the table above with every metric",
        "- `cv_fold_results.csv`: every model × fold",
        "- `confusion_<Model>.csv`: student-test confusion matrices",
        "- `per_model/`: one folder per model (README, confusion matrix, fold scores)",
        "- Training code: `scripts/03_train_models.py` (all models) or `scripts/models/train_<model>.py` (one model)",
    ]
    (M / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print("written", [n for *_, n in links])


if __name__ == "__main__":
    main()
