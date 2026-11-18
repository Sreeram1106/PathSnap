"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected LightGBM model.

Objective : maximise mean observation-level Top-1 accuracy over a 3-fold
            GroupKFold of the 80 training trips (student-test is never used here).
Then      : refit best params on all training trips, compare with the untuned
            model on the same 5-fold CV + student-test, save OOF probabilities
            (for decoder tuning) and the final model.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation, record_evaluation
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathsnap.config import PROCESSED, RESULTS, MODELS, FEATURES, SEED
from pathsnap.data_io import list_trips, load_trip
from pathsnap.metrics import pair_metrics, top1_accuracy, confusion
import importlib
tm = importlib.import_module("03_train_models")

OUT = RESULTS / "optuna"
OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

SEARCH_SPACE = {
    "n_estimators": "int [100, 600]",
    "learning_rate": "float log [0.01, 0.3]",
    "num_leaves": "int log [15, 255]",
    "min_child_samples": "int log [10, 300]",
    "subsample": "float [0.5, 1.0] (bagging_freq=1)",
    "colsample_bytree": "float [0.5, 1.0]",
    "reg_lambda": "float log [1e-3, 10]",
    "reg_alpha": "float log [1e-3, 10]",
}


def make(params):
    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, subsample_freq=1, **params)


def cv_eval(params, tr, tr_obs, folds, keep_oof=False):
    rows, oof = [], np.zeros(len(tr))
    for k, (ti, vi) in enumerate(folds):
        m = make(params)
        m.fit(tr[FEATURES].iloc[ti].to_numpy(np.float32), tr["label"].iloc[ti].to_numpy())
        dva = tr.iloc[vi].copy()
        dva["p"] = m.predict_proba(dva[FEATURES].to_numpy(np.float32))[:, 1]
        oof[vi] = dva["p"].to_numpy()
        vt = set(dva.trip_id)
        rows.append({"fold": k, **pair_metrics(dva["label"].to_numpy(), dva["p"].to_numpy()),
                     **top1_accuracy(dva, tr_obs[tr_obs.trip_id.isin(vt)], "p")})
    return pd.DataFrame(rows), oof


def main():
    tr, tr_obs = tm.load_train()
    groups = tr["trip_id"].to_numpy()
    folds3 = list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(tr, groups=groups))
    folds5 = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(tr, groups=groups))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
        t0 = time.time()
        res, _ = cv_eval(params, tr, tr_obs, folds3)
        trial.set_user_attr("cv3_f1_macro", float(res.f1_macro.mean()))
        trial.set_user_attr("cv3_log_loss", float(res.log_loss.mean()))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(res.top1_acc.mean())

    storage = f"sqlite:///{(OUT / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(direction="maximize", study_name="lgbm_top1",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                storage=storage, load_if_exists=True)
    # trial 0 = the untuned defaults, so the search starts from the baseline
    if len(study.trials) == 0:
        study.enqueue_trial({"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20,
                             "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    t_search = time.time()
    remaining = N_TRIALS - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    t_search = time.time() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    print("best", study.best_value, best)
    json.dump({"objective": "mean 3-fold GroupKFold observation Top-1 accuracy (80 train trips)",
               "sampler": "TPE (seed 42, 8 random start-up trials)", "n_trials": len(trials),
               "search_space": SEARCH_SPACE, "best_value": study.best_value, "best_params": best,
               "search_seconds_this_run": t_search},
              open(OUT / "best_params.json", "w"), indent=2)

    # optimisation history plot
    fig, ax = plt.subplots(figsize=(7, 4))
    v = trials["value"].to_numpy()
    ax.plot(np.arange(len(v)), v, "o", alpha=.6, label="trial")
    ax.plot(np.arange(len(v)), np.maximum.accumulate(v), "-", color="C3", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("3-fold CV Top-1 accuracy"); ax.set_title("Optuna TPE search (LightGBM)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "optuna_history.png", dpi=150); plt.close(fig)

    # untuned vs tuned on identical 5 folds + student test
    st = pd.read_parquet(PROCESSED / "student_test_cands.parquet")
    st_obs = pd.read_parquet(PROCESSED / "student_test_obs.parquet")
    st_arcs = {t: load_trip(t, "student_test", with_route=False).arcs for t in list_trips("student_test")}
    untuned = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20,
               "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0, "reg_alpha": 0.0}
    rows = []
    for name, params in [("LightGBM (untuned)", untuned), ("LightGBM (Bayes-tuned)", best)]:
        cv, oof = cv_eval(params, tr, tr_obs, folds5, keep_oof=True)
        m = make(params)
        t0 = time.time()
        m.fit(tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy())
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(st[FEATURES].to_numpy(np.float32))[:, 1]
        inf = time.time() - t0
        d = st.assign(p=p)
        rows.append({"model": name, "cv_f1_macro": cv.f1_macro.mean(), "cv_f1_macro_std": cv.f1_macro.std(),
                     "cv_top1": cv.top1_acc.mean(), "cv_log_loss": cv.log_loss.mean(),
                     **{f"st_{k}": v for k, v in pair_metrics(d.label.to_numpy(), p).items()},
                     "st_top1_acc": top1_accuracy(d, st_obs, "p")["top1_acc"],
                     "st_route_continuity": tm.continuity_for(d, "p", st_arcs),
                     "fit_time_s": fit_s, "inference_ms_per_obs": 1000 * inf / len(st_obs)})
        print(rows[-1], flush=True)
        if "tuned" in name and "untuned" not in name:
            joblib.dump(m, MODELS / "final_model.joblib")
            np.savetxt(OUT / "confusion_tuned.csv", confusion(d.label.to_numpy(), p), fmt="%d", delimiter=",")
            tr[["trip_id", "obs", "arc", "label"]].assign(p_oof=oof).to_parquet(PROCESSED / "train_oof.parquet")
            st[["trip_id", "obs", "arc", "label"]].assign(p=p).to_parquet(PROCESSED / "student_test_pred.parquet")
            json.dump({"algorithm": "LightGBM binary classifier over (GPS observation, candidate arc) pairs",
                       "features": FEATURES, "params": best, "trained_on": "80 training trips (reliable labels)",
                       "metrics": rows[-1]}, open(MODELS / "final_model.json", "w"), indent=2)
            # genuine boosting-round learning curve on fold 0
            ti, vi = folds5[0]
            ev = {}
            mm = make(best)
            mm.fit(tr[FEATURES].iloc[ti].to_numpy(np.float32), tr["label"].iloc[ti].to_numpy(),
                   eval_set=[(tr[FEATURES].iloc[ti].to_numpy(np.float32), tr["label"].iloc[ti].to_numpy()),
                             (tr[FEATURES].iloc[vi].to_numpy(np.float32), tr["label"].iloc[vi].to_numpy())],
                   eval_names=["train", "valid"], eval_metric="binary_logloss",
                   callbacks=[record_evaluation(ev), log_evaluation(0)])
            pd.DataFrame({"iteration": np.arange(1, len(ev["train"]["binary_logloss"]) + 1),
                          "train_logloss": ev["train"]["binary_logloss"],
                          "valid_logloss": ev["valid"]["binary_logloss"]}).to_csv(OUT / "lgbm_learning_curve.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(pd.DataFrame(rows).round(4).T)


if __name__ == "__main__":
    main()
