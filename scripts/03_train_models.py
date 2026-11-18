"""Train and compare the candidate-scoring models.

* 5-fold GroupKFold over the 80 training trips (a trip never spans two folds)
* every model is then refit on all 80 trips and evaluated on the 10 student-test trips
* results -> results/models/
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
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import PROCESSED, RESULTS, MODELS, FEATURES, SEED, LABEL_MAX_DIST_M
from pathsnap.data_io import list_trips, load_trip
from pathsnap.metrics import pair_metrics, top1_accuracy, top1_choice, route_continuity, confusion
from pathsnap.models import MODEL_SPECS

OUT = RESULTS / "models"
OUT.mkdir(parents=True, exist_ok=True)
(MODELS / "experiments").mkdir(parents=True, exist_ok=True)


def subsample_obs(df, max_rows, rng):
    """Subsample whole observations (all their candidates) to about max_rows rows."""
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df[["trip_id", "obs"]].drop_duplicates()
    frac = max_rows / len(df)
    keep = keys.sample(frac=frac, random_state=rng)
    return df.merge(keep, on=["trip_id", "obs"])


def continuity_for(df, score_col, arcs_by_trip):
    vals, n = 0.0, 0
    ch = top1_choice(df, score_col)
    for tid, g in ch.groupby("trip_id"):
        g = g.sort_values("obs")
        c = route_continuity(g["arc"].to_numpy(), arcs_by_trip[tid])
        vals += c * (len(g) - 1)
        n += len(g) - 1
    return vals / n


def load_train():
    """Training rows, excluding observations whose reference label is unreliable
    (GPS more than LABEL_MAX_DIST_M from the aligned reference arc, see EDA)."""
    tr = pd.read_parquet(PROCESSED / "train_cands.parquet")
    tr_obs = pd.read_parquet(PROCESSED / "train_obs.parquet")
    tr_obs = tr_obs[tr_obs.ref_dist <= LABEL_MAX_DIST_M]
    tr = tr.merge(tr_obs[["trip_id", "obs"]], on=["trip_id", "obs"]).reset_index(drop=True)
    print(f"train: {len(tr_obs)} reliable observations, {len(tr)} candidate rows")
    return tr, tr_obs


def main(only=None):
    tr, tr_obs = load_train()
    st = pd.read_parquet(PROCESSED / "student_test_cands.parquet")
    st_obs = pd.read_parquet(PROCESSED / "student_test_obs.parquet")
    st_arcs = {t: load_trip(t, "student_test", with_route=False).arcs for t in list_trips("student_test")}
    X, y, groups = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(gkf.split(X, y, groups))
    json.dump({"n_splits": 5, "seed": SEED,
               "folds": [sorted(set(groups[v])) for _, v in folds]}, open(OUT / "cv_folds.json", "w"), indent=1)

    fold_rows, summary = [], []
    specs = {k: v for k, v in MODEL_SPECS.items() if only is None or k in only}
    for name, spec in specs.items():
        print(f"\n=== {name}", flush=True)
        for k, (ti, vi) in enumerate(folds):
            dtr = subsample_obs(tr.iloc[ti], spec["max_train_rows"], SEED)
            m = spec["make"]()
            t0 = time.time()
            m.fit(dtr[FEATURES].to_numpy(np.float32), dtr["label"].to_numpy())
            fit_s = time.time() - t0
            dva = tr.iloc[vi].copy()
            dva["p"] = m.predict_proba(dva[FEATURES].to_numpy(np.float32))[:, 1]
            # train-fold score on the same subsample, to look for over/under-fitting
            ptr = m.predict_proba(dtr[FEATURES].to_numpy(np.float32))[:, 1]
            r = {"model": name, "fold": k, "fit_s": fit_s,
                 "train_f1_macro": pair_metrics(dtr["label"].to_numpy(), ptr)["f1_macro"],
                 **pair_metrics(dva["label"].to_numpy(), dva["p"].to_numpy()),
                 **top1_accuracy(dva, tr_obs[tr_obs.trip_id.isin(set(groups[vi]))], "p")}
            print({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in r.items()}, flush=True)
            fold_rows.append(r)

        # final fit on all 80 training trips -> student test
        dtr = subsample_obs(tr, spec["max_train_rows"], SEED)
        m = spec["make"]()
        t0 = time.time()
        m.fit(dtr[FEATURES].to_numpy(np.float32), dtr["label"].to_numpy())
        fit_s = time.time() - t0
        ptr = m.predict_proba(dtr[FEATURES].to_numpy(np.float32))[:, 1]
        t0 = time.time()
        p = m.predict_proba(st[FEATURES].to_numpy(np.float32))[:, 1]
        inf_s = time.time() - t0
        dst = st.assign(p=p)
        pm = pair_metrics(dst["label"].to_numpy(), p)
        t1 = top1_accuracy(dst, st_obs, "p")
        cont = continuity_for(dst, "p", st_arcs)
        cm = confusion(dst["label"].to_numpy(), p)
        np.savetxt(OUT / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",")
        joblib.dump(m, MODELS / "experiments" / f"{name}.joblib", compress=3)
        cvf = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        row = {"model": name, "family": spec["family"], "config": spec["config"],
               "train_rows": len(dtr),
               "cv_f1_macro_mean": cvf.f1_macro.mean(), "cv_f1_macro_std": cvf.f1_macro.std(),
               "cv_top1_mean": cvf.top1_acc.mean(), "cv_top1_std": cvf.top1_acc.std(),
               "cv_log_loss_mean": cvf.log_loss.mean(),
               "train_f1_macro_full": pair_metrics(dtr["label"].to_numpy(), ptr)["f1_macro"],
               **{f"st_{k}": v for k, v in pm.items()},
               "st_top1_acc": t1["top1_acc"], "st_route_continuity": cont,
               "fit_time_s": fit_s, "inference_ms_per_obs": 1000 * inf_s / len(st_obs)}
        summary.append(row)
        print(row, flush=True)

    fr = pd.DataFrame(fold_rows)
    sm = pd.DataFrame(summary)
    if only is not None and (OUT / "model_comparison.csv").exists():
        old = pd.read_csv(OUT / "model_comparison.csv")
        sm = pd.concat([old[~old.model.isin(sm.model)], sm], ignore_index=True)
        oldf = pd.read_csv(OUT / "cv_fold_results.csv")
        fr = pd.concat([oldf[~oldf.model.isin(fr.model)], fr], ignore_index=True)
    fr.to_csv(OUT / "cv_fold_results.csv", index=False)
    sm.to_csv(OUT / "model_comparison.csv", index=False)
    print(sm[["model", "cv_f1_macro_mean", "cv_top1_mean", "st_f1_macro", "st_top1_acc", "st_log_loss",
              "st_route_continuity", "inference_ms_per_obs"]].round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
