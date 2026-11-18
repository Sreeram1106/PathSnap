"""Evaluation metrics: candidate-level classification + observation/route-level."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, confusion_matrix)


def pair_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
    }


def top1_choice(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs)."""
    idx = df.groupby(["trip_id", "obs"])[score_col].idxmax()
    return df.loc[idx, ["trip_id", "obs", "arc"]]


def top1_accuracy(df: pd.DataFrame, obs_df: pd.DataFrame, score_col: str) -> dict:
    """obs_df: one row per observation with trip_id, obs, ref_arc (all observations)."""
    ch = top1_choice(df, score_col).rename(columns={"arc": "pred_arc"})
    m = obs_df.merge(ch, on=["trip_id", "obs"], how="left")
    correct = (m["pred_arc"] == m["ref_arc"])
    has = m["pred_arc"].notna()
    return {"top1_acc": correct.mean(), "top1_acc_with_cands": correct[has].mean()}


def candidate_miss_rate(df: pd.DataFrame, obs_df: pd.DataFrame) -> float:
    hit = df.loc[df["label"] == 1, ["trip_id", "obs"]].drop_duplicates()
    return 1.0 - len(hit) / len(obs_df)


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive observation pairs whose predicted arcs are the same arc
    or directly connected (end node of the first == start node of the second)."""
    p = pred_arcs[~pd.isna(pred_arcs)].astype(int)
    if len(p) < 2:
        return np.nan
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_overlap(pred_route: list, ref_route: np.ndarray, length: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 between predicted and reference arc sets."""
    P, R = set(int(x) for x in pred_route), set(int(x) for x in ref_route)
    inter = sum(length[list(P & R)]) if P & R else 0.0
    lp = sum(length[list(P)]) if P else 0.0
    lr = sum(length[list(R)]) if R else 0.0
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def confusion(y, p, thr=0.5):
    return confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
