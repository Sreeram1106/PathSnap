"""Candidate generation and feature engineering for one trip.

Every row of the output table is one (GPS observation, candidate arc) pair.
Only GPS observations and the road network are used -- never the route.
"""
import numpy as np
import pandas as pd

from .config import CAND_RADIUS_M, MAX_CANDS, DENSIFY_STEP_M
from .geo import LocalProjection, angle_diff
from .network import RoadNetwork


def motion_features(xy: np.ndarray, t: np.ndarray):
    N = len(xy)
    dt = np.diff(t)
    dt_prev = np.r_[np.nan, dt]
    dt_next = np.r_[dt, np.nan]
    i0 = np.clip(np.arange(N) - 2, 0, N - 1)
    i1 = np.clip(np.arange(N) + 2, 0, N - 1)
    v = xy[i1] - xy[i0]
    disp = np.hypot(v[:, 0], v[:, 1])
    heading = np.arctan2(v[:, 0], v[:, 1])
    span = np.maximum(t[i1] - t[i0], 1e-6)
    speed = disp / span
    heading_valid = (disp > 5.0).astype(float)
    return dict(dt_prev=np.nan_to_num(dt_prev, nan=1.0), dt_next=np.nan_to_num(dt_next, nan=1.0),
                speed=speed, heading=heading, heading_valid=heading_valid)


def _group_rank(key, val):
    """Rank (0 = smallest) of val within groups of key (key sorted)."""
    s = pd.Series(val)
    return s.groupby(key).rank(method="first").to_numpy() - 1


def prepare_trip(trip):
    proj = LocalProjection.for_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.fwd(trip.track[:, :2])
    net.build_index(xy, CAND_RADIUS_M, DENSIFY_STEP_M)
    return proj, net, xy


def build_candidates(trip, net, xy, radius=CAND_RADIUS_M, max_k=MAX_CANDS) -> pd.DataFrame:
    t = trip.track[:, 2]
    N = len(xy)
    pi, ai, d, frac = net.candidates(xy, radius, max_k)
    mf = motion_features(xy, t)
    df = pd.DataFrame({"obs": pi, "arc": ai, "dist": d, "proj_frac": frac})
    if df.empty:
        return df
    g = df.groupby("obs")
    df["n_cands"] = g["arc"].transform("size")
    df["dist_rank"] = _group_rank(df["obs"], df["dist"])
    dmin = g["dist"].transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)
    w = np.exp(-0.5 * (df["dist"] / 10.0) ** 2)
    df["dist_share"] = w / w.groupby(df["obs"]).transform("sum").clip(lower=1e-12)

    # context: how close the neighbouring observations are to the same arc
    ctx = []
    for k in (-3, -2, -1, 1, 2, 3):
        j = np.clip(df["obs"].to_numpy() + k, 0, N - 1)
        dk, _ = net.dist_to_arcs(xy[j], df["arc"].to_numpy())
        ctx.append(dk)
    ctx = np.vstack(ctx)
    df["ctx_dist_mean"] = ctx.mean(0)
    df["ctx_dist_max"] = ctx.max(0)
    df["ctx_minus_min"] = df["ctx_dist_mean"] - df.groupby("obs")["ctx_dist_mean"].transform("min")
    df["ctx_rank"] = _group_rank(df["obs"], df["ctx_dist_mean"])

    o = df["obs"].to_numpy()
    a = df["arc"].to_numpy()
    hd = angle_diff(mf["heading"][o], net.bearing[a])
    valid = mf["heading_valid"][o]
    df["heading_diff"] = np.where(valid > 0, np.degrees(hd), 90.0)
    df["cos_heading"] = np.where(valid > 0, np.cos(hd), 0.0)
    df["heading_rank"] = _group_rank(df["obs"], df["heading_diff"])
    df["heading_valid"] = valid
    df["speed"] = mf["speed"][o]
    df["dt_prev"] = mf["dt_prev"][o]
    df["dt_next"] = mf["dt_next"][o]
    df["arc_len"] = net.length[a]
    df["end_dist"] = np.minimum(df["proj_frac"], 1 - df["proj_frac"]) * df["arc_len"]
    df["out_deg"] = net.out_deg[net.arcs[a, 1]]
    df["in_deg"] = net.in_deg[net.arcs[a, 0]]
    df["t"] = t[o]
    return df
