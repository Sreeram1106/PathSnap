"""Reading and cleaning the raw trip files (.track, .nodes, .arcs, .route)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray            # (N, 3) lon, lat, t  -- cleaned, sorted by time
    nodes: np.ndarray            # (M, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from_node, to_node
    route: np.ndarray | None     # (R,) arc ids, only loaded for evaluation/labels
    cleaning: dict = field(default_factory=dict)


def _read(path: Path, dtype=float) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype)
    return df.to_numpy()


def list_trips(partition: str, root: Path = DATA_RAW) -> list[str]:
    d = Path(root) / partition
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def clean_track(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop invalid / duplicate observations and sort by time."""
    info = {"raw_points": int(len(raw))}
    ok = np.isfinite(raw).all(axis=1)
    ok &= (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    ok &= ~((raw[:, 0] == 0) & (raw[:, 1] == 0))
    info["invalid_dropped"] = int((~ok).sum())
    tr = raw[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    info["out_of_order"] = int((np.diff(tr[:, 2]) < 0).sum())
    tr = tr[order]
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    info["duplicate_time_dropped"] = int((~keep).sum())
    tr = tr[keep]
    info["clean_points"] = int(len(tr))
    return tr, info


def load_trip(trip_id: str, partition: str, root: Path = DATA_RAW, with_route: bool = True) -> Trip:
    d = Path(root) / partition / trip_id
    track, info = clean_track(_read(d / f"{trip_id}.track"))
    nodes = _read(d / f"{trip_id}.nodes")
    arcs = _read(d / f"{trip_id}.arcs", dtype=np.int64)
    route = None
    if with_route and (d / f"{trip_id}.route").exists():
        route = _read(d / f"{trip_id}.route", dtype=np.int64).ravel()
    return Trip(trip_id, partition, track, nodes, arcs, route, info)


def load_route(trip_id: str, partition: str, root: Path = DATA_RAW) -> np.ndarray | None:
    p = Path(root) / partition / trip_id / f"{trip_id}.route"
    return _read(p, dtype=np.int64).ravel() if p.exists() else None
