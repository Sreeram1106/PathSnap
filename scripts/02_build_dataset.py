"""Build the candidate/feature tables for the train and student_test partitions.

Outputs (data/processed/):
  {partition}_cands.parquet  one row per (observation, candidate arc) with features + label
  {partition}_obs.parquet    one row per observation with reference arc + label quality
  results/eda/radius_sweep.csv  candidate miss rate vs search radius (train only)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import PROCESSED, RESULTS, CAND_RADIUS_M
from pathsnap.data_io import list_trips, load_trip
from pathsnap.features import prepare_trip, build_candidates
from pathsnap.labels import align_route


def process(trip_id, partition):
    t0 = time.time()
    trip = load_trip(trip_id, partition)
    proj, net, xy = prepare_trip(trip)
    ref, pos, ref_d = align_route(xy, net, trip.route)
    cands = build_candidates(trip, net, xy)
    cands.insert(0, "trip_id", trip_id)
    cands["label"] = (cands["arc"].to_numpy() == ref[cands["obs"].to_numpy()]).astype(np.int8)
    obs = pd.DataFrame({"trip_id": trip_id, "obs": np.arange(len(xy)), "t": trip.track[:, 2],
                        "lon": trip.track[:, 0], "lat": trip.track[:, 1],
                        "ref_arc": ref, "route_pos": pos, "ref_dist": ref_d})
    info = dict(trip.cleaning, trip_id=trip_id, partition=partition, seconds=time.time() - t0,
                n_cands=len(cands))
    return cands, obs, info


def main(n_jobs=4):
    (RESULTS / "eda").mkdir(parents=True, exist_ok=True)
    for part in ["train", "student_test"]:
        trips = list_trips(part)
        t0 = time.time()
        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process)(t, part) for t in trips)
        cands = pd.concat([o[0] for o in out], ignore_index=True)
        obs = pd.concat([o[1] for o in out], ignore_index=True)
        info = pd.DataFrame([o[2] for o in out])
        for c in cands.columns:
            if cands[c].dtype == np.float64:
                cands[c] = cands[c].astype(np.float32)
        cands.to_parquet(PROCESSED / f"{part}_cands.parquet", index=False)
        obs.to_parquet(PROCESSED / f"{part}_obs.parquet", index=False)
        info.to_csv(RESULTS / "eda" / f"{part}_cleaning_log.csv", index=False)
        hit = cands.loc[cands.label == 1, ["trip_id", "obs"]].drop_duplicates()
        print(f"{part}: {len(trips)} trips, {len(obs)} obs, {len(cands)} candidate rows, "
              f"cands/obs={len(cands)/len(obs):.2f}, miss rate={1-len(hit)/len(obs):.4f}, "
              f"{time.time()-t0:.0f}s")
        if part == "train":
            radii = [10, 20, 30, 40, 50, 75, 100, 150]
            sweep = pd.DataFrame({"radius_m": radii,
                                  "miss_rate_ignoring_cap": [(obs.ref_dist > r).mean() for r in radii]})
            sweep.to_csv(RESULTS / "eda" / "radius_sweep.csv", index=False)
            print(sweep)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
