"""PathSnap FastAPI backend  (runs on LAPTOP 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + data status
  GET  /trips             the reserved evaluation trips (ids + sizes only)
  POST /predict           {"trip_id": "..."} -> full pipeline -> predicted route + accuracy

The prediction pipeline only receives the GPS track and the road network.
The reference route is read afterwards, server-side, purely to score the
prediction; it is never returned to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.config import DATA_RAW, MODELS  # noqa: E402
from pathsnap.data_io import list_trips, load_trip, load_route  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor, evaluate, route_geometry  # noqa: E402

# ------------------------------------------------------------------ settings
EVAL_ROOT = Path(os.environ.get("PATHSNAP_EVAL_ROOT", DATA_RAW))
EVAL_PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION",
                                "evaluator_test" if (EVAL_ROOT / "evaluator_test").exists() else "student_test")
MAX_GPS_POINTS_RETURNED = 2500

(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(ROOT / "logs" / "api.log", encoding="utf-8")])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching of noisy GPS tracks: LightGBM candidate scorer + HMM/Viterbi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = PathSnapPredictor(Path(os.environ.get("PATHSNAP_MODEL", MODELS / "final_model.joblib")),
                              MODELS / "decoder_config.json")
log.info("Model loaded: %s | decoder: %s", predictor.meta.get("algorithm", "?"), predictor.decoder)
log.info("Evaluation trips: %s / %s -> %s", EVAL_ROOT, EVAL_PARTITION, list_trips(EVAL_PARTITION, EVAL_ROOT))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.url.query or "", request.client.host if request.client else "?",
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["/health", "/trips", "/predict"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta.get("algorithm"), "params": predictor.meta.get("params"),
            "decoder": predictor.decoder, "evaluation_partition": EVAL_PARTITION,
            "n_trips": len(list_trips(EVAL_PARTITION, EVAL_ROOT))}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_PARTITION, EVAL_ROOT):
        t = load_trip(tid, EVAL_PARTITION, EVAL_ROOT, with_route=False)
        out.append({"trip_id": tid, "n_points": int(len(t.track)),
                    "duration_s": float(t.track[-1, 2] - t.track[0, 2]),
                    "n_road_arcs": int(len(t.arcs))})
    return {"partition": EVAL_PARTITION, "trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips(EVAL_PARTITION, EVAL_ROOT):
        raise HTTPException(404, f"unknown trip {tid}")
    t0 = time.perf_counter()
    log.info("PREDICT start trip=%s", tid)
    trip = load_trip(tid, EVAL_PARTITION, EVAL_ROOT, with_route=False)   # no ground truth here
    pred = predictor.predict(trip)
    infer_s = time.perf_counter() - t0
    ref = load_route(tid, EVAL_PARTITION, EVAL_ROOT)                     # server-side scoring only
    metrics = evaluate(pred, ref) if ref is not None else None
    total_s = time.perf_counter() - t0

    step = max(1, len(trip.track) // MAX_GPS_POINTS_RETURNED)
    gps = [[float(la), float(lo)] for lo, la in trip.track[::step, :2]]
    matched = pred["proj"].inv(np.array([pred["net"].A[a] + (pred["net"].B[a] - pred["net"].A[a]) * 0.5
                                         for a in pred["obs_arcs"][::step]]))
    if metrics:
        log.info("PREDICT done  trip=%s points=%d top1(HMM)=%.4f top1(clf)=%.4f continuity=%.4f route_F1=%.4f "
                 "inference=%.2fs total=%.2fs", tid, pred["n_obs"], metrics["top1_accuracy_hmm"],
                 metrics["top1_accuracy_classifier_only"], metrics["route_continuity_hmm"], metrics["route_f1"],
                 infer_s, total_s)
    return {
        "trip_id": tid,
        "n_points": pred["n_obs"],
        "n_candidates": pred["n_candidates"],
        "hmm_chain_breaks": pred["n_breaks"],
        "gps_points": gps,
        "predicted_route": route_geometry(pred),
        "predicted_route_arc_ids": [int(a) for a in pred["route"]],
        "matched_arc_midpoints": [[float(la), float(lo)] for lo, la in matched],
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                      "inference_total": round(infer_s, 3), "request_total": round(total_s, 3)},
    }
