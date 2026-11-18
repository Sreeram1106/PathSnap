"""Central configuration: paths, seeds and pipeline constants."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

SEED = 42
EARTH_R = 6371008.8  # mean Earth radius in metres

# Candidate generation (chosen from the radius sweep in scripts/01_eda.py)
CAND_RADIUS_M = 50.0
MAX_CANDS = 12
DENSIFY_STEP_M = 10.0

# Observations further than this from the aligned reference arc are flagged as
# unreliable reference labels (kept in evaluation, reported in EDA).
LABEL_MAX_DIST_M = 100.0

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "dist_share",
    "ctx_dist_mean", "ctx_dist_max", "ctx_minus_min", "ctx_rank",
    "heading_diff", "cos_heading", "heading_rank", "heading_valid",
    "speed", "dt_prev", "dt_next",
    "arc_len", "proj_frac", "end_dist", "out_deg", "in_deg",
]

for p in (PROCESSED, RESULTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)
