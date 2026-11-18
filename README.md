# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University). PathSnap takes a noisy GPS track and the trip's road network and reconstructs the road segment (arc) the vehicle was on at every GPS observation. It also rebuilds the full driven route.

```
raw .track/.nodes/.arcs ──► cleaning + local metric projection
                         ──► candidate arcs within 50 m of every GPS point
                         ──► 22 engineered features per (point, candidate)
                         ──► LightGBM candidate scorer (Bayes-tuned with Optuna TPE)
                         ──► HMM + Viterbi over candidates (network shortest paths)
                         ──► predicted arc per observation + connected route
                         ──► FastAPI (Laptop 1)  ◄──HTTP──►  Streamlit evaluator UI (Laptop 2)
```

## Repository layout

| Folder | What is inside |
|---|---|
| `pathsnap/` | Library code: `data_io.py` (reading/cleaning), `geo.py` (projection, geometry), `network.py` (spatial index, Dijkstra), `labels.py` (route→observation label alignment), `features.py` (candidates + feature engineering), `models.py` (model zoo), `hmm.py` (HMM/Viterbi + route reconstruction), `metrics.py`, `pipeline.py` (inference + server-side evaluation) |
| `scripts/` | Numbered, reproducible experiment scripts (see below) |
| `api/` | FastAPI backend (`api/main.py`), runs on **Laptop 1** |
| `ui/` | Streamlit evaluator interface (`ui/app.py`), runs on **Laptop 2** |
| `models/` | `final_model.joblib` (tuned LightGBM), `final_model.json` (features, params, metrics), `decoder_config.json` (HMM parameters), `experiments/` (the other compared models) |
| `results/eda/` | Dataset inventory, summary statistics, data-quality record, EDA plots |
| `results/models/` | Per-fold CV results, model comparison table, confusion matrices, MLP epoch history |
| `results/optuna/` | Bayesian search record (`optuna_study.db`, `optuna_trials.csv`), best params, tuned vs untuned, learning curve |
| `results/hmm/` | Decoder grid search, classifier-only vs HMM per student-test trip |
| `results/maps/` | Interactive Folium HTML maps (EDA + predictions) |
| `results/plots/` | All comparison plots used in the report |
| `report/` | Technical report (PDF) and the script that builds it |
| `logs/` | Console logs of every experiment run |

## 1. Installation (Laptop 1: training + API)

Python 3.11 is recommended.

```bash
git clone https://github.com/Sreeram1106/PathSnap.git
cd PathSnap
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the organisers' data in `data/raw/` with one folder per partition. The data is not redistributed in this repo.

```
data/raw/train/00000001/00000001.{track,nodes,arcs,route}
data/raw/student_test/...
data/raw/evaluator_test/...      <- the 10 reserved trips, on evaluation day
```

## 2. Reproduce the experiments

Run the scripts in this order from the repo root. Each one writes its outputs under `results/` and `models/`.

```bash
python scripts/02_build_dataset.py      # candidates, features, labels  -> data/processed/
python scripts/01_eda.py                # inventory, data quality, plots, EDA maps
python scripts/03_train_models.py       # 7 models, 5-fold GroupKFold CV + student-test
python scripts/03b_mlp_history.py       # epoch-wise train/val loss for the MLP
python scripts/04_bayes_opt.py 25       # Optuna TPE search on LightGBM, final model
python scripts/05_hmm_eval.py           # decoder tuning + classifier-only vs HMM
python scripts/06_maps.py               # Folium maps of predictions on student-test trips
python scripts/07_plots.py              # comparison plots
python report/build_report.py           # technical report PDF
```

All randomness is seeded (`SEED = 42` in `pathsnap/config.py`). Cross-validation groups by trip, so no trip appears in two folds. The evaluator partition is never used for training, feature selection, tuning or decoder settings.

## 3. Deployment on two laptops

| Laptop | Runs | Needs |
|---|---|---|
| **Laptop 1: Model + API** | `api/main.py`: classifier, HMM/Viterbi, road networks, server-side scoring | the repo, `requirements.txt`, the data in `data/raw/` |
| **Laptop 2: UI** | `ui/app.py`: lists the 10 evaluation trips, calls the API, draws the result | the repo, `requirements-ui.txt` only (no data, no model) |

Connect both laptops to the same Wi-Fi or hotspot.

### Laptop 1 (API)

```bash
# find this laptop's IP address:  ipconfig  (IPv4 Address, e.g. 192.168.1.20)
set PATHSNAP_EVAL_PARTITION=evaluator_test        # folder under data/raw holding the 10 reserved trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If Windows Firewall asks, allow Python on private networks. You can also run this once in an admin terminal:
`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`

Every request is logged in this console and in `logs/api.log`. The log shows the trip, the client IP, the accuracy and the timings, so you can show these API logs to the evaluator. Interactive API docs are at `http://<laptop1-ip>:8000/docs`.

### Laptop 2 (UI)

```bash
git clone https://github.com/Sreeram1106/PathSnap.git
cd PathSnap
pip install -r requirements-ui.txt
streamlit run ui/app.py
```

The UI opens at `http://localhost:8501`. Type Laptop 1's address in the sidebar (e.g. `http://192.168.1.20:8000`), pick one of the 10 trips and click **Run map-matching**.

### API endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | model, decoder parameters, evaluation partition |
| GET | `/trips` | none | the reserved trip ids with point counts (no labels) |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, predicted route polyline + arc ids, metrics, timings |

Test from any terminal:
```bash
curl -X POST http://<laptop1-ip>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000000\"}"
```

**Ground-truth handling:** `PathSnapPredictor.predict()` receives only the GPS track and the road network. The `.route` file is read *after* prediction, inside the API, only to compute accuracy. The reference route is never sent to the UI. Predictions are never hardcoded: every request runs the full pipeline.

## 4. Method summary

* **Pre-processing:** drop invalid coordinates and duplicate timestamps, then sort by time. Each trip gets its own local equirectangular projection to metres (trips are worldwide; extents are city-scale).
* **Labels:** `.route` is an ordered arc list, so each observation is aligned to a route position with a monotone dynamic programme. Observations more than 100 m from their aligned arc are treated as unreliable labels and left out of training.
* **Candidates:** up to 12 arcs within 50 m, found with a KD-tree over arcs densified every 10 m.
* **Features (22):** perpendicular distance and its rank/ratio/softmax share; distance of ±3 neighbouring fixes to the same arc; heading difference between GPS motion and arc direction; speed; time gaps; arc length; projection position; node degrees.
* **Models compared:** Logistic Regression, Gaussian NB, KNN, Decision Tree, Random Forest, LightGBM, MLP.
* **Selection metric:** observation-level Top-1 accuracy, with Macro F1, log loss and route continuity as secondary checks.
* **Bayesian optimisation:** Optuna TPE, 8 hyper-parameters, objective = 3-fold GroupKFold Top-1.
* **HMM:** the hidden states are the candidate arcs. Emission = classifier probability (renormalised). Transition = `exp(-|d_network − d_gps| / β)`, using Dijkstra. Time gaps over 60 s or unreachable steps start a new chain. Decoder parameters are tuned on out-of-fold probabilities of training trips.

See `report/PathSnap_Technical_Report.pdf` for full results.

## 5. Results (copied from `results/`)

**Model comparison:** CV = 5-fold GroupKFold over 80 train trips; ST = 10 student-test trips; classifier only.

| Model | CV Macro F1 | CV Top-1 | ST Top-1 | ST Macro F1 | ST Log loss | ST Route continuity |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9041 | 0.8424 | 0.8537 | 0.9059 | 0.1076 | 0.9413 |
| Gaussian NB | 0.8157 | 0.6735 | 0.6864 | 0.8138 | 1.0412 | 0.8910 |
| KNN (k=31) | 0.9108 | 0.8459 | 0.8660 | 0.9167 | 0.1153 | 0.9391 |
| Decision Tree | 0.9234 | 0.8590 | 0.8891 | 0.9338 | 0.0783 | 0.9495 |
| Random Forest | 0.9275 | 0.8651 | 0.8828 | 0.9340 | 0.0717 | 0.9541 |
| **LightGBM** | 0.9269 | 0.8648 | **0.8941** | **0.9361** | 0.0727 | 0.9510 |
| MLP (64,32) | 0.9233 | 0.8606 | 0.8783 | 0.9270 | 0.0783 | 0.9510 |

**Bayesian optimisation** (Optuna TPE, 25 trials) of LightGBM:

| | CV Macro F1 | CV Top-1 | ST Top-1 | ST Macro F1 | ST Log loss |
|---|---|---|---|---|---|
| Untuned | 0.9269 | 0.8648 | 0.8941 | 0.9361 | 0.0727 |
| Tuned | 0.9282 | 0.8661 | 0.8991 | 0.9385 | 0.0674 |

**HMM/Viterbi** (10 student-test trips):

| | Top-1 | Route continuity | Route F1 (length-weighted) |
|---|---|---|---|
| Classifier only | 0.8999 | 0.9538 | 0.8764 |
| Classifier + HMM | **0.9408** | **0.9788** | **0.9872** |

## 6. AI-assistance disclosure

As required by the event's policy: an AI coding assistant (Claude) helped write the code and the first draft of the report. Every experiment was actually run. All numbers, tables and plots come from the committed scripts and logs. The team has reviewed the work and must be able to explain all of it.
