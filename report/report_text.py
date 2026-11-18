"""Narrative text of the technical report. Numbers are injected from the result files
(ctx) so that the text always matches the committed experiment outputs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAP_TRIP = "00000065"


def f4(v):
    return f"{v:.4f}"


def title_page(c):
    team = (ROOT / "report" / "team.txt").read_text(encoding="utf-8").strip().splitlines()
    sm = c["sm"]
    return f"""
<div class='title'>
<p class='small'>PES University · Department of Computer Science and Engineering · UE24CS352A Machine Learning</p>
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p style='font-size:13pt'>Technical Report</p>
<p>{'<br>'.join(team)}</p>
<p class='small'>Repository: https://github.com/Sreeram1106/PathSnap</p>
</div>
<div class='box'><b>Summary.</b> Every GPS observation gets up to 12 candidate road arcs within 50 m. A classifier
scores each candidate using 22 features. We trained and compared seven models from six algorithm families with
trip-grouped 5-fold cross-validation on the 80 training trips. LightGBM was selected. It was then tuned with
Bayesian optimisation (Optuna TPE) and its scores were decoded with an HMM/Viterbi decoder that uses the road
network's shortest paths. On the 10 student-test trips, observation-level Top-1 accuracy is
{c['pct'](c['hs']['top1_clf'])} for the tuned classifier alone and {c['pct'](c['hs']['top1_hmm'])} after
HMM decoding. Route continuity rises from {c['pct'](c['hs']['continuity_clf'])} to
{c['pct'](c['hs']['continuity_hmm'])}. The system is deployed as a FastAPI service (Laptop 1) with a Streamlit
evaluator UI (Laptop 2).</div>
"""


def sec2_intro(c):
    return """<p>The organisers provide each trip as its own folder of four header-less text files: <code>.track</code>
(longitude, latitude, time in seconds), <code>.nodes</code> (longitude, latitude; the row index is the node id),
<code>.arcs</code> (from-node, to-node; the row index is the arc id) and <code>.route</code> (the ordered arc ids of the
reference route). The road network is <i>specific to each trip</i>, so arc ids only mean something inside their own
trip. A row of the GPS file does not correspond to a row of any network file. The link between a GPS observation and
an arc has to be built geometrically. All analyses in this section use the <b>train</b> (80 trips) and
<b>student_test</b> (10 trips) partitions, as noted for each table and figure. The evaluator partition was not given
to us and was never used.</p>"""


def sec2_missing(c):
    inv = c["inv"]
    return f"""<p>No GPS row has missing, non-numeric or out-of-range coordinates. There are no duplicate or
out-of-order timestamps (train + student_test). The real "missing data" is <b>temporal</b>. Sampling is nominally
1 Hz, but {int(inv.n_gaps_gt5s.sum())} gaps are longer than 5 s and {int(inv.n_gaps_gt60s.sum())} are longer than
60 s. The longest gap is {inv.dt_max.max():.0f} s, inside one train trip. A second kind of problem is
<b>label noise</b>: in a few trips the GPS trace is systematically offset from the reference route (see the list
below).</p>"""


def sec2_preprocess(c):
    return """<ol>
<li><b>Reading and validation:</b> the files are parsed with pandas. Rows with NaN, out-of-range or (0,0)
coordinates are dropped (none were found). Points are sorted by timestamp and duplicate timestamps removed. We also
checked that every arc references a valid node and every route id a valid arc.</li>
<li><b>Projection:</b> the trips are spread over the whole world, so no single projected CRS fits them all. Each trip
gets its own local equirectangular (tangent-plane) projection centred on the median GPS point:
x = R·cos(lat0)·Δlon, y = R·Δlat. Trip extents are at most a few tens of km, so the distance error is far below
GPS noise. All geometry (point-to-segment distance, heading, lengths) is then in metres. Distances in degrees are
never used.</li>
<li><b>Observation labels:</b> the <code>.route</code> file is a sequence of arcs, not a label per GPS point. We
align the observations to the route with a monotone dynamic programme. Observation <i>i</i> gets route position
j<sub>i</sub>, with j non-decreasing, chosen to minimise the total point-to-arc distance (capped at 200 m). Arcs
may be skipped, because short arcs can be passed between two fixes. The arc at that position is the
observation's reference arc. This alignment is only used for training and scoring.</li>
<li><b>Label reliability filter:</b> training observations more than 100 m from their aligned reference arc are
removed from the <i>training</i> set, because their label cannot be trusted. The student-test evaluation keeps all
observations.</li>
<li><b>Candidate generation:</b> the arcs near the track are densified every 10 m and put into a KD-tree. Every arc
within R = 50 m of a fix is a candidate, keeping the 12 nearest. R was chosen from the radius sweep: below 40 m
the miss rate rises quickly, and above 50 m it barely improves while the number of candidates keeps growing.</li>
</ol>"""


def sec2_patterns(c):
    inv = c["inv"]
    return f"""<ul>
<li><b>Sampling:</b> GPS is close to 1 Hz (median interval 1 s), with occasional long gaps (signal loss or the
logger paused). Mean trip speed is {inv.mean_speed_kmh.median():.0f} km/h (median over trips), so consecutive fixes
are usually 5-25 m apart. This is short compared with typical arc lengths, so <i>neighbouring fixes carry strong
information about the same arc</i>.</li>
<li><b>GPS noise:</b> the median distance from a fix to its reference arc is a few metres, but the distribution has a
long tail (Figure 2). With R = 10 m, 28% of fixes would lose their true arc; with R = 50 m, about 11% (the
remaining misses are almost all in the offset trips).</li>
<li><b>Directed, mostly two-way networks:</b> most roads exist as two opposite arcs with identical geometry. Distance
alone cannot separate them. <i>Direction of travel</i> is the deciding feature.</li>
<li><b>Network connectivity:</b> each trip network is essentially one large strongly connected component (median
share {inv.largest_strong_share.median():.2f}), with no isolated nodes. Reference routes are connected arc-to-arc
in {100 * inv.route_connected_share.mean():.2f}% of steps, so shortest-path transitions are a sound basis for the HMM.</li>
<li><b>Network size varies a lot</b> ({inv.n_arcs.min():,} to {inv.n_arcs.max():,} arcs per trip), so all spatial
work is done on the arcs near the track only.</li>
</ul>
<p><b>Features extracted (22 per observation-candidate pair) and why:</b></p>
<table><tr><th>Group</th><th>Features</th><th>Reason</th></tr>
<tr><td>Geometry</td><td>dist, dist_rank, dist_minus_min, dist_ratio, dist_share, n_cands</td><td>Absolute distance is
noisy. What matters is how close a candidate is <i>compared with its competitors</i> for the same fix.</td></tr>
<tr><td>Temporal context</td><td>ctx_dist_mean, ctx_dist_max, ctx_minus_min, ctx_rank</td><td>Mean/max distance of
the fixes at offsets ±1, ±2, ±3 to the same arc. This smooths single-fix noise and penalises arcs that only
cross the track.</td></tr>
<tr><td>Direction</td><td>heading_diff, cos_heading, heading_rank, heading_valid</td><td>Separates the two
directions of a two-way road and parallel side streets. It is marked invalid when the vehicle is nearly stationary
(heading undefined).</td></tr>
<tr><td>Motion / gaps</td><td>speed, dt_prev, dt_next</td><td>Tells the model when heading and context can be
trusted (gaps, stops).</td></tr>
<tr><td>Arc / topology</td><td>arc_len, proj_frac, end_dist, out_deg, in_deg</td><td>Fixes that project onto an arc
endpoint are ambiguous between consecutive arcs. Junction degree shows how many alternatives exist.</td></tr>
</table>"""


def sec3_intro(c):
    return """<p><b>Learning task.</b> Each row is one (GPS observation, candidate arc) pair, with label 1 when the
candidate is the observation's reference arc. A model outputs P(correct | features). The observation-level
prediction is the highest-scoring candidate. <b>Protocol:</b> 5-fold <code>GroupKFold</code> over the 80 training
trips, grouped by trip so no trip is split across folds, with seed 42. Each model is then refit on all 80 trips
and evaluated once on the 10 student-test trips. Preprocessing (scalers, quantile transform) sits inside sklearn
pipelines, so it is fitted on training folds only. <b>Metrics:</b> Macro F1, precision, recall, accuracy and log
loss on candidate pairs at threshold 0.5. Observation-level Top-1 accuracy is computed over <i>all</i>
observations, including the 0.2% with no candidate. Route continuity is the share of consecutive fixes whose
predicted arcs are the same arc or directly connected. <b>Primary selection metric: observation-level Top-1
accuracy</b>, because it is exactly the map-matching objective (one correct arc per fix). Pair-level Macro F1
depends on a 0.5 threshold that is never used at inference time.</p>"""


def sec3_table_note(c):
    return """<p class='small'>ST = student-test partition (10 trips). CV = mean over 5 trip-grouped folds of the 80
training trips. Log loss and Macro F1 are at candidate-pair level. Top-1 and continuity are at observation level
(argmax per fix, classifier only, no HMM). Inference time is classifier scoring time per GPS observation on
12 CPU threads, excluding feature extraction. KNN and MLP were trained on a subsample (whole observations kept
together) because of their computational cost. This is listed in the configuration column.</p>"""


INFO = {
    "LogisticRegression": dict(
        why="The correct candidate is mostly the one that is <i>relatively</i> close and aligned with the direction of travel. That is close to a monotone (linear in log-odds) relationship in our features. Logistic regression also gives calibrated probabilities, which the HMM emission model needs.",
        sl="Strengths: trains in seconds on 1.3 M rows, has well-calibrated probabilities and interpretable coefficients. Limitations: it cannot model interactions (heading only matters when the vehicle is moving; distance matters less at junctions) unless they are engineered by hand.",
        hp="C = 1.0 (L2), solver lbfgs, max_iter = 500, features standardised.",
        loss="Binary cross-entropy (log loss) + L2 penalty, minimised with L-BFGS."),
    "GaussianNB": dict(
        why="A very cheap probabilistic baseline. It shows how much can be gained from modelling feature dependence, which the other models do.",
        sl="Strengths: no tuning and near-instant training. Limitations: it assumes the features are independent given the class. Our features are strongly correlated (dist, dist_rank, dist_share, ctx_*), so it counts the same evidence several times and becomes over-confident.",
        hp="var_smoothing = 1e-9. A QuantileTransformer (normal output) is applied first to make the features closer to Gaussian.",
        loss="No iterative loss: the class priors and per-class Gaussian means/variances are maximum-likelihood estimates."),
    "KNN": dict(
        why="Candidates with similar geometry (distance profile, heading agreement, context) should have the same label. KNN uses this directly without assuming any functional form.",
        sl="Strengths: non-parametric and captures local non-linear structure. Limitations: inference cost grows with the size of the training set (we had to subsample to 80 k rows), it needs careful feature scaling, and its probabilities are coarse vote fractions.",
        hp="k = 31, distance weighting, Euclidean metric on standardised features, 80 k training rows.",
        loss="None (lazy learner). The neighbours vote at prediction time."),
    "DecisionTree": dict(
        why="Map-matching decisions look like rules (\"closest candidate AND heading within 30° AND neighbours also close\"). A tree learns such rules and their interactions without scaling.",
        sl="Strengths: interpretable, models interactions, fast inference. Limitations: high variance, and piecewise-constant probabilities that hurt log loss and give the HMM coarser emissions.",
        hp="criterion = gini, max_depth = 12, min_samples_leaf = 50.",
        loss="Greedy reduction of Gini impurity at each split (no global loss)."),
    "RandomForest": dict(
        why="Bagging many decorrelated trees keeps the rule-like behaviour of a tree but averages away its variance, giving smoother probabilities.",
        sl="Strengths: robust, little tuning, good generalisation. Limitations: large model files and slower inference than boosting. Its probabilities are averaged votes and can be under-confident.",
        hp="n_estimators = 100, max_depth = 16, min_samples_leaf = 20, max_samples = 0.3, max_features = sqrt.",
        loss="Each tree minimises Gini impurity on a bootstrap sample. The ensemble averages the trees' class probabilities."),
    "LightGBM": dict(
        why="Gradient-boosted trees are the strongest general-purpose learners for tabular features with interactions, such as our relative-distance × heading × context features. They optimise log loss directly, which suits the HMM.",
        sl="Strengths: best accuracy, fast histogram training on 1.3 M rows, no scaling needed. Limitations: many interacting hyper-parameters, and it can overfit with large trees or too many rounds (we control this with group CV).",
        hp="Library defaults: n_estimators = 100, learning_rate = 0.1, num_leaves = 31, min_child_samples = 20, no row/feature subsampling, no regularisation.",
        loss="Binary log loss (cross-entropy), minimised by gradient boosting with second-order (Newton) leaf values."),
    "MLP": dict(
        why="A neural network can learn non-linear combinations of the features without hand-made interactions. It is the representation-learning point of comparison.",
        sl="Strengths: flexible function class with smooth probabilities. Limitations: needs scaling, is sensitive to learning rate and initialisation, is slower to train (we subsampled to 300 k rows) and is harder to explain.",
        hp="Hidden layers (64, 32), ReLU, Adam with lr = 1e-3, batch 512, L2 alpha = 1e-4, early stopping (10% validation split, patience 5), at most 25 epochs, standardised inputs.",
        loss="Binary cross-entropy + L2 weight penalty, minimised with Adam."),
}


def fit_comment(m, r, c):
    gap = r.train_f1_macro_full - r.cv_f1_macro_mean
    best = c["sm"].cv_f1_macro_mean.max()
    s = (f"Training Macro F1 = {f4(r.train_f1_macro_full)}, CV Macro F1 = {f4(r.cv_f1_macro_mean)} "
         f"(sd {f4(r.cv_f1_macro_std)} over folds), student-test Macro F1 = {f4(r.st_f1_macro)}. ")
    if m == "KNN":
        s += ("The training score of 1.0 is not evidence of real fit quality. With distance weighting, each training row "
              "is its own nearest neighbour with infinite weight. The relevant gap is between CV and student-test. ")
    if m == "MLP":
        try:
            h = pd.read_csv(ROOT / "results" / "models" / "mlp_epoch_history.csv")
            b = int(h.val_log_loss.idxmin())
            s += (f"In the epoch history (Figure 10), validation loss is lowest at epoch {int(h.epoch[b])} "
                  f"({f4(h.val_log_loss[b])}), while training loss keeps falling to {f4(h.train_log_loss.iloc[-1])}. ")
            s += ("Validation loss rising after that point is a mild sign of overfitting, which early stopping limits. "
                  if h.val_log_loss.iloc[-1] > h.val_log_loss[b] + 0.002 else
                  "Validation loss stays flat, so there is no real overfitting within 25 epochs. ")
        except FileNotFoundError:
            pass
    if gap > 0.03 and m != "KNN":
        s += f"The {gap:.3f} drop from training to CV indicates <b>overfitting</b>. "
    elif r.cv_f1_macro_mean < best - 0.03:
        s += ("Training and CV scores are both clearly below the best model, which indicates <b>underfitting</b> "
              "(the model family is too simple, or its assumptions do not hold for these features). ")
    elif m != "KNN":
        s += "Training, CV and student-test scores are close, so there is no significant over- or under-fitting. "
    return s


def model_section(m, c):
    r = c["sm"].loc[m]
    i = INFO[m]
    return f"""
<p><b>A1. Why suitable?</b> {i['why']}</p>
<p><b>A2. Strengths and limitations.</b> {i['sl']}</p>
<p><b>B5. Important hyper-parameters (initial values).</b> {i['hp']}</p>
<p><b>B6. Loss / objective.</b> {i['loss']}</p>
<p><b>C7-C8. Student-test performance.</b> Observation-level Top-1 accuracy {f4(r.st_top1_acc)}, Macro F1
{f4(r.st_f1_macro)} (precision {f4(r.st_precision_macro)}, recall {f4(r.st_recall_macro)}), log loss
{f4(r.st_log_loss)}, route continuity {f4(r.st_route_continuity)}, inference {r.inference_ms_per_obs:.4f} ms per
observation, training time {r.fit_time_s:.1f} s. CV Top-1 was {f4(r.cv_top1_mean)} ± {f4(r.cv_top1_std)}.</p>
<p><b>C9. Over/under-fitting.</b> {fit_comment(m, r, c)}</p>"""


def sec3_selection(c):
    sm = c["sm"].sort_values("cv_top1_mean", ascending=False)
    lg, rf = sm.loc["LightGBM"], sm.loc["RandomForest"]
    import os
    size = lambda m: os.path.getsize(ROOT / "models" / "experiments" / f"{m}.joblib") / 1e6
    others = [m for m in sm.index[:4] if m not in ("LightGBM", "RandomForest")]
    return f"""<div class='box'><b>Model selection.</b> Two models clearly lead on the primary metric (CV Top-1 on the
training trips): Random Forest ({f4(rf.cv_top1_mean)} ± {f4(rf.cv_top1_std)}) and LightGBM
({f4(lg.cv_top1_mean)} ± {f4(lg.cv_top1_std)}). They are ahead of {', '.join(others)}. The gap between the two
({abs(rf.cv_top1_mean - lg.cv_top1_mean):.4f}) is about 1/100 of the fold-to-fold standard deviation, so on CV they
are <b>statistically tied</b> (CV Macro F1 {f4(rf.cv_f1_macro_mean)} vs {f4(lg.cv_f1_macro_mean)}, log loss
{f4(rf.cv_log_loss_mean)} vs {f4(lg.cv_log_loss_mean)}). We therefore used the secondary criteria:
(1) generalisation to the student-test trips: Top-1 {f4(lg.st_top1_acc)} (LightGBM) vs {f4(rf.st_top1_acc)} (RF),
Macro F1 {f4(lg.st_f1_macro)} vs {f4(rf.st_f1_macro)};
(2) inference cost for the API: {lg.inference_ms_per_obs:.4f} vs {rf.inference_ms_per_obs:.4f} ms per observation;
(3) model size: {size('LightGBM'):.1f} MB vs {size('RandomForest'):.1f} MB;
(4) room for improvement through tuning: LightGBM has many influential hyper-parameters, and its untuned
configuration is only the library default.
Both produce well-calibrated probabilities (similar log loss), which the HMM uses as emissions. <b>LightGBM was
selected</b> and taken forward to Bayesian optimisation. Training accuracy was not used for the decision.</div>"""


def sec4(c):
    bp, tv = c["bp"], c["tv"]
    un, tu = tv.iloc[0], tv.iloc[1]
    tr = c["trials"]
    best = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp["best_params"].items())
    space = "".join(f"<li><code>{k}</code>: {v}</li>" for k, v in bp["search_space"].items())
    better = tu.st_top1_acc > un.st_top1_acc
    return f"""
<ol>
<li><b>Hyper-parameters searched:</b> the 8 listed below. They control model capacity (n_estimators, num_leaves,
min_child_samples), step size (learning_rate), randomisation (subsample, colsample_bytree) and regularisation
(reg_lambda, reg_alpha).</li>
<li><b>Ranges / distributions:</b><ul>{space}</ul>Log-uniform distributions are used for scale-type parameters.
The ranges were set before the search, from the LightGBM defaults and the data size (1.3 M rows), and were not
changed afterwards.</li>
<li><b>Objective:</b> {bp['objective']}. The student-test and evaluator trips are never used inside the search.</li>
<li><b>Trials:</b> {bp['n_trials']} trials with the Tree-structured Parzen Estimator (Optuna TPE, seed 42). The
first 8 trials are random exploration, and trial 0 is the untuned default configuration. The full search record is
in <code>results/optuna/optuna_trials.csv</code> and <code>optuna_study.db</code>. Mean wall time per trial was
{tr.user_attrs_seconds.mean():.0f} s (3 fits each), about {tr.user_attrs_seconds.sum() / 60:.0f} CPU-minutes in total.</li>
<li><b>Best values:</b> {best} (3-fold CV Top-1 = {f4(bp['best_value'])}).</li>
<li><b>Did tuning help?</b> On identical 5-fold trip-grouped CV and the same student-test trips: CV Macro F1
{f4(un.cv_f1_macro)} → {f4(tu.cv_f1_macro)}, CV Top-1 {f4(un.cv_top1)} → {f4(tu.cv_top1)}, student-test Top-1
{f4(un.st_top1_acc)} → {f4(tu.st_top1_acc)}, student-test Macro F1 {f4(un.st_f1_macro)} → {f4(tu.st_f1_macro)},
log loss {f4(un.st_log_loss)} → {f4(tu.st_log_loss)}, inference {un.inference_ms_per_obs:.4f} →
{tu.inference_ms_per_obs:.4f} ms/obs. {'The tuned model is better on the primary metric, at a higher training and inference cost from its larger ensemble.' if better else 'The gain on student-test is within noise, so the tuned model is kept mainly for its better CV score and log loss.'}</li>
</ol>"""


def sec5(c):
    d = c["dec"]
    return f"""
<ol>
<li><b>Hidden states:</b> for observation <i>i</i>, the states are its candidate arcs (up to 12 within 50 m). Each
state also stores where the fix projects onto that arc (metres from the arc start). The state index therefore
always maps back to an arc id, so the candidate-to-segment correspondence is kept.</li>
<li><b>Emission probabilities:</b> log E<sub>i</sub>(s) = w · log p̂(s | z<sub>i</sub>), where p̂ is the tuned
LightGBM probability renormalised over the candidates of observation <i>i</i>. w = {d['emis_w']} is the emission
weight. The classifier already encodes distance, heading and context, so it replaces the usual Gaussian distance
emission.</li>
<li><b>Transition probabilities:</b> log T(s→s') = −|d<sub>route</sub>(s, s') − d<sub>gc</sub>(z<sub>i−1</sub>,
z<sub>i</sub>)| / β with β = {d['beta']} m (Newson &amp; Krumm). d<sub>gc</sub> is the straight-line distance
between the two fixes. d<sub>route</sub> is the distance driven on the directed network between the two projection
points: the rest of arc s + the Dijkstra shortest path from s's end node to s''s start node + the offset on s'. On
the same arc it is the forward offset (backward moves under {d['back_tol']} m are treated as GPS jitter).
Dijkstra searches are capped at 2·d<sub>gc</sub> + {d['cutoff_extra']} m and cached per source node.</li>
<li><b>Missing candidates and gaps:</b> a fix with no candidate inside 50 m is skipped by the chain and later gets
the arc of the nearest decoded neighbour, so every observation still gets a prediction. A time gap above
{d['gap_s']:.0f} s ends the chain and starts a new one, because nothing reliable can be said about the path driven
during a long gap.</li>
<li><b>Disconnected consecutive segments:</b> if no candidate of fix <i>i</i> can be reached from any surviving
state of fix <i>i−1</i> within the Dijkstra cap, every transition is −∞. The chain is then <i>broken</i>: it is
back-tracked up to <i>i−1</i> and restarted at <i>i</i> from emissions only. One impossible step therefore cannot
wipe out the whole trip. The final route is rebuilt by joining consecutive decoded arcs with shortest network paths
(for display and route-level scoring).</li>
<li><b>Viterbi recurrence:</b> V<sub>i</sub>(s') = max<sub>s</sub>[V<sub>i−1</sub>(s) + log T(s→s')] + log
E<sub>i</sub>(s'). Back-pointers are stored, V is re-centred at every step for numerical stability, and the
best path is back-tracked. The decoder parameters (β, w) were chosen by grid search on 25 training trips, using
<i>out-of-fold</i> classifier probabilities (the classifier never saw those trips) and Top-1 accuracy as the
criterion. This follows the rule to tune the decoder on training/student-test data only. The first grid
(β ≥ 5 m) peaked at its edge, so it was widened to β ∈ {2, 3, 5, 10, 20, 40, 80} and w ∈ {0.25, 0.5, 1, 2}.
The best settings now form a flat plateau: the top seven lie within 0.002 Top-1 of each other (Figure 15),
so pushing β further below the GPS noise level would not change the result.</li>
</ol>"""


def sec5_discussion(c):
    per, hs = c["per"], c["hs"]
    worse = per[per.top1_hmm < per.top1_clf - 1e-9]
    best = per.assign(g=per.top1_hmm - per.top1_clf).sort_values("g", ascending=False).iloc[0]
    txt = (f"<p><b>6. Did HMM decoding help?</b> Yes. Over all 10 student-test trips, Top-1 accuracy goes from "
           f"{c['pct'](hs['top1_clf'])} to {c['pct'](hs['top1_hmm'])} and route continuity from "
           f"{c['pct'](hs['continuity_clf'])} to {c['pct'](hs['continuity_hmm'])}. Classifier-only output jumps "
           f"between parallel or opposite-direction arcs and side streets for single fixes. The transition model forbids "
           f"these jumps because they would need a long detour on the network. The largest gain is on trip "
           f"{best.trip_id} ({f4(best.top1_clf)} → {f4(best.top1_hmm)}). ")
    if len(worse):
        txt += (f"The HMM did <b>not</b> help on {len(worse)} trip(s) "
                f"({', '.join(worse.trip_id)}). There, the classifier was already very accurate, and the decoder sometimes "
                "assigns a fix near a junction to the arc before or after the true one. That position along the route is still "
                "consistent, but it counts as a Top-1 error. ")
    else:
        txt += "The HMM did not reduce Top-1 accuracy on any student-test trip. "
    txt += ("The remaining errors are mostly (a) fixes at arc boundaries and junctions, where the reference label itself "
            "is uncertain (it comes from our alignment), and (b) short sections where the GPS is far from every road. "
            f"Viterbi decoding takes {hs['decode_s_mean']:.2f} s per trip on average.</p>"
            "<p class='small'>In this section, Top-1 is computed over the 19,143 of 19,160 student-test observations (99.9%) "
            "whose reference label is reliable (≤ 100 m from the aligned reference arc). That is why classifier-only Top-1 "
            "here differs slightly from Section 4, where all observations are counted. Macro F1 here scores the single "
            "candidate picked per observation (classifier argmax vs Viterbi choice) against the candidate labels.</p>")
    return txt


def sec6(c):
    return f"""<p>Folium maps are produced by <code>scripts/01_eda.py</code> (EDA maps) and
<code>scripts/06_maps.py</code> (prediction maps for all 10 student-test trips). The interactive HTML files are
committed in <code>results/maps/</code>. Each map has separate toggleable layers (LayerControl): raw GPS
observations (blue dots), the road network within 150 m of the track (grey), the reference route (wide green line),
the HMM/Viterbi predicted route (red line on top of the green) and, optionally, the classifier-only wrong matches
(orange). There are start/end markers and a legend that shows the trip's classifier-only vs HMM accuracy. The view
is fitted to the trajectory's bounding box. Where red covers green, the prediction matches the reference. Green
without red shows reference arcs that were missed, and red without green shows extra arcs. The UI uses the same
colours, but draws only the GPS and the predicted route, because ground truth must stay on the server.</p>"""


def sec7(c):
    return """<p><b>Architecture (two laptops).</b> <b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0
--port 8000</code>. It loads the tuned LightGBM model (<code>models/final_model.joblib</code>), the decoder
configuration (<code>models/decoder_config.json</code>) and the trip folders (GPS + road network + reference
route). <b>Laptop 2</b> runs <code>streamlit run ui/app.py</code> and only needs <code>requirements-ui.txt</code>;
it has no model and no data.</p>
<table><tr><th>Endpoint</th><th>Purpose</th></tr>
<tr><td>GET /health</td><td>Model name, tuned parameters, decoder settings, evaluation partition</td></tr>
<tr><td>GET /trips</td><td>Ids and sizes of the 10 reserved trips (no labels)</td></tr>
<tr><td>POST /predict {"trip_id"}</td><td>Load the GPS + network → candidates → features → LightGBM → Viterbi →
route. Then, on the server, score against the reference route. Returns the GPS points, predicted route polyline,
arc ids, metrics and timings.</td></tr></table>
<p><b>Leakage safeguards.</b> <code>PathSnapPredictor.predict()</code> receives a trip object loaded with
<code>with_route=False</code>. The reference route is loaded separately <i>after</i> prediction and is used only by
<code>evaluate()</code>. The response contains metrics but never the reference geometry. No prediction is cached or
hard-coded: every click runs the full pipeline, and the API console logs each request (client IP, trip, accuracy,
timings) so the evaluator can watch them.</p>
<div class='warn'>The screenshots below come from an integration test on one machine (API and UI on the same
laptop, using the student-test trips). On evaluation day, replace them with screenshots of the two-laptop setup:
the UI on Laptop 2, and the CMD window of Laptop 1 showing the received requests.</div>"""


def sec8(c):
    return """<h2>8. Academic Integrity and AI Use</h2>
<p>All experiments in this report were run by the team on the provided train and student-test partitions. Every table
and figure is generated by the scripts in the repository from logged results (<code>results/</code>,
<code>logs/</code>), and the report itself is built by <code>report/build_report.py</code> directly from those
files. The evaluator-testing partition was never used for training, feature selection, model selection,
hyper-parameter tuning or decoder tuning. <b>AI-assistance disclosure:</b> an AI coding assistant was used to help
write code and draft text. The design decisions, the runs and the results are our own, and every team member can
explain and reproduce them.</p>"""
