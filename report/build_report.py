"""Build the technical report (HTML -> PDF with headless Microsoft Edge / Chrome).

Every number in the report is read from the CSV/JSON files written by the
experiment scripts; the narrative text is in report/report_text.py.

    python report/build_report.py
"""
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_text as T  # noqa: E402

R = ROOT / "results"
SHOTS = ROOT / "report" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("chromium") or "", shutil.which("google-chrome") or ""]:
        if p and Path(p).exists():
            return p
    raise SystemExit("Edge/Chrome not found")


def screenshot(url, out, w=1400, h=900, wait_ms=6000, wait_text=None, full_page=False):
    """Screenshot a page with Playwright driving the installed Edge (falls back to Edge CLI)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.goto(url, wait_until="load", timeout=120000)
            if wait_text:
                pg.get_by_text(wait_text).first.wait_for(timeout=180000)
            pg.wait_for_timeout(wait_ms)
            pg.screenshot(path=str(out), full_page=full_page)
            b.close()
        return
    except ImportError:
        pass
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                    f"--virtual-time-budget={wait_ms}", f"--screenshot={out}", url],
                   check=False, capture_output=True, timeout=180)


def img(path, width="100%", caption=None):
    p = Path(path)
    if not p.exists():
        return f"<p class='missing'>[missing figure: {p.name}]</p>"
    b = base64.b64encode(p.read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure><img src='data:image/png;base64,{b}' style='width:{width}'>{cap}</figure>"


def table(df, fmt=4, cls=""):
    return df.to_html(index=False, float_format=lambda v: f"{v:.{fmt}f}", border=0, classes=cls, escape=False)


def pct(v):
    return f"{100 * v:.2f}%"


CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #1b1b1b; line-height: 1.45; }
h1 { font-size: 22pt; margin: 0 0 4px; color: #0b3d62; }
h2 { font-size: 15pt; color: #0b3d62; border-bottom: 2px solid #0b3d62; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #134e7a; margin: 16px 0 6px; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10px 0 4px; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 6px 0 10px; page-break-inside: avoid; }
th { background: #0b3d62; color: white; padding: 4px 5px; text-align: left; }
td { padding: 3px 5px; border-bottom: 1px solid #ddd; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fa; }
figure { margin: 8px 0 12px; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #555; margin-top: 3px; }
.box { background: #eef5fb; border-left: 4px solid #0b3d62; padding: 6px 10px; margin: 8px 0; }
.warn { background: #fff6e5; border-left: 4px solid #e69500; padding: 6px 10px; margin: 8px 0; }
.title { text-align: center; margin-top: 60px; }
.small { font-size: 9pt; color: #444; }
.pb { page-break-before: always; }
ol li, ul li { margin-bottom: 2px; }
code { background: #f0f0f0; padding: 0 3px; font-size: 9pt; }
"""


def main():
    inv = pd.read_csv(R / "eda" / "trip_inventory.csv", dtype={"trip_id": str})
    part = pd.read_csv(R / "eda" / "partition_summary.csv")
    sweep = pd.read_csv(R / "eda" / "radius_sweep.csv")
    dq = (R / "eda" / "data_quality.md").read_text(encoding="utf-8").splitlines()[1:]
    sm = pd.read_csv(R / "models" / "model_comparison.csv")
    fr = pd.read_csv(R / "models" / "cv_fold_results.csv")
    tv = pd.read_csv(R / "optuna" / "tuned_vs_untuned.csv")
    bp = json.load(open(R / "optuna" / "best_params.json"))
    trials = pd.read_csv(R / "optuna" / "optuna_trials.csv")
    grid = pd.read_csv(R / "hmm" / "decoder_grid_summary.csv")
    per = pd.read_csv(R / "hmm" / "student_test_per_trip.csv", dtype={"trip_id": str})
    hs = json.load(open(R / "hmm" / "student_test_summary.json"))
    dec = json.load(open(ROOT / "models" / "decoder_config.json"))

    # ---------- screenshots of real artefacts
    for name in ["eda_train_00000018", f"student_test_{T.MAP_TRIP}"]:
        src = R / "maps" / f"{name}.html"
        if src.exists():
            screenshot(src.resolve().as_uri(), str(SHOTS / f"{name}.png"))

    S = {}
    S["sm"] = sm.set_index("model")
    ctx = dict(sm=S["sm"], fr=fr, tv=tv.set_index("model"), bp=bp, hs=hs, dec=dec, per=per, inv=inv,
               trials=trials, grid=grid, pct=pct)

    html = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]
    html.append(T.title_page(ctx))

    # ---------------- Section 2
    html.append("<h2 class='pb'>2. Dataset Understanding and Exploratory Data Analysis</h2>")
    html.append(T.sec2_intro(ctx))
    p2 = part.rename(columns={"partition": "Partition", "trips": "Trips", "gps_points": "GPS observations",
                              "nodes": "Road nodes", "arcs": "Road arcs", "route_arcs": "Route arcs",
                              "invalid": "Invalid coords", "gaps_gt60s": "Gaps > 60 s"})
    html.append("<h3>2.1 Dataset inventory (partitions: train + student_test)</h3>" + table(p2, 0))
    miss = inv.groupby("partition")[["raw_points", "invalid_dropped", "duplicate_time_dropped", "out_of_order",
                                     "n_gaps_gt5s", "n_gaps_gt60s"]].sum().reset_index()
    miss.columns = ["Partition", "Raw GPS rows", "Invalid/NaN coords", "Duplicate timestamps", "Out-of-order",
                    "Gaps > 5 s", "Gaps > 60 s"]
    html.append("<h3>2.2 Missing-value and validity analysis</h3>" + T.sec2_missing(ctx) + table(miss, 0))
    html.append("<ul>" + "".join(f"<li>{l[2:]}</li>" for l in dq if l.startswith("*")) + "</ul>")
    html.append("<h3>2.3 Preprocessing</h3>" + T.sec2_preprocess(ctx))
    html.append(img(R / "eda" / "eda_overview.png", caption="Figure 1 - EDA overview (train = blue, student_test = orange)."))
    html.append(img(R / "eda" / "gps_noise_and_radius.png", caption="Figure 2 - Train partition: GPS distance to the aligned reference arc and candidate miss rate vs search radius."))
    sw = sweep.copy(); sw.columns = ["Radius (m)", "Miss rate (train)"]
    html.append(table(sw, 4))
    html.append(img(R / "eda" / "trip_00000018_network.png", "70%", "Figure 3 - Train trip 00000018: road network, reference route and raw GPS (static plot)."))
    html.append(img(SHOTS / "eda_train_00000018.png", caption="Figure 4 - Folium map (train trip 00000018): road network within 150 m (grey), reference route (green), raw GPS (blue). Interactive HTML: results/maps/eda_train_00000018.html"))
    html.append("<h3>2.4 Patterns observed and features extracted</h3>" + T.sec2_patterns(ctx))

    # ---------------- Section 3
    html.append("<h2 class='pb'>3. Model Experimentation and Comparison</h2>" + T.sec3_intro(ctx))
    t = sm.copy()
    t["CV Macro F1"] = t.apply(lambda r: f"{r.cv_f1_macro_mean:.4f} ± {r.cv_f1_macro_std:.4f}", axis=1)
    t["Top-1 Acc (ST)"] = t.st_top1_acc.map(lambda v: f"{v:.4f}")
    t["Route continuity (ST)"] = t.st_route_continuity.map(lambda v: f"{v:.4f}")
    t["ST Macro F1"] = t.st_f1_macro.map(lambda v: f"{v:.4f}")
    t["Log loss (ST)"] = t.st_log_loss.map(lambda v: f"{v:.4f}")
    t["Inference (ms/obs)"] = t.inference_ms_per_obs.map(lambda v: f"{v:.4f}")
    t = t[["model", "family", "config", "CV Macro F1", "Top-1 Acc (ST)", "Route continuity (ST)", "ST Macro F1",
           "Log loss (ST)", "Inference (ms/obs)"]].rename(columns={"model": "Model", "family": "Algorithm family",
                                                                    "config": "Key configuration"})
    html.append("<h3>3.1 Overall model comparison</h3>" + table(t) + T.sec3_table_note(ctx))
    pr = sm[["model", "st_accuracy", "st_precision_macro", "st_recall_macro", "st_f1_macro", "st_precision_pos",
             "st_recall_pos", "cv_top1_mean", "cv_top1_std", "fit_time_s"]]
    pr.columns = ["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "Precision (pos)",
                  "Recall (pos)", "CV Top-1", "CV Top-1 sd", "Fit time (s)"]
    html.append("<h4>Additional student-test metrics (candidate-pair level, threshold 0.5)</h4>" + table(pr))
    html.append("<h3>3.2 Per-model analysis</h3>")
    for i, m in enumerate(sm.model, 1):
        html.append(f"<h3>3.2.{i} {m}</h3>" + T.model_section(m, ctx))
    html.append("<h3 class='pb'>3.3 Model visualisations</h3>")
    P = R / "plots"
    html.append(img(P / "cv_macro_f1.png", "85%", "Figure 5 - Cross-validation Macro F1 per fold for every model."))
    html.append(img(P / "student_test_top1_f1.png", "85%", "Figure 6 - Student-test Top-1 accuracy and Macro F1."))
    html.append(img(P / "logloss_inference.png", "95%", "Figure 7 - Student-test log loss and inference time (log scales)."))
    html.append(img(P / "cv_top1_continuity.png", "85%", "Figure 8 - CV Top-1 (mean ± sd over 5 folds) and student-test route continuity."))
    html.append(img(P / "confusion_matrices.png", "100%", "Figure 9 - Confusion matrices on student-test candidate pairs (diagnostic)."))
    html.append(img(R / "models" / "mlp_epoch_history.png", "60%", "Figure 10 - MLP training/validation loss per epoch (the only epoch-trained model)."))
    html.append(T.sec3_selection(ctx))

    # ---------------- Section 4
    html.append("<h2 class='pb'>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>" + T.sec4(ctx))
    tvt = tv[["model", "cv_f1_macro", "cv_top1", "cv_log_loss", "st_top1_acc", "st_f1_macro", "st_log_loss",
              "st_route_continuity", "fit_time_s", "inference_ms_per_obs"]]
    tvt.columns = ["Model", "CV Macro F1", "CV Top-1", "CV log loss", "ST Top-1", "ST Macro F1", "ST log loss",
                   "ST continuity", "Fit time (s)", "Inference (ms/obs)"]
    html.append(table(tvt))
    html.append(img(R / "optuna" / "optuna_history.png", "70%", "Figure 11 - Optuna TPE optimisation history (objective: 3-fold GroupKFold Top-1)."))
    html.append(img(P / "tuned_vs_untuned.png", "70%", "Figure 12 - Untuned vs tuned LightGBM on identical folds and data."))
    html.append(img(P / "lgbm_learning_curve.png", "60%", "Figure 13 - Tuned LightGBM boosting-round history on CV fold 0."))
    top = trials.sort_values("value", ascending=False).head(8)
    cols = [c for c in top.columns if c.startswith("params_")] + ["value"]
    tt = top[["number"] + cols].rename(columns=lambda c: c.replace("params_", ""))
    html.append("<h4>Top 8 of the recorded trials (full record: results/optuna/optuna_trials.csv)</h4>" + table(tt, 4))

    # ---------------- Section 5
    html.append("<h2 class='pb'>5. HMM and Viterbi Route Reconstruction</h2>" + T.sec5(ctx))
    comp = pd.DataFrame([
        ["Top-1 accuracy (obs-weighted)", hs["top1_clf"], hs["top1_hmm"]],
        ["Macro F1 (candidate pairs, decoded choice)", hs["f1_macro_clf"], hs["f1_macro_hmm"]],
        ["Route continuity", hs["continuity_clf"], hs["continuity_hmm"]],
        ["Route precision (length-weighted)", hs["route_precision_clf"], hs["route_precision_hmm"]],
        ["Route recall (length-weighted)", hs["route_recall_clf"], hs["route_recall_hmm"]],
        ["Route F1 (length-weighted)", hs["route_f1_clf"], hs["route_f1_hmm"]]],
        columns=["Metric (10 student-test trips)", "Classifier only", "Classifier + HMM/Viterbi"])
    html.append(table(comp))
    pt = per[["trip_id", "n_obs", "top1_clf", "top1_hmm", "continuity_clf", "continuity_hmm", "route_f1_clf",
              "route_f1_hmm", "breaks", "decode_s"]]
    pt.columns = ["Trip", "Obs", "Top-1 clf", "Top-1 HMM", "Cont. clf", "Cont. HMM", "Route F1 clf", "Route F1 HMM",
                  "Chain breaks", "Viterbi time (s)"]
    html.append("<h4>Per-trip comparison</h4>" + table(pt))
    html.append(img(P / "hmm_vs_classifier.png", "100%", "Figure 14 - Classifier-only vs classifier + HMM per student-test trip."))
    html.append(img(P / "decoder_grid.png", "55%", "Figure 15 - Decoder parameter grid on 25 training trips (out-of-fold probabilities)."))
    html.append(T.sec5_discussion(ctx))

    # ---------------- Section 6
    html.append("<h2 class='pb'>6. Geographical Visualisation Using Folium</h2>" + T.sec6(ctx))
    html.append(img(SHOTS / f"student_test_{T.MAP_TRIP}.png", caption=f"Figure 16 - Folium map of student-test trip {T.MAP_TRIP}: raw GPS (blue), road network (grey), reference route (green), HMM predicted route (red), with layer control and legend. Interactive: results/maps/student_test_{T.MAP_TRIP}.html"))

    # ---------------- Section 7
    html.append("<h2 class='pb'>7. API, UI and Deployment</h2>" + T.sec7(ctx))
    for f, cap in [("ui_screenshot.png", "Figure 17 - Evaluator UI (Streamlit) showing a trip list, metrics and the predicted route returned by the API."),
                   ("api_console.png", "Figure 18 - API console (Laptop 1) logging the incoming /trips and /predict requests.")]:
        html.append(img(SHOTS / f, caption=cap))
    html.append(T.sec8(ctx))
    html.append("</body></html>")

    out_html = ROOT / "report" / "PathSnap_Technical_Report.html"
    out_html.write_text("\n".join(html), encoding="utf-8")
    out_pdf = ROOT / "report" / "PathSnap_Technical_Report.pdf"
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.resolve().as_uri()], check=False, capture_output=True,
                   timeout=300)
    print("written", out_pdf, out_pdf.exists())


if __name__ == "__main__":
    main()
