"""Epoch-wise training / validation loss for the MLP (the only model in the zoo that
trains in epochs). 64 training trips for fitting, 16 held-out training trips for
validation (grouped by trip). Same architecture as models.py "MLP"."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathsnap.config import RESULTS, FEATURES, SEED
import importlib
tm = importlib.import_module("03_train_models")


def main(epochs=25):
    tr, _ = tm.load_train()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    ti, vi = next(gss.split(tr, groups=tr.trip_id))
    dtr = tm.subsample_obs(tr.iloc[ti], 300_000, SEED)
    dva = tr.iloc[vi]
    sc = StandardScaler().fit(dtr[FEATURES])
    Xtr, ytr = sc.transform(dtr[FEATURES]), dtr.label.to_numpy()
    Xva, yva = sc.transform(dva[FEATURES]), dva.label.to_numpy()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, learning_rate_init=1e-3, random_state=SEED)
    rng = np.random.default_rng(SEED)
    hist = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), 512):
            b = order[s:s + 512]
            m.partial_fit(Xtr[b], ytr[b], classes=[0, 1])
        hist.append({"epoch": ep, "train_log_loss": log_loss(ytr, m.predict_proba(Xtr)[:, 1]),
                     "val_log_loss": log_loss(yva, m.predict_proba(Xva)[:, 1])})
        print(hist[-1], flush=True)
    h = pd.DataFrame(hist)
    out = RESULTS / "models"
    h.to_csv(out / "mlp_epoch_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h.epoch, h.train_log_loss, "o-", label="train (64 trips)")
    ax.plot(h.epoch, h.val_log_loss, "s-", label="validation (16 held-out trips)")
    ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.set_title("MLP(64,32) training history"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "mlp_epoch_history.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
