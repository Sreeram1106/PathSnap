"""Train and evaluate ONE model: Multi-layer Perceptron (neural network); epoch curve in scripts/03b_mlp_history.py

Model (defined in pathsnap/models.py -> MODEL_SPECS["MLP"]):
    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, learning_rate_init=1e-3, max_iter=25, early_stopping=True))  # 300k rows

Protocol (identical for all 7 models, implemented in scripts/03_train_models.py):
  1. 5-fold GroupKFold over the 80 training trips (folds split by trip, seed 42)
  2. per fold: fit, predict candidate probabilities, compute Macro F1 / log loss / Top-1 accuracy
  3. refit on all 80 training trips, evaluate once on the 10 student-test trips
Outputs: results/models/model_comparison.csv + cv_fold_results.csv (row for this model is updated),
         results/models/confusion_MLP.csv, models/experiments/MLP.joblib
Then run scripts/08_model_cards.py to refresh results/models/per_model/.

    python scripts/models/train_mlp.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

if __name__ == "__main__":
    importlib.import_module("03_train_models").main(["MLP"])
