"""Train and evaluate ONE model: K-Nearest Neighbours (instance-based)

Model (defined in pathsnap/models.py -> MODEL_SPECS["KNN"]):
    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance"))  # 80k training rows

Protocol (identical for all 7 models, implemented in scripts/03_train_models.py):
  1. 5-fold GroupKFold over the 80 training trips (folds split by trip, seed 42)
  2. per fold: fit, predict candidate probabilities, compute Macro F1 / log loss / Top-1 accuracy
  3. refit on all 80 training trips, evaluate once on the 10 student-test trips
Outputs: results/models/model_comparison.csv + cv_fold_results.csv (row for this model is updated),
         results/models/confusion_KNN.csv, models/experiments/KNN.joblib
Then run scripts/08_model_cards.py to refresh results/models/per_model/.

    python scripts/models/train_knn.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

if __name__ == "__main__":
    importlib.import_module("03_train_models").main(["KNN"])
