"""Model zoo: every candidate-scoring model the team experimented with.

All models are binary classifiers over (observation, candidate arc) pairs:
label 1 = the candidate is the reference arc for that observation.
`max_train_rows` subsamples training rows (whole observations kept together)
for algorithms whose cost grows badly with data size (KNN, MLP)."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

MODEL_SPECS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        config="StandardScaler + L2 logistic regression, C=1.0, lbfgs, max_iter=500",
        make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
        max_train_rows=None),
    "GaussianNB": dict(
        family="Linear / probabilistic",
        config="QuantileTransformer(normal) + Gaussian Naive Bayes, var_smoothing=1e-9",
        make=lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                   GaussianNB()),
        max_train_rows=None),
    "KNN": dict(
        family="Instance-based",
        config="StandardScaler + distance-weighted KNN, k=31, 80k training rows",
        make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1)),
        max_train_rows=80_000),
    "DecisionTree": dict(
        family="Tree-based",
        config="CART, gini, max_depth=12, min_samples_leaf=50",
        make=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
        max_train_rows=None),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        config="100 trees, max_depth=16, min_samples_leaf=20, max_samples=0.3, sqrt features",
        make=lambda: RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=20, max_samples=0.3,
                                            n_jobs=-1, random_state=SEED),
        max_train_rows=None),
    "LightGBM": dict(
        family="Boosting",
        config="LightGBM defaults: 100 trees, 31 leaves, lr=0.1",
        make=lambda: LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1),
        max_train_rows=None),
    "MLP": dict(
        family="Neural network",
        config="StandardScaler + MLP(64,32) ReLU, Adam lr=1e-3, batch 512, early stopping, 300k rows, max 25 epochs",
        make=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512,
                                                                   learning_rate_init=1e-3, max_iter=25,
                                                                   early_stopping=True, n_iter_no_change=5,
                                                                   random_state=SEED)),
        max_train_rows=300_000),
}
