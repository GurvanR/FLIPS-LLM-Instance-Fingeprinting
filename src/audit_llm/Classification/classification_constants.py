"""Shared constants for the Classification package.

Classifier templates, cross-validation splitters, and metric definitions
used by both SingleTokenPairClassification and MultiTokenPairClassification.
"""

import xgboost as xgb
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSIFIERS_TEMPLATES_MAP = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), random_state=42),
    "MLP_heavy": MLPClassifier(
        max_iter=5000,
        hidden_layer_sizes=(100, 50),
        random_state=42,
        tol=1e-6,
        n_iter_no_change=50,
        learning_rate_init=0.001,
        early_stopping=False,
    ),
    "MLP_strong": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=3000,
        early_stopping=False,
        n_iter_no_change=40,
        tol=1e-5,
        alpha=1e-3,
        learning_rate_init=0.001,
        random_state=42,
    ),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

SPLITTER_MAP = {
    "KFold": KFold,
    "StratifiedKFold": StratifiedKFold,
    "ShuffleSplit": ShuffleSplit,
    "StratifiedShuffleSplit": StratifiedShuffleSplit,
    "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
}

CLASSIFIER_METRICS = ("accuracy", "precision", "recall", "f1", "balanced_accuracy")

CLASSIFIER_METRICS_FUN_MAP = {
    "accuracy": lambda y_val, y_pred: accuracy_score(y_val, y_pred),
    "precision": lambda y_val, y_pred: precision_score(y_val, y_pred, average="weighted", zero_division=0),
    "recall": lambda y_val, y_pred: recall_score(y_val, y_pred, average="weighted", zero_division=0),
    "f1": lambda y_val, y_pred: f1_score(y_val, y_pred, average="weighted", zero_division=0),
    "balanced_accuracy": lambda y_val, y_pred: balanced_accuracy_score(y_val, y_pred),
}
