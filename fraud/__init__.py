"""fraud -- a small, honest fraud-detection ML pipeline.

Modules
-------
make_data : synthetic imbalanced transactions generator
features  : feature engineering (raw rows -> numeric design matrix)
metrics   : imbalanced-aware metrics + PR-curve threshold selection
model     : estimators, the FraudModel wrapper, score() API, persistence
"""

from .model import FraudModel, build_estimators
from .metrics import (
    confusion_counts,
    precision_recall_f1,
    ranking_metrics,
    select_threshold,
)
from . import features, make_data

__all__ = [
    "FraudModel",
    "build_estimators",
    "confusion_counts",
    "precision_recall_f1",
    "ranking_metrics",
    "select_threshold",
    "features",
    "make_data",
]
