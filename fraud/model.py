"""Fraud classifier: training, persistence and the score() API.

Two estimators are trained and compared:

  * ``LogisticRegression(class_weight="balanced")`` -- a transparent linear
    baseline whose imbalance handling is explicit (rare positives are
    up-weighted in the loss).
  * ``GradientBoostingClassifier`` -- a tree ensemble that captures the
    non-linear interactions in the generative process (e.g. high amount AND
    foreign AND late-night).

Features are standardised inside a Pipeline for the linear model so its
coefficients and probabilities are well-behaved. The better model on
validation PR-AUC is selected as the deployed estimator.

The fitted :class:`FraudModel` exposes ``score(transaction) -> probability`` in
[0, 1] and a frozen decision ``threshold`` chosen along the PR curve.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F


def build_estimators(seed: int = 0) -> dict:
    """Return the candidate estimators keyed by name.

    The logistic model uses ``class_weight="balanced"`` -- this is the
    imbalance handling: it re-weights the loss so the rare fraud class is not
    drowned out by the negative majority.
    """
    logreg = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    C=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    gboost = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        random_state=seed,
    )
    return {"logreg_balanced": logreg, "gradient_boosting": gboost}


@dataclass
class FraudModel:
    """A fitted estimator plus its frozen decision threshold."""

    estimator: object
    threshold: float
    name: str
    feature_names: list

    # ----- scoring ------------------------------------------------------
    def score_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of fraud for each row of an engineered feature frame."""
        return self.estimator.predict_proba(X[self.feature_names])[:, 1]

    def score(self, transaction: dict) -> float:
        """Probability of fraud in [0, 1] for one raw transaction dict.

        Required keys: amount, hour, category, distance_from_home,
        txns_last_hour, is_foreign.
        """
        X = F.transform_one(transaction)
        p = float(self.estimator.predict_proba(X[self.feature_names])[:, 1][0])
        return p

    def predict(self, transaction: dict) -> int:
        """Hard fraud decision (1/0) using the frozen threshold."""
        return int(self.score(transaction) >= self.threshold)

    # ----- persistence --------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "estimator": self.estimator,
                    "threshold": self.threshold,
                    "name": self.name,
                    "feature_names": self.feature_names,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "FraudModel":
        with open(Path(path), "rb") as fh:
            d = pickle.load(fh)
        return cls(
            estimator=d["estimator"],
            threshold=d["threshold"],
            name=d["name"],
            feature_names=d["feature_names"],
        )
