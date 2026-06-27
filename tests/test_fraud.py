"""Tests for the fraud-detection pipeline.

Covered:
  * score() returns probabilities in [0, 1]
  * confusion / precision / recall / F1 computed correctly on a hand-built case
  * the trained model beats a trivial baseline on PR-AUC and on
    recall-at-fixed-precision by a clear margin
  * threshold selection picks an operating point that meets the precision
    target on validation data
  * imbalance handling (class_weight) actually changes recall vs unweighted
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud import features as F
from fraud.make_data import make_transactions
from fraud.metrics import (
    confusion_counts,
    precision_recall_f1,
    ranking_metrics,
    select_threshold,
)
from fraud.model import FraudModel, build_estimators

SEED = 7


# --------------------------------------------------------------------------- #
# Shared fixture: one synthetic dataset, engineered + split once.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def split():
    df = make_transactions(n=24000, fraud_rate=0.015, seed=SEED)
    X = F.transform(df)
    y = df["is_fraud"].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    # Ensure the imbalance is real.
    assert 0.005 < y.mean() < 0.05
    return dict(
        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, X_te=X_te, y_te=y_te
    )


@pytest.fixture(scope="module")
def trained_model(split):
    # Mirror train.py: fit both candidates, deploy the one with the best
    # validation PR-AUC, then freeze a threshold along the PR curve.
    candidates = build_estimators(seed=SEED)
    fitted, val_pr = {}, {}
    for name, est in candidates.items():
        est.fit(split["X_tr"], split["y_tr"])
        fitted[name] = est
        val_pr[name] = ranking_metrics(
            split["y_val"], est.predict_proba(split["X_val"])[:, 1]
        )["pr_auc"]
    best = max(val_pr, key=val_pr.get)
    est = fitted[best]
    p_val = est.predict_proba(split["X_val"])[:, 1]
    sel = select_threshold(split["y_val"], p_val, target_precision=0.5)
    return FraudModel(
        estimator=est,
        threshold=sel["threshold"],
        name=best,
        feature_names=list(F.FEATURE_NAMES),
    )


# --------------------------------------------------------------------------- #
# 1. score() returns probabilities in [0, 1]
# --------------------------------------------------------------------------- #
def test_score_returns_probability_in_unit_interval(trained_model):
    txs = [
        {"amount": 12.0, "hour": 14, "category": "grocery",
         "distance_from_home": 2.0, "txns_last_hour": 0, "is_foreign": 0},
        {"amount": 3200.0, "hour": 2, "category": "luxury",
         "distance_from_home": 220.0, "txns_last_hour": 6, "is_foreign": 1},
        {"amount": 95.0, "hour": 21, "category": "travel",
         "distance_from_home": 40.0, "txns_last_hour": 1, "is_foreign": 0},
    ]
    for tx in txs:
        p = trained_model.score(tx)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    # The clearly high-risk transaction should score above the clearly low one.
    assert trained_model.score(txs[1]) > trained_model.score(txs[0])


# --------------------------------------------------------------------------- #
# 2. Metrics correct on a hand-built confusion case
# --------------------------------------------------------------------------- #
def test_metrics_on_handbuilt_case():
    # 10 examples. By construction:
    #   TP=3, FP=1, FN=2, TN=4
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

    c = confusion_counts(y_true, y_pred)
    assert (c.tp, c.fp, c.fn, c.tn) == (3, 1, 2, 4)

    m = precision_recall_f1(y_true, y_pred)
    # precision = 3 / (3+1) = 0.75
    assert m["precision"] == pytest.approx(0.75)
    # recall = 3 / (3+2) = 0.6
    assert m["recall"] == pytest.approx(0.6)
    # F1 = 2*0.75*0.6 / (0.75+0.6) = 0.9 / 1.35 = 0.6667
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    # accuracy = (3+4)/10 = 0.7
    assert m["accuracy"] == pytest.approx(0.7)


def test_perfect_and_empty_prediction_edge_cases():
    y = [1, 0, 1, 0]
    perfect = precision_recall_f1(y, y)
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    # Predict all-negative: precision/recall/F1 are 0 (no division blow-up).
    none_flagged = precision_recall_f1(y, [0, 0, 0, 0])
    assert none_flagged["precision"] == 0.0
    assert none_flagged["recall"] == 0.0
    assert none_flagged["f1"] == 0.0


# --------------------------------------------------------------------------- #
# 3. Trained model beats a trivial baseline by a clear margin
# --------------------------------------------------------------------------- #
def test_model_beats_baseline_on_pr_auc(split, trained_model):
    p_te = trained_model.score_matrix(split["X_te"])
    rank = ranking_metrics(split["y_te"], p_te)
    prevalence = float(np.mean(split["y_te"]))

    # A no-skill classifier has PR-AUC ~= prevalence and ROC-AUC = 0.5.
    # Require a clear margin over both (prevalence here is ~1.5%).
    assert rank["pr_auc"] > 10 * prevalence
    assert rank["pr_auc"] > 0.30
    assert rank["roc_auc"] > 0.85


def test_model_beats_baseline_on_recall_at_fixed_precision(split, trained_model):
    # The real model should actually hit a 50% precision floor while still
    # recovering a substantial share of fraud at that precision.
    p_te = trained_model.score_matrix(split["X_te"])
    sel = select_threshold(split["y_te"], p_te, target_precision=0.5)
    assert sel["met_target"]
    assert sel["precision"] >= 0.5 - 1e-9
    assert sel["recall"] > 0.30  # clear margin over a no-skill baseline

    # A random-score baseline cannot reach 50% precision at all on imbalanced
    # data: there is no threshold whose precision clears the floor, so
    # select_threshold reports met_target=False (it falls back to F1-optimal,
    # which just floods alerts at ~prevalence precision).
    rng = np.random.default_rng(0)
    p_rand = rng.random(len(split["y_te"]))
    base = select_threshold(split["y_te"], p_rand, target_precision=0.5)
    assert base["met_target"] is False


# --------------------------------------------------------------------------- #
# 4. Threshold selection meets the precision target on validation
# --------------------------------------------------------------------------- #
def test_threshold_selection_meets_precision_target(split):
    est = build_estimators(seed=SEED)["logreg_balanced"]
    est.fit(split["X_tr"], split["y_tr"])
    p_val = est.predict_proba(split["X_val"])[:, 1]

    sel = select_threshold(split["y_val"], p_val, target_precision=0.6)
    assert sel["met_target"]
    assert sel["precision"] >= 0.6 - 1e-9
    assert 0.0 < sel["threshold"] < 1.0
    # Among all feasible thresholds it should pick high recall; sanity-check
    # that a stricter threshold does not yield more recall.
    stricter = (p_val >= min(sel["threshold"] + 0.2, 0.999)).astype(int)
    from fraud.metrics import precision_recall_f1 as prf
    assert prf(split["y_val"], stricter)["recall"] <= sel["recall"] + 1e-9


def test_threshold_recall_mode(split):
    est = build_estimators(seed=SEED)["logreg_balanced"]
    est.fit(split["X_tr"], split["y_tr"])
    p_val = est.predict_proba(split["X_val"])[:, 1]
    sel = select_threshold(split["y_val"], p_val, target_recall=0.7)
    assert sel["met_target"]
    assert sel["recall"] >= 0.7 - 1e-9


# --------------------------------------------------------------------------- #
# 5. Imbalance handling actually changes recall vs unweighted
# --------------------------------------------------------------------------- #
def test_class_weight_increases_recall(split):
    # Same linear model, same threshold (0.5), only class_weight differs.
    def fit_and_recall(class_weight):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight=class_weight, max_iter=2000, random_state=SEED)),
        ])
        pipe.fit(split["X_tr"], split["y_tr"])
        p = pipe.predict_proba(split["X_te"])[:, 1]
        pred = (p >= 0.5).astype(int)
        return precision_recall_f1(split["y_te"], pred)["recall"]

    recall_unweighted = fit_and_recall(None)
    recall_balanced = fit_and_recall("balanced")

    # Up-weighting the rare class must materially raise recall at a fixed
    # threshold (the unweighted model under-flags fraud on heavy imbalance).
    assert recall_balanced > recall_unweighted + 0.15


# --------------------------------------------------------------------------- #
# 6. Save / load round-trip preserves scoring
# --------------------------------------------------------------------------- #
def test_model_save_load_roundtrip(tmp_path, trained_model):
    p = tmp_path / "m.pkl"
    trained_model.save(p)
    loaded = FraudModel.load(p)
    tx = {"amount": 500.0, "hour": 3, "category": "electronics",
          "distance_from_home": 90.0, "txns_last_hour": 2, "is_foreign": 1}
    assert loaded.score(tx) == pytest.approx(trained_model.score(tx))
    assert loaded.threshold == trained_model.threshold
