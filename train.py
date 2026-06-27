"""Train, evaluate and persist the fraud classifier.

Pipeline:
  1. Load (or generate) the synthetic transactions dataset.
  2. Engineer features and split train/val/test (stratified on the label).
  3. Fit a class-weighted LogisticRegression and a GradientBoosting ensemble.
  4. Select the deployed model by validation PR-AUC.
  5. Pick a decision threshold on the validation set along the PR curve
     (meet a precision floor, maximise recall).
  6. Report HONEST metrics on the held-out test set: precision, recall, F1,
     ROC-AUC, PR-AUC (accuracy reported but not headlined).
  7. Save the model, results.json, and figures (confusion matrix + PR curve).

Run:  python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from fraud import features as F
from fraud.make_data import make_transactions
from fraud.metrics import precision_recall_f1, ranking_metrics, select_threshold
from fraud.model import FraudModel, build_estimators

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "transactions.csv"
FIGS = ROOT / "figures"
MODEL_PATH = ROOT / "data" / "model.pkl"
RESULTS_PATH = ROOT / "results.json"

SEED = 1729
TARGET_PRECISION = 0.50  # operating-point precision floor


def load_data() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)
    df = make_transactions(n=20000, fraud_rate=0.015, seed=SEED)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def plot_pr_curve(y_true, y_score, chosen_threshold, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#DC2626", lw=2.2, label="PR curve")
    base = float(np.mean(y_true))
    ax.axhline(base, color="#94A3B8", ls="--", lw=1.4,
               label=f"no-skill (= prevalence {base:.3f})")
    # Mark the chosen operating point.
    pr = precision[:-1]
    rc = recall[:-1]
    th = thresholds
    idx = int(np.argmin(np.abs(th - chosen_threshold)))
    ax.scatter([rc[idx]], [pr[idx]], color="#111827", zorder=5, s=60,
               label=f"chosen thr={chosen_threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: dict, path: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Reds")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]:,}",
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "#111827",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1], ["pred legit", "pred fraud"])
    ax.set_yticks([0, 1], ["true legit", "true fraud"])
    ax.set_title("Confusion matrix at chosen threshold (test set)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    prevalence = float(df["is_fraud"].mean())
    print(f"Dataset: {len(df):,} rows, fraud rate {prevalence:.4f} "
          f"({int(df['is_fraud'].sum())} positives)")

    X = F.transform(df)
    y = df["is_fraud"].to_numpy()

    # Stratified train / val / test = 60 / 20 / 20.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    print(f"Split: train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    # Fit candidates, compare on validation PR-AUC.
    estimators = build_estimators(seed=SEED)
    val_scores = {}
    fitted = {}
    for name, est in estimators.items():
        est.fit(X_tr, y_tr)
        fitted[name] = est
        p_val = est.predict_proba(X_val)[:, 1]
        rm = ranking_metrics(y_val, p_val)
        val_scores[name] = rm
        print(f"  {name:20s} val PR-AUC={rm['pr_auc']:.4f}  "
              f"ROC-AUC={rm['roc_auc']:.4f}")

    best_name = max(val_scores, key=lambda k: val_scores[k]["pr_auc"])
    best_est = fitted[best_name]
    print(f"Selected model: {best_name}")

    # Threshold selection on validation set (meet precision floor, max recall).
    p_val_best = best_est.predict_proba(X_val)[:, 1]
    sel = select_threshold(y_val, p_val_best, target_precision=TARGET_PRECISION)
    threshold = sel["threshold"]
    print(f"Chosen threshold={threshold:.4f}  "
          f"(val precision={sel['precision']:.3f}, recall={sel['recall']:.3f}, "
          f"target {sel['target']}, met={sel['met_target']})")

    model = FraudModel(
        estimator=best_est,
        threshold=threshold,
        name=best_name,
        feature_names=list(F.FEATURE_NAMES),
    )
    model.save(MODEL_PATH)

    # ----- HONEST test-set evaluation -----------------------------------
    p_te = model.score_matrix(X_te)
    y_pred = (p_te >= threshold).astype(int)
    hard = precision_recall_f1(y_te, y_pred)
    rank = ranking_metrics(y_te, p_te)

    # Figures.
    plot_pr_curve(y_te, p_te, threshold, FIGS / "pr_curve.png")
    plot_confusion(hard["confusion"], FIGS / "confusion_matrix.png")

    results = {
        "dataset": {
            "n_total": int(len(df)),
            "fraud_rate": prevalence,
            "n_positive": int(df["is_fraud"].sum()),
            "source": "synthetic (fraud.make_data, seed=%d)" % SEED,
        },
        "selected_model": best_name,
        "threshold": threshold,
        "threshold_selection": sel,
        "validation_ranking": val_scores,
        "test": {
            "precision": hard["precision"],
            "recall": hard["recall"],
            "f1": hard["f1"],
            "accuracy": hard["accuracy"],
            "roc_auc": rank["roc_auc"],
            "pr_auc": rank["pr_auc"],
            "confusion": hard["confusion"],
        },
        "figures": ["figures/pr_curve.png", "figures/confusion_matrix.png"],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== HELD-OUT TEST METRICS (honest, imbalanced) ===")
    print(f"  precision : {hard['precision']:.4f}")
    print(f"  recall    : {hard['recall']:.4f}")
    print(f"  F1        : {hard['f1']:.4f}")
    print(f"  ROC-AUC   : {rank['roc_auc']:.4f}")
    print(f"  PR-AUC    : {rank['pr_auc']:.4f}")
    print(f"  accuracy  : {hard['accuracy']:.4f}  (not a headline on imbalance)")
    print(f"  confusion : {hard['confusion']}")
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved results -> {RESULTS_PATH}")
    print(f"Saved figures -> {FIGS}/pr_curve.png, {FIGS}/confusion_matrix.png")


if __name__ == "__main__":
    main()
