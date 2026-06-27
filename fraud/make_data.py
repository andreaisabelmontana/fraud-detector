"""Synthetic transaction generator.

This produces a *clearly synthetic*, heavily imbalanced transactions dataset
with a genuine (but noisy) fraud signal, so that a model trained on it has
something real to learn while remaining honest about its provenance.

Nothing here is real customer data. The generative process is fully specified
below so the signal a model picks up is auditable:

  - ~1.5% of rows are fraud (configurable), giving the classic imbalance.
  - Fraud probability rises with merchant-category risk, transaction amount,
    late-night hour, distance-from-home, and a recent burst of activity on the
    card. Each of these is a real, documented driver of the label.
  - Gaussian noise is injected into both the latent fraud propensity and the
    observed features, so the classes overlap and no model can be perfect.

The output is a tidy DataFrame / CSV with raw columns; feature engineering
lives separately in ``fraud.features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Merchant categories with a base fraud-risk weight. Higher = riskier.
CATEGORIES = {
    "grocery": 0.02,
    "food": 0.03,
    "retail": 0.04,
    "fuel": 0.05,
    "subscription": 0.04,
    "pharmacy": 0.03,
    "services": 0.03,
    "home": 0.06,
    "hardware": 0.10,
    "auto": 0.12,
    "electronics": 0.18,
    "travel": 0.24,
    "luxury": 0.30,
}

_CAT_NAMES = list(CATEGORIES.keys())
_CAT_RISK = np.array([CATEGORIES[c] for c in _CAT_NAMES])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_transactions(
    n: int = 20000,
    fraud_rate: float = 0.015,
    seed: int = 1729,
) -> pd.DataFrame:
    """Generate ``n`` synthetic transactions with an engineered fraud signal.

    Parameters
    ----------
    n : int
        Number of transactions.
    fraud_rate : float
        Approximate share of positive (fraud) rows. The intercept of the
        latent log-odds is calibrated so the realised rate lands near this.
    seed : int
        RNG seed for full reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: amount, hour, category, distance_from_home,
        txns_last_hour, is_foreign, is_fraud.
    """
    rng = np.random.default_rng(seed)

    # ----- raw features -------------------------------------------------
    # Amount: log-normal, heavier tail for travel/luxury (applied after cat).
    cat_idx = rng.integers(0, len(_CAT_NAMES), size=n)
    category = np.array(_CAT_NAMES)[cat_idx]
    cat_risk = _CAT_RISK[cat_idx]

    amount = np.exp(rng.normal(2.7, 1.0, size=n))
    high_ticket = np.isin(category, ["travel", "luxury", "electronics"])
    amount[high_ticket] *= rng.uniform(2.0, 5.0, size=high_ticket.sum())
    amount = np.clip(amount, 1.0, 6000.0)

    # Hour of day: bimodal (lunch + evening) with a late-night tail.
    hour = np.clip(rng.normal(15, 5, size=n).round(), 0, 23).astype(int)
    late_night = (hour <= 4).astype(float)

    # Distance from cardholder's home, in km (most local, some far).
    distance_from_home = np.abs(rng.normal(0, 25, size=n)) + rng.exponential(8, size=n)

    # Velocity: number of transactions on this card in the last hour.
    txns_last_hour = rng.poisson(0.6, size=n)

    # Foreign-country flag (rare, raises risk).
    is_foreign = (rng.random(n) < 0.05).astype(int)

    # ----- latent fraud propensity (log-odds) ---------------------------
    # Each coefficient encodes a documented driver. Standardise the
    # continuous inputs so the coefficients are interpretable.
    z_amount = (np.log(amount) - np.log(amount).mean()) / np.log(amount).std()
    z_dist = (distance_from_home - distance_from_home.mean()) / distance_from_home.std()
    z_cat = (cat_risk - cat_risk.mean()) / cat_risk.std()
    z_vel = (txns_last_hour - txns_last_hour.mean()) / (txns_last_hour.std() + 1e-9)

    logit = (
        1.40 * z_cat
        + 1.20 * z_amount
        + 1.10 * z_dist
        + 0.90 * z_vel
        + 1.60 * late_night
        + 1.90 * is_foreign
        + rng.normal(0, 0.7, size=n)  # irreducible noise -> class overlap
    )

    # Calibrate the intercept so realised positive rate ~ fraud_rate.
    # Find a bias b such that mean(sigmoid(logit + b)) ~= fraud_rate.
    target = fraud_rate
    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    bias = (lo + hi) / 2

    p_fraud = _sigmoid(logit + bias)
    is_fraud = (rng.random(n) < p_fraud).astype(int)

    df = pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour": hour,
            "category": category,
            "distance_from_home": np.round(distance_from_home, 2),
            "txns_last_hour": txns_last_hour,
            "is_foreign": is_foreign,
            "is_fraud": is_fraud,
        }
    )
    return df


if __name__ == "__main__":
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="Generate synthetic transactions CSV.")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--fraud-rate", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1] / "data" / "transactions.csv"),
    )
    args = ap.parse_args()

    df = make_transactions(n=args.n, fraud_rate=args.fraud_rate, seed=args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rate = df["is_fraud"].mean()
    print(f"Wrote {len(df):,} rows to {out}  (fraud rate = {rate:.4f}, "
          f"{int(df['is_fraud'].sum())} positives)")
