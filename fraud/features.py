"""Feature engineering for the fraud model.

Takes raw transaction rows (as produced by ``fraud.make_data``) and turns them
into a numeric design matrix. The transform is deterministic and stateless
except for the one-hot category vocabulary, which is fixed at import time so
training and scoring always agree on column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .make_data import CATEGORIES

# Fixed category vocabulary -> stable one-hot column order.
CATEGORY_VOCAB = list(CATEGORIES.keys())

# Continuous / binary columns kept as-is or lightly transformed.
_BASE_NUMERIC = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "is_late_night",
    "distance_from_home",
    "txns_last_hour",
    "is_foreign",
]

FEATURE_NAMES = _BASE_NUMERIC + [f"cat_{c}" for c in CATEGORY_VOCAB]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw transaction columns.

    Returns a DataFrame with exactly ``FEATURE_NAMES`` columns, in order.
    Accepts either a full dataset (with ``is_fraud``) or a single-row /
    label-free frame; the label is ignored here and handled by callers.
    """
    out = pd.DataFrame(index=df.index)

    # Amount is heavy-tailed -> log compresses it.
    out["log_amount"] = np.log1p(df["amount"].to_numpy(dtype=float))

    # Hour is cyclical: encode on a circle so 23 and 0 are adjacent.
    hour = df["hour"].to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_late_night"] = (hour <= 4).astype(float)

    out["distance_from_home"] = df["distance_from_home"].to_numpy(dtype=float)
    out["txns_last_hour"] = df["txns_last_hour"].to_numpy(dtype=float)
    out["is_foreign"] = df["is_foreign"].to_numpy(dtype=float)

    # One-hot the category against the fixed vocabulary.
    cat = df["category"].astype(str).to_numpy()
    for c in CATEGORY_VOCAB:
        out[f"cat_{c}"] = (cat == c).astype(float)

    return out[FEATURE_NAMES]


def transform_one(transaction: dict) -> pd.DataFrame:
    """Engineer features for a single transaction dict.

    Required keys: amount, hour, category, distance_from_home,
    txns_last_hour, is_foreign.
    """
    df = pd.DataFrame([transaction])
    return transform(df)
