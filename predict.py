"""Score a transaction with the trained fraud model.

Usage:
    # From JSON on the command line
    python predict.py --json '{"amount": 1850, "hour": 3, "category": "luxury",
        "distance_from_home": 120, "txns_last_hour": 4, "is_foreign": 1}'

    # Or a few built-in example transactions
    python predict.py --demo

Prints the fraud probability, the frozen decision threshold, and the hard
decision (flag / clear).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud.model import FraudModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "model.pkl"

DEMO = [
    {
        "label": "low-risk grocery run",
        "tx": {"amount": 23.40, "hour": 18, "category": "grocery",
               "distance_from_home": 3.0, "txns_last_hour": 0, "is_foreign": 0},
    },
    {
        "label": "high-risk: foreign, late-night, big luxury ticket, velocity",
        "tx": {"amount": 2400.0, "hour": 3, "category": "luxury",
               "distance_from_home": 180.0, "txns_last_hour": 5, "is_foreign": 1},
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a transaction for fraud.")
    ap.add_argument("--json", type=str, help="Transaction as a JSON object.")
    ap.add_argument("--demo", action="store_true", help="Score example rows.")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")

    model = FraudModel.load(MODEL_PATH)
    print(f"Model: {model.name}   threshold={model.threshold:.4f}\n")

    rows = []
    if args.json:
        rows.append({"label": "input", "tx": json.loads(args.json)})
    if args.demo or not args.json:
        rows.extend(DEMO)

    for row in rows:
        p = model.score(row["tx"])
        decision = "FLAG (fraud)" if p >= model.threshold else "clear"
        print(f"[{row['label']}]")
        print(f"  fraud probability = {p:.4f}  ->  {decision}")
        print()


if __name__ == "__main__":
    main()
