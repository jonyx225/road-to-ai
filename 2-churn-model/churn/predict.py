"""Use the saved model: score customers and list who to contact.

Run:  python -m churn.predict --input data/churn.csv --top 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn.config import ID_COLUMN, MODELS_DIR, REPORTS_DIR


def score_customers(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Return customers with their churn risk and whether they should be contacted."""
    missing = [c for c in artifact["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")

    risk = artifact["pipeline"].predict_proba(df[artifact["features"]])[:, 1]
    out = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].values if ID_COLUMN in df.columns else range(len(df)),
            "churn_risk": risk.round(4),
            "contact": risk >= artifact["threshold"],
        }
    )
    return out.sort_values("churn_risk", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", required=True, help="CSV with the same feature columns as the training data")
    parser.add_argument("--model", default=MODELS_DIR / "churn_model.joblib")
    parser.add_argument("--output", default=REPORTS_DIR / "at_risk_customers.csv")
    parser.add_argument("--top", type=int, default=10, help="how many rows to print")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    scored = score_customers(pd.read_csv(args.input), artifact)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    print(f"Model: {artifact['model_name']} | threshold {artifact['threshold']:.2f}")
    print(f"{int(scored['contact'].sum()):,} of {len(scored):,} customers flagged for a retention offer")
    print(scored.head(args.top).to_string(index=False))
    print(f"\nSaved full list to {args.output}")


if __name__ == "__main__":
    main()
