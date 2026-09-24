"""Synthetic subscription-churn data, plus loading helpers.

We generate data so the project runs offline and reproducibly. The generator
deliberately includes two real-world traps:
  * class imbalance (roughly 1 in 5 customers churns)
  * a leaky column (`final_invoice_sent`) that only exists after a customer leaves

To use real data instead (for example the Kaggle "Telco Customer Churn" file),
save it as a CSV, then update the column names in `churn/config.py`.

Usage:
    python -m churn.data --n 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import DATA_PATH, ID_COLUMN, RANDOM_STATE, TARGET

TARGET_CHURN_RATE = 0.18


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_churn_data(n: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(2.0, 14.0, n).round(), 1, 72)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], size=n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(
        ["credit-card", "bank-transfer", "electronic-check", "mailed-check"],
        size=n,
        p=[0.30, 0.25, 0.30, 0.15],
    )
    tech_support = np.where(internet == "none", 0, (rng.random(n) < 0.35).astype(int))
    num_products = 1 + rng.poisson(1.2, n)
    age = np.clip(rng.normal(45, 15, n), 18, 90).round()

    base_charge = np.select([internet == "fiber", internet == "dsl"], [65.0, 45.0], default=20.0)
    monthly_charges = np.clip(base_charge + 5 * (num_products - 1) + rng.normal(0, 8, n), 15, 150).round(2)
    tickets_90d = rng.poisson(0.7 + 0.7 * (internet == "fiber"))
    late_payments_12m = rng.poisson(0.25 + 0.6 * (payment == "electronic-check"))
    gb_scale = np.select([internet == "fiber", internet == "dsl"], [250.0, 90.0], default=0.0)
    avg_monthly_gb = np.round(gb_scale * rng.lognormal(0, 0.5, n), 1)
    total_charges = np.round(tenure * monthly_charges * rng.uniform(0.9, 1.1, n), 2)

    # Hidden "true" churn risk. The model has to rediscover this pattern from the data.
    logit = (
        np.select([contract == "month-to-month", contract == "one-year"], [1.0, -0.2], default=-1.1)
        - 0.035 * tenure
        + 0.018 * (monthly_charges - 60)
        + 0.35 * tickets_90d
        + 0.40 * late_payments_12m
        + 0.45 * (internet == "fiber")
        - 0.60 * tech_support
        + 0.40 * (payment == "electronic-check")
        - 0.15 * (num_products - 1)
        + rng.normal(0, 0.9, n)  # things we cannot observe
    )

    # Find the intercept that gives the target churn rate (keeps the data imbalanced on purpose).
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() > TARGET_CHURN_RATE:
            hi = mid
        else:
            lo = mid
    churned = (rng.random(n) < _sigmoid(logit + (lo + hi) / 2)).astype(int)

    # LEAKAGE: a final invoice is only sent when a customer cancels.
    final_invoice_sent = np.where(churned == 1, rng.random(n) < 0.96, rng.random(n) < 0.02).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"C{i:05d}" for i in range(1, n + 1)],
            "tenure_months": tenure.astype(int),
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "tech_support": tech_support,
            "num_products": num_products,
            "age": age.astype(int),
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets_90d": tickets_90d,
            "late_payments_12m": late_payments_12m,
            "avg_monthly_gb": avg_monthly_gb,
            "final_invoice_sent": final_invoice_sent,
            TARGET: churned,
        }
    )

    # Real data is messy: knock out a few values.
    for column, rate in [("avg_monthly_gb", 0.03), ("total_charges", 0.01), ("age", 0.02)]:
        df.loc[rng.random(n) < rate, column] = np.nan
    return df


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected a '{TARGET}' column in {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic churn dataset.")
    parser.add_argument("--n", type=int, default=8000, help="number of customers")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", default=DATA_PATH)
    args = parser.parse_args()

    df = generate_churn_data(args.n, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output} (churn rate {df[TARGET].mean():.1%})")


if __name__ == "__main__":
    main()
