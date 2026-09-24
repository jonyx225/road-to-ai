"""Step 1: look at the data before modelling anything.

Checks class imbalance, missing values, and target leakage, then saves a
couple of charts. Run:  python -m churn.explore
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churn.config import (
    CV_FOLDS,
    FIGURES_DIR,
    ID_COLUMN,
    LEAKY_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from churn.data import load_data
from churn.features import build_preprocessor, select_features
from churn.leakage import find_leaky, single_feature_auc


def _cv_auc(X: pd.DataFrame, y: pd.Series) -> float:
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(X)),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def main() -> None:
    df = load_data()
    y = df[TARGET]

    print("=== 1. Shape and imbalance ===")
    print(f"{len(df):,} customers, {df.shape[1]} columns")
    print(f"Churn rate: {y.mean():.1%}  ->  a model that always says 'stays' is {1 - y.mean():.1%} accurate")
    print("So accuracy is a misleading metric here. We will use precision, recall and PR-AUC.\n")

    print("=== 2. Missing values ===")
    missing = df.isna().mean()
    missing = missing[missing > 0]
    print(missing.map("{:.1%}".format).to_string() if len(missing) else "none")
    print(f"Duplicate customer IDs: {df[ID_COLUMN].duplicated().sum()}\n")

    print("=== 3. Churn rate by segment ===")
    for col in ["contract", "internet_service", "payment_method"]:
        if col in df.columns:
            print(df.groupby(col)[TARGET].mean().sort_values(ascending=False).map("{:.1%}".format).to_string(), "\n")

    print("=== 4. Leakage check (solo ROC-AUC per feature) ===")
    candidates = [c for c in df.columns if c not in (TARGET, ID_COLUMN)]
    scores = single_feature_auc(df, candidates)
    print(scores.round(3).to_string())
    flagged = find_leaky(scores)
    print(f"\nFlagged as suspicious (solo AUC >= 0.95): {flagged or 'none'}")
    print(f"Currently excluded in config.LEAKY_COLUMNS: {LEAKY_COLUMNS}")
    unhandled = [c for c in flagged if c not in LEAKY_COLUMNS]
    if unhandled:
        print(f"WARNING: add {unhandled} to LEAKY_COLUMNS in churn/config.py")

    print("\n=== 5. What leakage does to your score ===")
    clean = _cv_auc(select_features(df), y)
    leaky = _cv_auc(select_features(df).assign(**{c: df[c] for c in flagged}), y) if flagged else clean
    print(f"ROC-AUC without leaky columns: {clean:.3f}   <- honest estimate")
    print(f"ROC-AUC with leaky columns:    {leaky:.3f}   <- looks amazing, useless in production")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scores.rename("solo_roc_auc").to_csv(REPORTS_DIR / "leakage_scores.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "contract" in df.columns:
        df.groupby("contract")[TARGET].mean().sort_values().plot.barh(ax=axes[0], color="#4C72B0")
        axes[0].set_title("Churn rate by contract type")
        axes[0].set_xlabel("share of customers who churned")
    if "tenure_months" in df.columns:
        df.assign(status=y.map({0: "stayed", 1: "churned"})).groupby("status")["tenure_months"].plot.hist(
            ax=axes[1], bins=25, alpha=0.6, density=True, legend=True
        )
        axes[1].set_title("Tenure (months): churners leave early")
        axes[1].set_xlabel("tenure (months)")
        axes[1].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda.png", dpi=130)
    print(f"\nSaved {FIGURES_DIR / 'eda.png'} and {REPORTS_DIR / 'leakage_scores.csv'}")


if __name__ == "__main__":
    main()
