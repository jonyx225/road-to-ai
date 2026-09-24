"""Turning a raw table into model-ready features."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import ID_COLUMN, LEAKY_COLUMNS, TARGET


def select_features(df: pd.DataFrame, extra_drop: list[str] | None = None) -> pd.DataFrame:
    """Drop the target, the ID, and any leaky columns, leaving only legitimate features."""
    to_drop = {TARGET, ID_COLUMN, *LEAKY_COLUMNS, *(extra_drop or [])}
    return df.drop(columns=[c for c in to_drop if c in df.columns])


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numbers, impute + one-hot encode categories.

    Everything happens inside the sklearn Pipeline, so imputation values and
    scaling are learned from training folds only (no leakage through preprocessing).
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
