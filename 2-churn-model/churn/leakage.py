"""Simple, model-agnostic leakage detection.

Idea: if ONE feature on its own predicts churn almost perfectly, it is very
likely something that is only known after the customer left. Real signals are
usually modest (single-feature ROC-AUC of 0.55 to 0.75).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from churn.config import CV_FOLDS, RANDOM_STATE, TARGET
from churn.features import build_preprocessor
from sklearn.pipeline import Pipeline


def single_feature_auc(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Cross-validated ROC-AUC of a small decision tree trained on each feature alone."""
    y = df[TARGET]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for col in feature_cols:
        X = df[[col]]
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(X)),
                ("tree", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
            ]
        )
        scores[col] = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    return pd.Series(scores).sort_values(ascending=False)


def find_leaky(scores: pd.Series, threshold: float = 0.95) -> list[str]:
    """Features whose solo ROC-AUC is suspiciously high."""
    return scores[scores >= threshold].index.tolist()
