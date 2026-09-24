"""Metrics that matter for imbalanced problems, and cost-based threshold tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from churn.config import BusinessCosts


def summarize(y_true, proba, threshold: float) -> dict:
    """Threshold-dependent metrics plus threshold-free ranking metrics."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "share_contacted": float((tp + fp) / len(y_true)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def profit(tp: int, fp: int, costs: BusinessCosts) -> float:
    """Net profit of a retention campaign that contacts `tp + fp` customers."""
    return tp * costs.gain_per_true_positive - fp * costs.loss_per_false_positive


def profit_curve(y_true, proba, costs: BusinessCosts, thresholds=None) -> pd.DataFrame:
    """Profit at every threshold, so we can pick the best one."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    thresholds = np.linspace(0.02, 0.98, 97) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = proba >= t
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "profit": profit(tp, fp, costs),
                "profit_per_customer": profit(tp, fp, costs) / len(y_true),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, costs: BusinessCosts) -> float:
    """Threshold that maximizes profit."""
    curve = profit_curve(y_true, proba, costs)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def baseline_profits(y_true, costs: BusinessCosts) -> dict:
    """Profit of the two 'no model' strategies, for comparison."""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    return {
        "contact_nobody": 0.0,
        "contact_everyone": profit(n_pos, n_neg, costs),
    }
