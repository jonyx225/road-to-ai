"""One evaluation function for every method, so the comparison is fair."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from tickets.common import Prediction
from tickets.config import LLM_PRICE_INPUT_PER_MTOK, LLM_PRICE_OUTPUT_PER_MTOK


def latency_stats(latencies_s: list[float]) -> dict:
    ms = np.asarray(latencies_s) * 1000.0
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
    }


def llm_cost_per_1000(
    preds: list[Prediction],
    price_in_per_mtok: float = LLM_PRICE_INPUT_PER_MTOK,
    price_out_per_mtok: float = LLM_PRICE_OUTPUT_PER_MTOK,
) -> float:
    """Average API cost of 1,000 requests, from the token counts the API reported."""
    per_call = [
        (p.input_tokens * price_in_per_mtok + p.output_tokens * price_out_per_mtok) / 1_000_000 for p in preds
    ]
    return float(np.mean(per_call) * 1000)


def break_even_requests_per_month(monthly_hosting_cost: float, llm_cost_per_request: float) -> float | None:
    """Monthly request volume above which self-hosting is cheaper than paying per LLM call.

    Ignores the (small) marginal cost of each self-hosted request and the one-off
    cost of training, so treat it as a rough guide.
    """
    if llm_cost_per_request <= 0:
        return None
    return monthly_hosting_cost / llm_cost_per_request


def evaluate(test: pd.DataFrame, preds: list[Prediction], labels: list[str]) -> dict:
    """Accuracy, macro-F1, accuracy by wording, per-class scores, confusion matrix, latency."""
    if len(test) != len(preds):
        raise ValueError(f"{len(preds)} predictions for {len(test)} test rows")

    y_true = test["label"].tolist()
    y_pred = [p.label for p in preds]

    by_wording = {}
    for wording, group in test.groupby("wording"):
        idx = group.index
        by_wording[str(wording)] = {
            "accuracy": float(np.mean([y_true[i] == y_pred[i] for i in idx])),
            "n": int(len(idx)),
        }

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "by_wording": by_wording,
        "per_class": {k: report[k] for k in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "unparsed": int(sum(p == "unparsed" for p in y_pred)),
        "latency": latency_stats([p.latency_s for p in preds]),
        "mean_input_tokens": float(np.mean([p.input_tokens for p in preds])),
        "mean_output_tokens": float(np.mean([p.output_tokens for p in preds])),
    }
