"""Charts saved as PNG files (matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def plot_pr_curves(y_true, proba_by_model: dict, path: Path) -> None:
    """Precision-recall curves. The dashed line is what random guessing achieves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in proba_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.axhline(float(pd.Series(y_true).mean()), color="gray", linestyle="--", label="random guessing")
    ax.set_xlabel("Recall (share of churners we catch)")
    ax.set_ylabel("Precision (share of flagged who really churn)")
    ax.set_title("Precision-recall on the held-out test set")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profit_curve(curve: pd.DataFrame, chosen: float, path: Path) -> None:
    """Expected profit at each threshold, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(curve["threshold"], curve["profit_per_customer"], linewidth=2, color="#4C72B0")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(chosen, color="#C44E52", linestyle="--", label=f"chosen threshold = {chosen:.2f}")
    ax.set_xlabel("Risk threshold (contact customers scoring above this)")
    ax.set_ylabel("Expected profit per customer ($)")
    ax.set_title("Choosing the threshold by profit (cross-validated on training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path, top: int = 10) -> None:
    """Permutation importance: how much the score drops when a feature is shuffled."""
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(data["feature"], data["importance"], xerr=data["std"], color="#55A868")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What drives the model's predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
