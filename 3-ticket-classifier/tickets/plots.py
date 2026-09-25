"""Charts for the comparison (matplotlib, saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_by_wording(results: dict, path: Path) -> None:
    """Grouped bars: accuracy of each method on each kind of wording."""
    methods = list(results)
    wordings = sorted({w for r in results.values() for w in r["by_wording"]},
                      key=lambda w: ["direct", "indirect", "novel"].index(w) if w in ("direct", "indirect", "novel") else 9)
    x = np.arange(len(wordings))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(methods):
        accs = [results[method]["by_wording"].get(w, {"accuracy": np.nan})["accuracy"] for w in wordings]
        bars = ax.bar(x + i * width - 0.4 + width / 2, accs, width, label=method)
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(wordings)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by ticket wording (test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
