"""Thin wrapper around the trained scikit-learn pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib


class SentimentModel:
    """Loads a saved pipeline and turns raw text into labelled predictions."""

    def __init__(self, pipeline, metadata: dict | None = None):
        self.pipeline = pipeline
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path | str) -> "SentimentModel":
        artifact = joblib.load(path)
        return cls(artifact["pipeline"], artifact.get("metadata", {}))

    def predict(self, texts: list[str]) -> list[dict]:
        probabilities = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        results = []
        for row in probabilities:
            scores = {label: round(float(p), 4) for label, p in zip(classes, row)}
            best = max(scores, key=scores.get)
            results.append(
                {"label": best, "confidence": scores[best], "probabilities": scores}
            )
        return results
