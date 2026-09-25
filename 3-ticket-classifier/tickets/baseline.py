"""The cheap baseline: TF-IDF features + logistic regression.

Always build this first. If a 1-millisecond model is nearly as good as a
fine-tuned transformer or an LLM, you have just saved yourself a lot of money.
"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tickets.common import Prediction
from tickets.config import RANDOM_STATE


class TfidfClassifier:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
                ("clf", LogisticRegression(max_iter=2000, C=10.0, random_state=RANDOM_STATE)),
            ]
        )

    def fit(self, train: pd.DataFrame) -> "TfidfClassifier":
        self.pipeline.fit(train["text"], train["label"])
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Predict one text at a time so latency is comparable with the other methods."""
        out = []
        for text in texts:
            start = time.perf_counter()
            label = str(self.pipeline.predict([text])[0])
            out.append(Prediction(label=label, latency_s=time.perf_counter() - start))
        return out
