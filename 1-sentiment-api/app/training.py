"""Train the sentiment classifier and save it to disk.

Usage:
    python -m app.training
    python -m app.training --data data/reviews.csv --output models/sentiment.joblib

The CSV must have two columns: `text` and `label`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import BASE_DIR, MODEL_PATH


def build_pipeline() -> Pipeline:
    """TF-IDF features (unigrams + bigrams) feeding a logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]
    )


def load_data(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    return df


def train(
    data_path: Path | str,
    output_path: Path | str = MODEL_PATH,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train, evaluate on a held-out split, save the model, return metrics."""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(classification_report(y_test, predictions))

    metadata = {
        **metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "classes": list(pipeline.classes_),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output_path)
    print(f"Saved model to {output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment model.")
    parser.add_argument("--data", default=BASE_DIR / "data" / "reviews.csv")
    parser.add_argument("--output", default=MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.test_size, args.seed)


if __name__ == "__main__":
    main()
