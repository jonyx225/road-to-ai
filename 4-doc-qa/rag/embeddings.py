"""Step 3: turn text into vectors.

An embedder maps text to a fixed-length vector so that texts with similar meaning
end up close together. We support two:

  sbert  A pretrained sentence-transformers model. Understands paraphrases
         ("my lamp gets hot" ~ "battery temperature is too high").
         Needs `pip install sentence-transformers` and a one-time model download.

  tfidf  Character n-gram TF-IDF fitted on your own chunks. No downloads, so it powers
         the offline demo and the tests. It is fuzzy *text* matching (it survives typos
         and word endings) but it does NOT understand meaning.

IMPORTANT: documents and queries must be embedded by the same embedder. The index
records which one it used and refuses to be queried with another.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.config import SBERT_MODEL


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...
    def save(self, directory: Path) -> None: ...
    def describe(self) -> dict: ...


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=8000,
        )
        self.vectorizer.fit(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fitted (or loaded) before use")
        # TfidfVectorizer already L2-normalises rows, so a dot product is cosine similarity.
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def save(self, directory: Path) -> None:
        joblib.dump(self.vectorizer, Path(directory) / "embedder.joblib")

    @classmethod
    def load(cls, directory: Path, description: dict | None = None) -> "TfidfEmbedder":
        return cls(joblib.load(Path(directory) / "embedder.joblib"))

    def describe(self) -> dict:
        return {"name": self.name, "dim": len(self.vectorizer.vocabulary_) if self.vectorizer else 0}


class SentenceTransformerEmbedder:
    name = "sbert"

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "The 'sbert' embedder needs sentence-transformers: pip install sentence-transformers "
                    "(or run with --embedder tfidf for the offline version)"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def max_tokens(self) -> int | None:
        """Longest input the model reads. Anything beyond it is silently cut off."""
        return getattr(self.model, "max_seq_length", None)

    def fit(self, texts: list[str]) -> None:
        self.model  # pretrained: nothing to learn, just make sure the weights are loaded

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32)

    def save(self, directory: Path) -> None:
        pass  # weights live in the model hub cache; the index only records the model name

    @classmethod
    def load(cls, directory: Path, description: dict | None = None) -> "SentenceTransformerEmbedder":
        return cls((description or {}).get("model", SBERT_MODEL))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "dim": int(self.model.get_sentence_embedding_dimension()),
        }


def create_embedder(name: str, model: str | None = None) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sbert":
        return SentenceTransformerEmbedder(model or SBERT_MODEL)
    raise ValueError(f"Unknown embedder '{name}'. Choose 'sbert' or 'tfidf'.")


def load_embedder(directory: Path, description: dict) -> Embedder:
    name = description.get("name")
    if name == "tfidf":
        return TfidfEmbedder.load(directory, description)
    if name == "sbert":
        return SentenceTransformerEmbedder.load(directory, description)
    raise ValueError(f"Index was built with an unknown embedder: {name!r}")
