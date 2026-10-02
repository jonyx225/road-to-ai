"""Optional step: rerank the retrieved candidates with a cross-encoder.

The retrievers score the query and each chunk separately (fast, approximate). A
cross-encoder reads the query and one chunk TOGETHER (slower, more accurate), so we
use it only on the ~20 candidates that survived the first stage.
"""
from __future__ import annotations

from rag.config import RERANK_MODEL


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerUnavailable(
                    "Reranking needs sentence-transformers: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]
