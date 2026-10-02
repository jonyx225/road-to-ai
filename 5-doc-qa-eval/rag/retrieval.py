"""Step 5: find the chunks most relevant to a question.

Three modes:
  dense   vector similarity (meaning)
  bm25    keyword matching (exact words, identifiers, numbers)
  hybrid  both, merged with Reciprocal Rank Fusion (RRF)

RRF in one line: each retriever ranks chunks; a chunk earns 1 / (60 + rank) from every
list it appears in; add the points up. Ranks are used instead of raw scores because
cosine similarities and BM25 scores live on completely different scales.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag.chunking import Chunk
from rag.config import CANDIDATE_K, RRF_K, TOP_K
from rag.store import Index

MODES = ("dense", "bm25", "hybrid")


@dataclass
class Hit:
    idx: int                       # position of the chunk in the index
    chunk: Chunk
    score: float                   # the score used for the final ordering
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return dict(fused)


class Retriever:
    def __init__(self, index: Index, reranker=None):
        self.index = index
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
        rerank: bool = False,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        pool = max(candidate_k, top_k)
        dense = self.index.dense_search(query, pool) if mode in ("dense", "hybrid") else []
        lexical = self.index.bm25.search(query, pool) if mode in ("bm25", "hybrid") else []

        hits: dict[int, Hit] = {}

        def hit_for(idx: int) -> Hit:
            return hits.setdefault(idx, Hit(idx=idx, chunk=self.index.chunks[idx], score=0.0))

        for rank, (idx, score) in enumerate(dense, start=1):
            h = hit_for(idx)
            h.dense_score, h.dense_rank = score, rank
        for rank, (idx, score) in enumerate(lexical, start=1):
            h = hit_for(idx)
            h.bm25_score, h.bm25_rank = score, rank

        if mode == "hybrid":
            fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in lexical]])
            for idx, h in hits.items():
                h.score = fused[idx]
        elif mode == "dense":
            for h in hits.values():
                h.score = h.dense_score
        else:
            for h in hits.values():
                h.score = h.bm25_score

        ordered = sorted(hits.values(), key=lambda h: (-h.score, h.idx))

        if rerank and ordered:
            if self.reranker is None:
                from rag.rerank import CrossEncoderReranker

                self.reranker = CrossEncoderReranker()
            candidates = ordered[:pool]
            scores = self.reranker.score(query, [h.chunk.contextual_text for h in candidates])
            for h, s in zip(candidates, scores):
                h.rerank_score, h.score = s, s
            ordered = sorted(candidates, key=lambda h: (-h.score, h.idx))

        return ordered[:top_k]
