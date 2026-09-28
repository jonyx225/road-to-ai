"""Ties it all together: question -> retrieve -> generate -> check -> Answer."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag import config
from rag.generation import ABSTAIN_TEXT, AnswerGenerator, extract_citations, is_abstention
from rag.retrieval import Hit, Retriever
from rag.store import Index


@dataclass
class Answer:
    question: str
    answer: str
    answered: bool                 # False when the model abstained ("I don't know...")
    citations: list[int]           # source numbers the answer actually cites (validated)
    sources: list[dict]            # every chunk the model was shown, numbered 1..k
    warnings: list[str] = field(default_factory=list)
    mode: str = "hybrid"
    top_k: int = 4
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _source_dict(n: int, hit: Hit, cited: bool) -> dict:
    return {
        "n": n,
        "source": hit.chunk.source,
        "section": hit.chunk.section,
        "text": hit.chunk.text,
        "score": float(hit.score),
        "dense_score": None if hit.dense_score is None else float(hit.dense_score),
        "dense_rank": hit.dense_rank,
        "bm25_score": None if hit.bm25_score is None else float(hit.bm25_score),
        "bm25_rank": hit.bm25_rank,
        "rerank_score": hit.rerank_score,
        "cited": cited,
    }


class RAGPipeline:
    def __init__(self, index: Index, generator: AnswerGenerator, retriever: Retriever | None = None):
        self.index = index
        self.generator = generator
        self.retriever = retriever or Retriever(index)

    @classmethod
    def load(cls, index_dir: Path | str = config.INDEX_DIR, generator: AnswerGenerator | None = None) -> "RAGPipeline":
        return cls(Index.load(index_dir), generator or AnswerGenerator())

    def ask(
        self,
        question: str,
        mode: str = config.DEFAULT_MODE,
        top_k: int = config.TOP_K,
        rerank: bool = False,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, mode=mode, top_k=top_k, rerank=rerank)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        result = self.generator.generate(question, hits)
        abstained = is_abstention(result.text) or not result.text.strip()

        warnings: list[str] = []
        citations: list[int] = []
        if not abstained:
            citations, invalid = extract_citations(result.text, len(hits))
            if invalid:
                warnings.append(f"The answer cites source numbers that do not exist: {invalid}")
            if not citations:
                warnings.append("The answer has no valid citations, so treat it with caution")

        cost = None
        if result.input_tokens or result.output_tokens:
            if result.model.startswith("claude"):
                cost = (
                    result.input_tokens * config.LLM_PRICE_INPUT_PER_MTOK
                    + result.output_tokens * config.LLM_PRICE_OUTPUT_PER_MTOK
                ) / 1_000_000

        return Answer(
            question=question,
            answer=ABSTAIN_TEXT if abstained and not result.text.strip() else result.text,
            answered=not abstained,
            citations=citations,
            sources=[_source_dict(n, h, n in citations) for n, h in enumerate(hits, start=1)],
            warnings=warnings,
            mode=mode,
            top_k=top_k,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
            retrieval_ms=retrieval_ms,
            generation_ms=result.latency_s * 1000,
        )
