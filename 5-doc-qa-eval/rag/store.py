"""Step 4: the index. Chunks + their vectors + a keyword index, saved to a folder.

Vector search is a matrix multiplication. With unit-length vectors, the dot product
of a query with every chunk vector IS the cosine similarity, and the best matches are
the largest numbers. For thousands of chunks plain numpy is plenty. A vector database
(Chroma, pgvector, ...) adds persistence, filtering and approximate search that pays
off at millions of chunks.

Folder layout:
    meta.json         embedder, chunk settings, document hashes, build time
    chunks.jsonl      one chunk per line
    embeddings.npy    float32 matrix, one row per chunk
    embedder.joblib   only for the tfidf embedder (its fitted vocabulary)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.bm25 import BM25
from rag.chunking import Chunk, chunk_document
from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from rag.embeddings import Embedder, load_embedder
from rag.loaders import Document

FORMAT_VERSION = 1


class IndexNotFoundError(FileNotFoundError):
    pass


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first. Ties keep the original order."""
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder, meta: dict):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embedding rows")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.meta = meta
        self.bm25 = BM25([c.contextual_text for c in chunks])  # cheap to rebuild, so not saved

    # ---------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        documents: list[Document],
        embedder: Embedder,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS,
    ) -> "Index":
        chunks: list[Chunk] = []
        doc_info: dict[str, dict] = {}
        for doc in documents:
            doc_chunks = chunk_document(doc, chunk_words, overlap_words)
            chunks += doc_chunks
            doc_info[doc.source] = {"sha256": doc.sha256, "chunks": len(doc_chunks)}
        if not chunks:
            raise ValueError("The documents produced no chunks (are they empty?)")

        texts = [c.contextual_text for c in chunks]
        embedder.fit(texts)
        embeddings = embedder.encode_documents(texts)
        meta = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedder": embedder.describe(),
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "n_chunks": len(chunks),
            "documents": doc_info,
        }
        return cls(chunks, embeddings, embedder, meta)

    # ---------------------------------------------------------------- search
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode_query(query)
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query vector has {q.shape[0]} dimensions but the index has {self.embeddings.shape[1]}. "
                "The index was built with a different embedder; rebuild it with `python -m rag.ingest`."
            )
        scores = self.embeddings @ q
        return [(int(i), float(scores[i])) for i in top_k_indices(scores, k)]

    # ---------------------------------------------------------------- persistence
    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings)
        self.embedder.save(directory)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "Index":
        """Only load index folders you created yourself: the tfidf embedder is stored with joblib (pickle)."""
        directory = Path(directory)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise IndexNotFoundError(f"No index found in {directory}. Build one with: python -m rag.ingest")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported index format {meta.get('format_version')}; rebuild with `python -m rag.ingest`")
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embeddings = np.load(directory / "embeddings.npy")
        embedder = load_embedder(directory, meta["embedder"])
        return cls(chunks, embeddings, embedder, meta)

    # ---------------------------------------------------------------- info
    def document_summary(self) -> list[dict]:
        return [{"source": s, "chunks": info["chunks"]} for s, info in self.meta["documents"].items()]
