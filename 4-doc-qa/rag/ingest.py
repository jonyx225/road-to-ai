"""Build (or rebuild) the search index from a folder of documents.

    python -m rag.ingest                       # uses sbert embeddings (downloads a model once)
    python -m rag.ingest --embedder tfidf      # fully offline

Re-run it whenever your documents change: the index is a snapshot, not a live view.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from rag import config
from rag.chunking import word_count
from rag.embeddings import create_embedder
from rag.loaders import load_documents
from rag.store import Index

WORDS_TO_TOKENS = 1.35  # rough English average; real tokenizers vary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the document index.")
    parser.add_argument("--docs", default=config.DOCS_DIR, help="folder (or single file) to index")
    parser.add_argument("--index", default=config.INDEX_DIR, help="where to save the index")
    parser.add_argument("--embedder", default=config.EMBEDDER, choices=["sbert", "tfidf"])
    parser.add_argument("--model", default=None, help="sentence-transformers model name (sbert only)")
    parser.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=config.CHUNK_OVERLAP_WORDS)
    args = parser.parse_args(argv)

    try:
        documents = load_documents(args.docs)
        embedder = create_embedder(args.embedder, args.model)
        start = time.perf_counter()
        index = Index.build(documents, embedder, args.chunk_words, args.overlap_words)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    index.save(args.index)

    sizes = [word_count(c.text) for c in index.chunks]
    print(f"Indexed {len(documents)} documents into {len(index.chunks)} chunks in {time.perf_counter() - start:.1f}s")
    print(f"Chunk size in words: min {min(sizes)}, median {int(statistics.median(sizes))}, max {max(sizes)}")
    print(f"Embedder: {index.meta['embedder']}")
    for row in index.document_summary():
        print(f"  {row['source']:<32}{row['chunks']:>3} chunks")

    limit = getattr(embedder, "max_tokens", None)
    if limit and max(sizes) * WORDS_TO_TOKENS > limit:
        print(
            f"\nWARNING: your longest chunk (~{int(max(sizes) * WORDS_TO_TOKENS)} tokens) may exceed the embedding "
            f"model's {limit}-token input limit. Text past the limit is silently ignored. "
            f"Lower --chunk-words or choose a model with a longer limit.",
            file=sys.stderr,
        )
    print(f"\nSaved index to {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
