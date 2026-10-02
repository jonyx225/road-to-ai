"""A quick RETRIEVAL check: does the right chunk show up in the top results?

    python -m rag.evaluate
    python -m rag.evaluate --show-misses

For each answerable question in data/eval_questions.jsonl we know a phrase that the
correct chunk must contain (`answer_contains`). We measure, for each retrieval mode:
  Hit@1   the right chunk is ranked first
  Hit@k   the right chunk is somewhere in the top k
  MRR@k   mean of 1/rank of the right chunk (0 when it is missing from the top k)
Questions are tagged "keyword" (they reuse the document's wording) or "paraphrase"
(different wording), because search methods behave differently on each.

This checks retrieval only (no LLM, no cost). Judging the final ANSWERS, including
abstentions and citation accuracy, is the job of a fuller evaluation suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import config
from rag.retrieval import MODES, Retriever
from rag.store import Index, IndexNotFoundError


def load_questions(path: Path | str = config.EVAL_PATH) -> list[dict]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    return questions


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def chunk_contains(chunk_text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(chunk_text)


def evaluate_retrieval(
    index: Index, questions: list[dict], modes: tuple[str, ...] = MODES, k: int = config.TOP_K
) -> dict:
    """Return {mode: {"hit_at_1", "hit_at_k", "mrr", "n", "by_tag": {...}, "misses": [...]}}."""
    answerable = [q for q in questions if q.get("answerable", True) and "answer_contains" in q]
    retriever = Retriever(index)
    results: dict[str, dict] = {}
    for mode in modes:
        ranks: list[int | None] = []
        tags: list[str] = []
        misses: list[str] = []
        for q in answerable:
            hits = retriever.retrieve(q["question"], mode=mode, top_k=k)
            rank = next((i for i, h in enumerate(hits, 1) if chunk_contains(h.chunk.text, q["answer_contains"])), None)
            ranks.append(rank)
            tags.append(q.get("tag", "all"))
            if rank is None:
                misses.append(q["question"])

        def summarize(rs: list[int | None]) -> dict:
            n = len(rs)
            return {
                "n": n,
                "hit_at_1": sum(r == 1 for r in rs) / n if n else 0.0,
                "hit_at_k": sum(r is not None for r in rs) / n if n else 0.0,
                "mrr": sum(1 / r for r in rs if r) / n if n else 0.0,
            }

        results[mode] = {
            **summarize(ranks),
            "by_tag": {t: summarize([r for r, tt in zip(ranks, tags) if tt == t]) for t in sorted(set(tags))},
            "misses": misses,
        }
    return results


def format_results(results: dict, k: int) -> str:
    tags = sorted({t for r in results.values() for t in r["by_tag"]})
    header = f"{'mode':<8}{'Hit@1':>8}{f'Hit@{k}':>8}{f'MRR@{k}':>8}" + "".join(f"{'Hit@' + str(k) + ' ' + t:>22}" for t in tags)
    lines = [header]
    for mode, r in results.items():
        row = f"{mode:<8}{r['hit_at_1']:>8.2f}{r['hit_at_k']:>8.2f}{r['mrr']:>8.2f}"
        row += "".join(f"{r['by_tag'][t]['hit_at_k']:>22.2f}" for t in tags)
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval modes on the eval questions.")
    parser.add_argument("--index", default=config.INDEX_DIR)
    parser.add_argument("--questions", default=config.EVAL_PATH)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args(argv)

    try:
        index = Index.load(args.index)
    except IndexNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    questions = load_questions(args.questions)
    results = evaluate_retrieval(index, questions, k=args.k)
    n = next(iter(results.values()))["n"]
    print(f"Embedder: {index.meta['embedder']['name']} | {n} answerable questions | k = {args.k}\n")
    print(format_results(results, args.k))
    if args.show_misses:
        for mode, r in results.items():
            if r["misses"]:
                print(f"\nMisses for {mode}:")
                print("\n".join(f"  - {q}" for q in r["misses"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
