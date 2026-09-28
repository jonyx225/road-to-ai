"""Ask questions from the command line.

    python -m rag.ask "How long is the warranty on tents?"
    python -m rag.ask                                   # interactive: type questions, Ctrl+C to quit
    python -m rag.ask "..." --llm fake                  # offline demo, no API key
    python -m rag.ask "..." --mode bm25 --top-k 6 --show-context
"""
from __future__ import annotations

import argparse
import json
import sys

from rag import config
from rag.generation import LLMConfigError, LLMError, make_generator
from rag.pipeline import Answer, RAGPipeline
from rag.rerank import RerankerUnavailable
from rag.retrieval import MODES
from rag.store import IndexNotFoundError


def format_answer(answer: Answer, show_context: bool = False) -> str:
    lines = ["", answer.answer, ""]
    if answer.warnings:
        lines += [f"  ! {w}" for w in answer.warnings] + [""]
    lines.append("Sources (retrieved for this question; * = cited in the answer):")
    for s in answer.sources:
        ranks = []
        if s["dense_rank"]:
            ranks.append(f"dense #{s['dense_rank']}")
        if s["bm25_rank"]:
            ranks.append(f"keyword #{s['bm25_rank']}")
        lines.append(f"  [{s['n']}]{'*' if s['cited'] else ' '} {s['source']} > {s['section']}  ({', '.join(ranks)})")
        if show_context:
            lines += ["      " + ln for ln in s["text"].splitlines()] + [""]
    cost = f", est. cost ${answer.cost_usd:.5f}" if answer.cost_usd is not None else ""
    lines += [
        "",
        f"[{answer.model} | {answer.mode} | retrieval {answer.retrieval_ms:.0f} ms, "
        f"generation {answer.generation_ms:.0f} ms | tokens in {answer.input_tokens}/out {answer.output_tokens}{cost}]",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--index", default=config.INDEX_DIR)
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=MODES)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--rerank", action="store_true", help="rerank candidates with a cross-encoder")
    parser.add_argument("--llm", default=config.LLM_BACKEND, choices=["anthropic", "fake"])
    parser.add_argument("--show-context", action="store_true", help="print the full text of each source")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args(argv)

    try:
        pipeline = RAGPipeline.load(args.index, generator=make_generator(args.llm))
    except IndexNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    def run(question: str) -> int:
        try:
            answer = pipeline.ask(question, mode=args.mode, top_k=args.top_k, rerank=args.rerank)
        except (LLMConfigError, LLMError, RerankerUnavailable, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(answer.to_dict(), indent=2) if args.json else format_answer(answer, args.show_context))
        return 0

    if args.question:
        return run(args.question)

    print("Ask a question (Ctrl+C to quit).")
    try:
        while True:
            question = input("\n> ").strip()
            if question:
                run(question)
    except (KeyboardInterrupt, EOFError):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
