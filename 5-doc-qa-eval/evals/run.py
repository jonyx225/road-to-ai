"""Run the evaluation suite: answer every eval question, judge each answer, and
optionally check for a regression against a saved baseline.

    python -m evals.run --llm fake --judge fake                     # free, deterministic
    python -m evals.run --llm fake --judge fake --check-regression  # what CI runs
    python -m evals.run --llm anthropic --judge anthropic           # the real thing (costs money)

Two runs matter, and they are not interchangeable:

  1. The OFFLINE run (`--llm fake --judge fake`) is deterministic. Its output is
     checked into the repo as `evals/baseline.json` and re-run in CI on every push,
     purely to prove the harness itself still works. It is NOT a measure of real
     answer quality, because the fake LLM is not a language model (see rag/fake_llm.py).

  2. The REAL run (`--llm anthropic --judge anthropic`) is the one that tells you
     whether your app actually answers well. Run it by hand after changing the prompt,
     the chunking, the retrieval mode, or the model, and compare the numbers to a
     previous real run you saved with --save-as, e.g.:

         python -m evals.run --llm anthropic --judge anthropic --save-as evals/real_baseline_before.json
         # ...make your change...
         python -m evals.run --llm anthropic --judge anthropic --check-regression --baseline evals/real_baseline_before.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals import config as eval_config
from evals.dataset import load_questions
from evals.judge import format_sources_for_judge, make_judge
from evals.metrics import score_one, summarize
from evals.report import write_report
from rag import config as rag_config
from rag.generation import LLMConfigError, LLMError
from rag.generation import make_generator
from rag.pipeline import RAGPipeline
from rag.rerank import RerankerUnavailable
from rag.store import IndexNotFoundError


def run_eval(pipeline: RAGPipeline, judge, questions: list[dict], mode: str, top_k: int, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    results = []
    for i, record in enumerate(questions, start=1):
        answer = pipeline.ask(record["question"], mode=mode, top_k=top_k)
        judge_result = None
        if judge is not None:
            judge_result = judge.judge(record["question"], format_sources_for_judge(answer), answer.answer)
        results.append(score_one(record, answer, judge_result))
        if i % 10 == 0:
            log(f"  {i}/{len(questions)} questions evaluated")
    summary = summarize(results)
    log(
        f"\npass rate {summary.pass_rate:.1%} | answer/abstain accuracy {summary.abstention_accuracy:.1%} | "
        f"citations valid {summary.citation_validity_rate:.1%}"
        + (f" | judge faithful {summary.judge_faithful_rate:.1%}" if summary.judge_faithful_rate is not None else "")
    )
    return {"results": results, "summary": summary}


def check_regression(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """Return a list of problems; an empty list means no regression beyond `tolerance`."""
    problems = []
    for key in ("pass_rate", "abstention_accuracy", "content_accuracy", "citation_validity_rate",
                "judge_faithful_rate", "judge_relevant_rate"):
        base_value, cur_value = baseline.get(key), current.get(key)
        if base_value is None or cur_value is None:
            continue
        if cur_value < base_value - tolerance:
            problems.append(f"{key} dropped from {base_value:.1%} to {cur_value:.1%} (tolerance {tolerance:.0%})")
    return problems


def _result_to_dict(r) -> dict:
    return {
        "question": r.question,
        "tag": r.tag,
        "passed": r.passed,
        "expected_answerable": r.expected_answerable,
        "answered": r.answer.answered,
        "correct_abstention": r.correct_abstention,
        "correct_content": r.correct_content,
        "citation_valid": r.citation_valid,
        "judge": None if r.judge is None else {"faithful": r.judge.faithful, "relevant": r.judge.relevant, "reason": r.judge.reason},
        "answer": r.answer.answer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG app's answers.")
    parser.add_argument("--index", default=rag_config.INDEX_DIR)
    parser.add_argument("--questions", default=rag_config.EVAL_PATH)
    parser.add_argument("--llm", default=rag_config.LLM_BACKEND, choices=["anthropic", "fake"])
    parser.add_argument("--judge", default=rag_config.LLM_BACKEND, choices=["anthropic", "fake", "none"])
    parser.add_argument("--mode", default=rag_config.DEFAULT_MODE, choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=rag_config.TOP_K)
    parser.add_argument("--reports-dir", default=eval_config.REPORTS_DIR)
    parser.add_argument("--baseline", default=eval_config.BASELINE_PATH)
    parser.add_argument("--check-regression", action="store_true")
    parser.add_argument("--tolerance", type=float, default=eval_config.DEFAULT_TOLERANCE)
    parser.add_argument("--save-as", default=None, help="also write the summary to this path")
    args = parser.parse_args(argv)

    try:
        pipeline = RAGPipeline.load(args.index, generator=make_generator(args.llm))
    except IndexNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        judge = None if args.judge == "none" else make_judge(args.judge)
        questions = load_questions(args.questions)
        outcome = run_eval(pipeline, judge, questions, mode=args.mode, top_k=args.top_k)
    except (LLMConfigError, LLMError, RerankerUnavailable, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_dict = outcome["summary"].to_dict()
    (reports_dir / "eval_results.json").write_text(
        json.dumps({"summary": summary_dict, "results": [_result_to_dict(r) for r in outcome["results"]]}, indent=2)
    )
    write_report(outcome["results"], outcome["summary"], reports_dir / "eval_report.md")
    print(f"\nWrote {reports_dir / 'eval_report.md'} and {reports_dir / 'eval_results.json'}")

    if args.save_as:
        Path(args.save_as).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_as).write_text(json.dumps(summary_dict, indent=2))
        print(f"Saved summary to {args.save_as}")

    if args.check_regression:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"Error: no baseline at {baseline_path}. Create one first with --save-as.", file=sys.stderr)
            return 1
        problems = check_regression(summary_dict, json.loads(baseline_path.read_text()), args.tolerance)
        if problems:
            print("\nREGRESSION DETECTED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"\nNo regression vs. baseline ({baseline_path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
