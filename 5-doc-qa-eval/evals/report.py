"""Writes reports/eval_report.md: a plain-language summary of an evaluation run."""
from __future__ import annotations

from pathlib import Path

from evals.metrics import QuestionResult, Summary


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def write_report(results: list[QuestionResult], summary: Summary, path: Path) -> None:
    tag_rows = "\n".join(
        f"| {tag} | {info['n']} | {_pct(info['pass_rate'])} | {_pct(info['abstention_accuracy'])} | {_pct(info['content_accuracy'])} |"
        for tag, info in summary.by_tag.items()
    )
    failures = [r for r in results if not r.passed]
    failure_blocks = []
    for r in failures:
        block = [
            f"- **{r.question}** (tag: {r.tag})",
            f"  - expected answerable: {r.expected_answerable}, got answered: {r.answer.answered}",
            f"  - answer: {r.answer.answer[:220]!r}",
        ]
        if r.judge is not None:
            block.append(f"  - judge: faithful={r.judge.faithful}, relevant={r.judge.relevant} \u2014 {r.judge.reason}")
        failure_blocks.append("\n".join(block))
    failure_text = "\n".join(failure_blocks) or "None. Every question passed."

    text = f"""# RAG Evaluation Report

{summary.n} questions evaluated.

| Metric | Value |
|---|---|
| Overall pass rate | {_pct(summary.pass_rate)} |
| Correct answer/abstain decision | {_pct(summary.abstention_accuracy)} |
| Answer contains the expected fact | {_pct(summary.content_accuracy)} |
| Citations are all valid (no invented source numbers) | {_pct(summary.citation_validity_rate)} |
| Judge: answer faithful to sources | {_pct(summary.judge_faithful_rate)} |
| Judge: answer relevant to question | {_pct(summary.judge_relevant_rate)} |

## By question type

| Tag | N | Pass rate | Answer/abstain accuracy | Content accuracy |
|---|---|---|---|---|
{tag_rows}

## What "pass" means

A question passes when the app answered exactly when it should have (and abstained
exactly when it should have), its citations are all valid, and, when a judge was used,
the judge found the answer faithful to the sources and relevant to the question.

## Failures

{failure_text}

## Trusting these numbers

- `content_accuracy` and `abstention_accuracy` are exact-match checks: cheap and
  reliable, but blind to answer quality beyond one fact or one yes/no decision.
- The judge score is only as good as the judge. Spot-check a random sample of its
  verdicts by hand periodically (see the README's "Trusting the judge" section),
  rather than trusting it blindly, especially before using it to block a release.
- A regression check compares this run's numbers to a saved baseline. It is only
  meaningful when both runs used the same `--llm` and `--judge` backend: comparing a
  `fake`-backed run to a real one will look like either a huge regression or a huge
  improvement, and means nothing.
"""
    Path(path).write_text(text)
