"""Turn one question's result into pass/fail checks, and a batch of them into aggregate metrics."""
from __future__ import annotations

from dataclasses import dataclass

from evals.judge import JudgeResult
from rag.pipeline import Answer


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def answer_contains_phrase(answer_text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(answer_text)


@dataclass
class QuestionResult:
    question: str
    tag: str
    expected_answerable: bool
    answer: Answer
    judge: JudgeResult | None
    correct_content: bool | None    # None when there is nothing to check (e.g. no answer_contains)
    correct_abstention: bool        # did "answered" match what we expected?
    citation_valid: bool            # no "cites source numbers that do not exist" warning

    @property
    def passed(self) -> bool:
        checks = [self.correct_abstention, self.citation_valid]
        if self.correct_content is not None:
            checks.append(self.correct_content)
        if self.judge is not None:
            checks += [self.judge.faithful, self.judge.relevant]
        return all(checks)


def score_one(record: dict, answer: Answer, judge_result: JudgeResult | None) -> QuestionResult:
    expected_answerable = record.get("answerable", True)
    correct_content = None
    if expected_answerable and "answer_contains" in record and answer.answered:
        correct_content = answer_contains_phrase(answer.answer, record["answer_contains"])
    return QuestionResult(
        question=record["question"],
        tag=record.get("tag", "all"),
        expected_answerable=expected_answerable,
        answer=answer,
        judge=judge_result,
        correct_content=correct_content,
        correct_abstention=answer.answered == expected_answerable,
        citation_valid=not any("do not exist" in w for w in answer.warnings),
    )


def _rate(flags: list[bool | None]) -> float | None:
    present = [f for f in flags if f is not None]
    return sum(present) / len(present) if present else None


@dataclass
class Summary:
    n: int
    pass_rate: float
    abstention_accuracy: float
    content_accuracy: float | None
    citation_validity_rate: float
    judge_faithful_rate: float | None
    judge_relevant_rate: float | None
    by_tag: dict

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "pass_rate": self.pass_rate,
            "abstention_accuracy": self.abstention_accuracy,
            "content_accuracy": self.content_accuracy,
            "citation_validity_rate": self.citation_validity_rate,
            "judge_faithful_rate": self.judge_faithful_rate,
            "judge_relevant_rate": self.judge_relevant_rate,
            "by_tag": self.by_tag,
        }


def summarize(results: list[QuestionResult]) -> Summary:
    by_tag = {}
    for tag in sorted({r.tag for r in results}):
        group = [r for r in results if r.tag == tag]
        by_tag[tag] = {
            "n": len(group),
            "pass_rate": _rate([r.passed for r in group]),
            "abstention_accuracy": _rate([r.correct_abstention for r in group]),
            "content_accuracy": _rate([r.correct_content for r in group]),
        }
    return Summary(
        n=len(results),
        pass_rate=_rate([r.passed for r in results]) or 0.0,
        abstention_accuracy=_rate([r.correct_abstention for r in results]) or 0.0,
        content_accuracy=_rate([r.correct_content for r in results]),
        citation_validity_rate=_rate([r.citation_valid for r in results]) or 0.0,
        judge_faithful_rate=_rate([r.judge.faithful if r.judge else None for r in results]),
        judge_relevant_rate=_rate([r.judge.relevant if r.judge else None for r in results]),
        by_tag=by_tag,
    )
