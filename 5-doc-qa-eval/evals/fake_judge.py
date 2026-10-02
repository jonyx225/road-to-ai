"""A deterministic, offline stand-in for the judge. NOT a real evaluator.

It approximates two things with word-overlap heuristics so the eval suite's plumbing,
its tests, and CI can run for free and without an API key:
  * faithful: the answer shares enough words with the sources to look like it was
    built from them (or it is a correctly-shaped abstention).
  * relevant: the answer shares at least one word with the question.
This exists to make the harness testable, never to judge real model output. Whether a
prompt or chunking change actually improved answer quality is a question only the real
judge (or a human) can answer.
"""
from __future__ import annotations

from evals.judge import JudgeResult
from rag.bm25 import tokenize
from rag.generation import is_abstention

MIN_OVERLAP_WORDS = 2


class FakeJudge:
    def judge(self, question: str, sources_block: str, answer: str) -> JudgeResult:
        if is_abstention(answer):
            return JudgeResult(True, True, "Correctly shaped abstention.", raw="(fake judge)")
        answer_words = set(tokenize(answer))
        faithful = len(answer_words & set(tokenize(sources_block))) >= MIN_OVERLAP_WORDS
        relevant = len(answer_words & set(tokenize(question))) >= 1
        return JudgeResult(faithful, relevant, "word-overlap heuristic, not a real judgement", raw="(fake judge)")
