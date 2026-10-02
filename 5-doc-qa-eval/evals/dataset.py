"""Loads and validates the eval question set.

Reuses rag.evaluate.load_questions (the same file backs the retrieval-only check from
Project 4) and adds the stricter validation an evaluation suite needs: every question
must have a tag, and every answerable question must say what the correct answer should
contain.
"""
from __future__ import annotations

from pathlib import Path

from rag import config
from rag.evaluate import load_questions as _load_raw_questions


def load_questions(path: Path | str = config.EVAL_PATH) -> list[dict]:
    questions = _load_raw_questions(path)
    for q in questions:
        if not q.get("question", "").strip():
            raise ValueError(f"A question record is missing its text: {q}")
        if "tag" not in q:
            raise ValueError(f"Question is missing a 'tag': {q['question']!r}")
        if q.get("answerable", True) and "answer_contains" not in q:
            raise ValueError(f"Answerable question is missing 'answer_contains': {q['question']!r}")
    return questions
