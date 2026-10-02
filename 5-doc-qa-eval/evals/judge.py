"""LLM-as-judge: score whether a generated answer is faithful to its sources.

The cheap checks (does the answer contain the expected phrase? did it abstain when it
should have?) catch a lot, but they cannot tell you whether the WORDING of an answer is
actually supported by its sources, or whether it quietly adds a claim the sources never
made. That is what a judge is for: a second model call that reads the same sources the
first model saw, plus the answer it produced, and scores it.

A judge is not immune to the mistakes it is meant to catch: it is a language model
judging a language model, and it can be fooled the same way. Treat its score as a
strong signal, not as ground truth (see "Trusting the judge" in the README), and
spot-check a sample of its verdicts by hand before using it to block a release.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag import config
from rag.generation import LLMError, make_client
from rag.pipeline import Answer

JUDGE_SYSTEM_PROMPT = """You are grading whether an AI assistant's answer is faithful to the sources it was given.

You will see the numbered sources, the question, and the assistant's answer.

Score two things:
- "faithful": true only if every factual claim in the answer is directly supported by
  the sources. An answer that adds unsupported facts, overstates certainty, or
  contradicts a source is NOT faithful, even if the added facts happen to be true.
- "relevant": true if the answer actually addresses the question asked. A correct
  refusal to answer, when the sources genuinely lack the information, counts as relevant.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{"faithful": true or false, "relevant": true or false, "reason": "one short sentence"}"""


@dataclass
class JudgeResult:
    faithful: bool
    relevant: bool
    reason: str
    raw: str


class JudgeParseError(ValueError):
    """The judge's reply could not be parsed into the expected JSON shape."""


def format_sources_for_judge(answer: Answer) -> str:
    return "\n\n".join(
        f"[{s['n']}] (source: {s['source']} | section: {s['section']})\n{s['text']}" for s in answer.sources
    )


def build_judge_message(question: str, sources_block: str, answer: str) -> str:
    return f"<sources>\n{sources_block}\n</sources>\n\nQuestion: {question}\n\nAssistant's answer: {answer}"


def parse_judge_response(raw: str) -> JudgeResult:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise JudgeParseError(f"No JSON object found in the judge's reply: {raw!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"The judge's reply was not valid JSON: {raw!r}") from exc
    if not {"faithful", "relevant"} <= set(data):
        raise JudgeParseError(f"The judge's reply is missing required fields: {data!r}")
    return JudgeResult(faithful=bool(data["faithful"]), relevant=bool(data["relevant"]),
                        reason=str(data.get("reason", "")), raw=raw)


class LLMJudge:
    """Uses a chat client (the real Anthropic SDK, or anything with the same shape) to grade one answer."""

    def __init__(self, client=None, model: str = config.LLM_MODEL, max_tokens: int = 200):
        self.client = client   # None = create the real Anthropic client on first use
        self.model = model
        self.max_tokens = max_tokens

    def judge(self, question: str, sources_block: str, answer: str) -> JudgeResult:
        if self.client is None:
            self.client = make_client()
        message = build_judge_message(question, sources_block, answer)
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}], temperature=0,
            )
        except Exception as exc:
            raise LLMError(f"The judge request failed: {exc}") from exc
        raw = "".join(b.text for b in response.content if getattr(b, "type", "text") == "text").strip()
        return parse_judge_response(raw)


def make_judge(backend: str = config.LLM_BACKEND):
    if backend == "anthropic":
        return LLMJudge()
    if backend == "fake":
        from evals.fake_judge import FakeJudge

        return FakeJudge()
    raise ValueError(f"Unknown judge backend '{backend}'. Choose 'anthropic' or 'fake'.")
