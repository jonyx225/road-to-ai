"""An offline stand-in for the Anthropic client. NOT a real language model.

It exposes the same `client.messages.create(...)` call as the SDK, but "answers" by
picking the source sentence that shares the most words with the question and citing it
(or abstaining when nothing overlaps enough). That is enough to run the whole app, the
web UI and the tests without an API key. Answers are extractive and rough, and they
say nothing about how a real model would behave.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from rag.bm25 import tokenize
from rag.chunking import SENTENCE_SPLIT
from rag.generation import ABSTAIN_TEXT

_SOURCE_BLOCK = re.compile(r"^\[(\d+)\] \(source: .*?\)\n(.*?)(?=\n\n\[\d+\] \(source: |\n</sources>)", re.S | re.M)
MIN_OVERLAP = 2


class FakeAnthropic:
    def __init__(self):
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        text = self._answer(content)
        usage = SimpleNamespace(input_tokens=max(1, len(content) // 4), output_tokens=max(1, len(text) // 4))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)

    @staticmethod
    def _answer(content: str) -> str:
        question = content.rsplit("Question:", 1)[-1].strip()
        wanted = set(tokenize(question))
        best_score, best_sentence, best_n = 0, "", 0
        for number, body in _SOURCE_BLOCK.findall(content):
            for line in body.split("\n"):
                for sentence in SENTENCE_SPLIT.split(line.strip().lstrip("-• ").strip()):
                    overlap = len(wanted & set(tokenize(sentence)))
                    if overlap > best_score:
                        best_score, best_sentence, best_n = overlap, sentence.strip(), int(number)
        if best_score < MIN_OVERLAP:
            return ABSTAIN_TEXT
        return f"{best_sentence} [{best_n}]"
