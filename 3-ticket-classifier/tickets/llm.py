"""Approach B: no training. Describe the labels in a prompt, show a few examples,
and let a hosted LLM classify each ticket.

Uses the Anthropic Python SDK (`pip install anthropic`, set ANTHROPIC_API_KEY).
Every response is cached on disk, so a crashed or repeated run costs nothing extra.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tickets.common import Prediction
from tickets.config import (
    LABEL_DESCRIPTIONS,
    LABELS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RANDOM_STATE,
    SHOTS_PER_CLASS,
)

UNPARSED = "unparsed"


def select_few_shot(train: pd.DataFrame, shots_per_class: int, seed: int = RANDOM_STATE) -> list[tuple[str, str]]:
    """Pick `shots_per_class` random training examples per label, then shuffle them.

    Shuffling matters: a block of same-label examples at the end can bias the model.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for label in LABELS:
        group = train[train["label"] == label]
        n = min(shots_per_class, len(group))
        for idx in rng.choice(group.index.to_numpy(), size=n, replace=False):
            examples.append((str(train.loc[idx, "text"]), label))
    rng.shuffle(examples)
    return examples


def build_system_prompt(
    labels: list[str] = LABELS,
    descriptions: dict[str, str] = LABEL_DESCRIPTIONS,
    examples: list[tuple[str, str]] | None = None,
) -> str:
    """The instructions the model sees on every request. Zero examples = zero-shot."""
    lines = [
        "You are a triage assistant for a customer support team.",
        "Classify each support ticket into exactly one of these categories:",
        "",
    ]
    lines += [f"- {label}: {descriptions[label]}" for label in labels]
    lines += [
        "",
        "Rules:",
        "- Choose the single best category, even if the ticket is ambiguous.",
        "- Reply with ONLY the category name, exactly as written above. No explanation.",
    ]
    if examples:
        lines += ["", "Examples:"]
        for text, label in examples:
            lines += ["", f"Ticket: {text}", f"Category: {label}"]
    return "\n".join(lines)


def parse_label(raw: str, labels: list[str] = LABELS) -> str:
    """Map the model's reply to a valid label, or 'unparsed' if it cannot be done."""
    cleaned = raw.strip().lower().strip("`*\"' .:\n")
    if cleaned in labels:
        return cleaned
    found = [label for label in labels if re.search(rf"\b{re.escape(label)}\b", cleaned)]
    return found[0] if len(found) == 1 else UNPARSED


def make_client():
    """Create the Anthropic client, failing early with a helpful message."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a key in the Anthropic Console and run "
            "`export ANTHROPIC_API_KEY=...` (Windows PowerShell: `$env:ANTHROPIC_API_KEY='...'`)."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install the SDK first: pip install anthropic") from exc
    return anthropic.Anthropic()


class LLMClassifier:
    def __init__(
        self,
        system_prompt: str,
        client=None,
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float | None = LLM_TEMPERATURE,
        cache_path: Path | None = None,
        labels: list[str] = LABELS,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.labels = labels
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                record = json.loads(line)
                self._cache[record["key"]] = record

    def _key(self, text: str) -> str:
        payload = json.dumps([self.model, self.temperature, self.system_prompt, text])
        return hashlib.sha256(payload.encode()).hexdigest()

    def _call_api(self, text: str) -> dict:
        if self.client is None:
            self.client = make_client()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Ticket: {text}\nCategory:"}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        start = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start
        return {
            "raw": "".join(block.text for block in response.content if getattr(block, "type", "text") == "text"),
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "latency_s": latency,
        }

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classify each text. Progress is saved after every call, so re-running resumes."""
        predictions = []
        for i, text in enumerate(texts, start=1):
            key = self._key(text)
            cached = key in self._cache
            record = self._cache[key] if cached else {**self._call_api(text), "key": key}
            if not cached:
                self._cache[key] = record
                if self.cache_path:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.cache_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
            predictions.append(
                Prediction(
                    label=parse_label(record["raw"], self.labels),
                    latency_s=record["latency_s"],
                    input_tokens=record["input_tokens"],
                    output_tokens=record["output_tokens"],
                    raw=record["raw"],
                    cached=cached,
                )
            )
            if self.verbose and i % 25 == 0:
                print(f"  LLM: {i}/{len(texts)} tickets classified")
        return predictions
