"""Types shared by every classifier so they can be compared like-for-like."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Prediction:
    label: str
    latency_s: float            # wall-clock time for this single request
    input_tokens: int = 0       # only meaningful for the LLM
    output_tokens: int = 0
    raw: str = ""               # raw model output (LLM only)
    cached: bool = False        # True if replayed from the on-disk LLM cache


class Classifier(Protocol):
    def predict(self, texts: list[str]) -> list[Prediction]: ...
