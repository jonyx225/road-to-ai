"""Request and response schemas (Pydantic validates everything for us)."""
from pydantic import BaseModel, Field, field_validator

from app.config import MAX_BATCH_SIZE, MAX_TEXT_LENGTH


def _clean(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty or whitespace only")
    return text


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH, examples=["Great product, works perfectly!"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        return _clean(value)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def each_valid(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            if len(value) > MAX_TEXT_LENGTH:
                raise ValueError(f"each text must be at most {MAX_TEXT_LENGTH} characters")
            cleaned.append(_clean(value))
        return cleaned


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
