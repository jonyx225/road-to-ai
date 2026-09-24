"""FastAPI application that serves the sentiment model."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app import config
from app.model import SentimentModel
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
)

logger = logging.getLogger("sentiment-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        app.state.model = SentimentModel.load(config.MODEL_PATH)
        logger.info("Loaded model from %s", config.MODEL_PATH)
    except FileNotFoundError:
        app.state.model = None
        logger.warning(
            "No model found at %s. Run `python scripts/train.py` first.", config.MODEL_PATH
        )
    yield


app = FastAPI(
    title="Sentiment API",
    description="Classifies review text as positive or negative.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model(request: Request) -> SentimentModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = request.app.state.model is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict", response_model=Prediction)
def predict(payload: PredictRequest, request: Request):
    model = get_model(request)
    return model.predict([payload.text])[0]


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request):
    model = get_model(request)
    return BatchPredictResponse(predictions=model.predict(payload.texts))
