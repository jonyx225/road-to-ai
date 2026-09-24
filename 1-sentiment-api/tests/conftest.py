"""Shared fixtures: train a throwaway model once, then point the app at it."""
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.training import train


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "sentiment.joblib"
    train(config.BASE_DIR / "data" / "reviews.csv", path)
    return path


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    # Using TestClient as a context manager runs the app's startup (lifespan) code.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    with TestClient(app) as test_client:
        yield test_client
