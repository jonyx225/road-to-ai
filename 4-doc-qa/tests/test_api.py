"""API tests. They are skipped automatically when FastAPI / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rag import config
from rag.api import app
from tests.helpers import sample_index


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("index")
    sample_index().save(path)
    return path


@pytest.fixture()
def client(index_dir, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "LLM_BACKEND", "fake")
    with TestClient(app) as test_client:  # the context manager runs the startup code
        yield test_client


@pytest.fixture()
def client_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "missing")
    monkeypatch.setattr(config, "LLM_BACKEND", "fake")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_loaded_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 36 and body["documents"] == 7


def test_sources_lists_the_indexed_documents(client):
    names = [d["source"] for d in client.get("/sources").json()["documents"]]
    assert "returns-and-refunds.md" in names and len(names) == 7


def test_ask_returns_an_answer_with_numbered_sources(client):
    response = client.post("/ask", json={"question": "How many days do I have to return unused gear?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True and body["citations"]
    assert body["sources"][0]["n"] == 1 and "source" in body["sources"][0]


def test_ask_abstains_on_an_unanswerable_question(client):
    body = client.post("/ask", json={"question": "Do you sell kayaks?"}).json()
    assert body["answered"] is False and body["citations"] == []


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_search_modes_are_accepted(client):
    for mode in ("dense", "bm25", "hybrid"):
        assert client.post("/ask", json={"question": "tent warranty", "mode": mode}).status_code == 200


def test_missing_index_gives_degraded_health_and_503(client_without_index):
    assert client_without_index.get("/health").json()["status"] == "no_index"
    assert client_without_index.post("/ask", json={"question": "hi"}).status_code == 503
    assert client_without_index.get("/sources").status_code == 503


def test_missing_api_key_gives_503_with_a_hint(index_dir, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(app) as test_client:
        response = test_client.post("/ask", json={"question": "How long is the warranty on tents?"})
    assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "<title>Ask the docs</title>" in response.text
