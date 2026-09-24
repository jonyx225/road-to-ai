import pytest

POSITIVE = "I absolutely love this product, it works perfectly and is great value!"
NEGATIVE = "Terrible quality, it broke after two days and I want a refund."


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_positive(client):
    response = client.post("/predict", json={"text": POSITIVE})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_predict_negative(client):
    response = client.post("/predict", json={"text": NEGATIVE})
    assert response.status_code == 200
    assert response.json()["label"] == "negative"


def test_predict_response_shape(client):
    body = client.post("/predict", json={"text": POSITIVE}).json()
    assert set(body) == {"label", "confidence", "probabilities"}
    assert 0.5 <= body["confidence"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_strips_whitespace(client):
    response = client.post("/predict", json={"text": f"   {POSITIVE}   \n"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_predict_rejects_blank_text(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_wrong_type(client):
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_predict_accepts_long_text_under_limit(client):
    response = client.post("/predict", json={"text": "great " * 800})  # 4,800 chars
    assert response.status_code == 200


def test_predict_rejects_text_over_limit(client):
    response = client.post("/predict", json={"text": "a" * 5001})
    assert response.status_code == 422


def test_batch_predict_keeps_order(client):
    response = client.post("/predict/batch", json={"texts": [POSITIVE, NEGATIVE]})
    assert response.status_code == 200
    labels = [p["label"] for p in response.json()["predictions"]]
    assert labels == ["positive", "negative"]


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_batch_rejects_too_many_items(client):
    response = client.post("/predict/batch", json={"texts": [POSITIVE] * 33})
    assert response.status_code == 422


def test_batch_rejects_blank_item(client):
    response = client.post("/predict/batch", json={"texts": [POSITIVE, "  "]})
    assert response.status_code == 422


def test_health_reports_degraded_without_model(client_without_model):
    body = client_without_model.get("/health").json()
    assert body == {"status": "degraded", "model_loaded": False}


def test_predict_returns_503_without_model(client_without_model):
    response = client_without_model.post("/predict", json={"text": POSITIVE})
    assert response.status_code == 503
