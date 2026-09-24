import pandas as pd
import pytest

from app import config
from app.model import SentimentModel
from app.training import load_data, train

DATA = config.BASE_DIR / "data" / "reviews.csv"


def test_train_saves_a_loadable_model(tmp_path):
    output = tmp_path / "model.joblib"
    metrics = train(DATA, output)

    assert output.exists()
    assert metrics["accuracy"] > 0.8

    model = SentimentModel.load(output)
    assert model.metadata["classes"] == ["negative", "positive"]
    result = model.predict(["Great product, works perfectly!"])[0]
    assert result["label"] == "positive"


def test_load_data_requires_expected_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"review": ["hi"], "sentiment": ["positive"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_data(bad_csv)


def test_load_data_drops_blank_and_duplicate_rows(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"text": ["good", "good", None, "bad"], "label": ["positive", "positive", "positive", "negative"]}
    ).to_csv(csv_path, index=False)
    assert len(load_data(csv_path)) == 2
