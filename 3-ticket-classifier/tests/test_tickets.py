import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tickets.baseline import TfidfClassifier
from tickets.common import Prediction
from tickets.compare import BASELINE_NAME, FINETUNE_NAME, run_comparison
from tickets.config import LABELS
from tickets.data import generate_datasets, load_datasets
from tickets.evaluate import break_even_requests_per_month, evaluate, llm_cost_per_1000
from tickets.llm import LLMClassifier, build_system_prompt, make_client, parse_label, select_few_shot

_KEYWORDS = {
    "refund_return": ["refund", "return", "money back"],
    "billing": ["charge", "invoice", "billing", "bill"],
    "shipping": ["package", "delivery", "tracking", "parcel"],
    "account_access": ["password", "log in", "account", "login"],
    "feature_request": ["add", "wish", "would be great", "love to see"],
}


class FakeClient:
    """Stands in for anthropic.Anthropic(): same `client.messages.create(...)` shape, no network."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"].lower()
        label = next((lab for lab, words in _KEYWORDS.items() if any(w in text for w in words)), "technical_issue")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=label)],
            usage=SimpleNamespace(input_tokens=500, output_tokens=4),
        )


def small_data():
    return generate_datasets(n_train_val=240, n_test=60, seed=3)


def without_api_key(fn):
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        return fn()
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ---------------- data ----------------

def test_generated_splits_have_expected_sizes_and_valid_labels():
    data = small_data()
    assert len(data["train"]) + len(data["val"]) == 240
    assert len(data["test"]) == 60
    for df in data.values():
        assert set(df["label"]) <= set(LABELS)
        assert {"text", "label", "wording"} <= set(df.columns)


def test_no_text_is_shared_between_splits():
    data = small_data()
    texts = [set(df["text"]) for df in data.values()]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])


def test_novel_wording_only_appears_in_the_test_set():
    data = small_data()
    assert "novel" not in set(data["train"]["wording"]) | set(data["val"]["wording"])
    assert "novel" in set(data["test"]["wording"])


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(small_data()["test"], small_data()["test"])


def test_load_datasets_validates_columns_and_labels(tmp_path):
    for split in ("train", "val", "test"):
        pd.DataFrame({"text": ["hi"], "label": ["billing"]}).to_csv(tmp_path / f"{split}.csv", index=False)
    loaded = load_datasets(tmp_path)
    assert (loaded["train"]["wording"] == "all").all()  # default when the column is missing

    pd.DataFrame({"text": ["hi"], "label": ["not_a_label"]}).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="labels not listed"):
        load_datasets(tmp_path)

    pd.DataFrame({"body": ["hi"], "label": ["billing"]}).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_datasets(tmp_path)


# ---------------- prompt building and parsing ----------------

def test_system_prompt_lists_every_label_and_optional_examples():
    zero = build_system_prompt()
    assert all(label in zero for label in LABELS)
    assert "Examples:" not in zero
    few = build_system_prompt(examples=[("My card was charged twice", "billing")])
    assert "Ticket: My card was charged twice" in few and "Category: billing" in few


def test_select_few_shot_is_balanced_and_deterministic():
    train = small_data()["train"]
    first = select_few_shot(train, 2)
    assert len(first) == 2 * len(LABELS)
    assert {label for _, label in first} == set(LABELS)
    assert first == select_few_shot(train, 2)


def test_parse_label_handles_messy_replies():
    assert parse_label("billing") == "billing"
    assert parse_label("  Billing.\n") == "billing"
    assert parse_label("`refund_return`") == "refund_return"
    assert parse_label("I think this is shipping") == "shipping"
    assert parse_label("billing or refund_return") == "unparsed"  # ambiguous: refuse to guess
    assert parse_label("no idea") == "unparsed"


# ---------------- LLM classifier (fake client, no network) ----------------

def test_llm_classifier_sends_expected_request_and_reads_usage():
    client = FakeClient()
    clf = LLMClassifier("SYSTEM PROMPT", client=client, model="test-model", temperature=0.0)
    preds = clf.predict(["I want a refund please", "The app keeps crashing"])
    assert [p.label for p in preds] == ["refund_return", "technical_issue"]
    assert preds[0].input_tokens == 500 and preds[0].output_tokens == 4
    call = client.calls[0]
    assert call["model"] == "test-model" and call["system"] == "SYSTEM PROMPT" and call["temperature"] == 0.0
    assert "I want a refund please" in call["messages"][0]["content"]


def test_temperature_is_omitted_when_none():
    client = FakeClient()
    LLMClassifier("p", client=client, temperature=None).predict(["hello"])
    assert "temperature" not in client.calls[0]


def test_cache_prevents_repeat_calls_and_respects_prompt_changes(tmp_path):
    cache = tmp_path / "cache.jsonl"
    first = FakeClient()
    LLMClassifier("prompt A", client=first, cache_path=cache).predict(["refund me", "password reset"])
    assert len(first.calls) == 2

    second = FakeClient()
    replay = LLMClassifier("prompt A", client=second, cache_path=cache).predict(["refund me", "password reset"])
    assert len(second.calls) == 0 and all(p.cached for p in replay)

    third = FakeClient()  # a different prompt is a different experiment: must not reuse old answers
    LLMClassifier("prompt B", client=third, cache_path=cache).predict(["refund me"])
    assert len(third.calls) == 1


def test_make_client_explains_missing_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        without_api_key(make_client)


# ---------------- evaluation and cost math ----------------

def test_evaluate_reports_accuracy_by_wording_and_unparsed():
    test = pd.DataFrame(
        {"text": list("abcd"), "label": ["billing", "billing", "shipping", "shipping"],
         "wording": ["direct", "direct", "novel", "novel"]}
    )
    preds = [Prediction("billing", 0.01), Prediction("shipping", 0.02),
             Prediction("shipping", 0.03), Prediction("unparsed", 0.04)]
    result = evaluate(test, preds, LABELS)
    assert result["accuracy"] == 0.5
    assert result["by_wording"]["direct"]["accuracy"] == 0.5
    assert result["by_wording"]["novel"]["accuracy"] == 0.5
    assert result["unparsed"] == 1
    assert result["latency"]["p50_ms"] == pytest.approx(25.0)


def test_evaluate_rejects_length_mismatch():
    test = pd.DataFrame({"text": ["a"], "label": ["billing"], "wording": ["direct"]})
    with pytest.raises(ValueError, match="predictions"):
        evaluate(test, [], LABELS)


def test_cost_and_break_even_math():
    preds = [Prediction("billing", 0.1, input_tokens=1000, output_tokens=100)]
    assert llm_cost_per_1000(preds, price_in_per_mtok=1.0, price_out_per_mtok=5.0) == pytest.approx(1.5)
    assert break_even_requests_per_month(150, 0.0015) == pytest.approx(100_000)
    assert break_even_requests_per_month(150, 0) is None


# ---------------- baseline ----------------

def test_baseline_learns_the_training_distribution():
    data = small_data()
    clf = TfidfClassifier().fit(data["train"])
    preds = clf.predict(data["val"]["text"].tolist())
    assert evaluate(data["val"], preds, LABELS)["accuracy"] > 0.9


# ---------------- end-to-end comparison ----------------

def test_run_comparison_with_fake_llm_writes_all_outputs(tmp_path):
    client = FakeClient()
    out = run_comparison(
        ["baseline", "finetune", "llm"], data=small_data(), reports_dir=tmp_path,
        finetuned_dir=tmp_path / "no-model-here", llm_client=client, shots=1, verbose=False,
    )
    assert BASELINE_NAME in out["results"]
    assert "LLM few-shot (1/class)" in out["results"]
    assert FINETUNE_NAME in out["skipped"]                      # no trained model in that folder
    assert out["extras"]["LLM few-shot (1/class)"]["cost_per_1000"] == pytest.approx(0.52)
    for name in ["comparison.md", "results.json", "predictions_baseline.csv", "predictions_llm.csv"]:
        assert (tmp_path / name).exists()
    assert (tmp_path / "figures" / "accuracy_by_wording.png").exists()
    assert "Fine-tuned DistilBERT" in (tmp_path / "comparison.md").read_text()  # listed under "Not run"


def test_limit_evaluates_a_subset_of_the_test_set(tmp_path):
    out = run_comparison(["baseline"], data=small_data(), reports_dir=tmp_path, limit=20, verbose=False)
    assert out["results"][BASELINE_NAME]["n"] == 20


def test_missing_api_key_skips_llm_but_keeps_other_methods(tmp_path):
    out = without_api_key(lambda: run_comparison(["baseline", "llm"], data=small_data(), reports_dir=tmp_path, verbose=False))
    assert BASELINE_NAME in out["results"]
    assert any("ANTHROPIC_API_KEY" in reason for reason in out["skipped"].values())


def test_comparison_fails_loudly_when_nothing_could_run(tmp_path):
    with pytest.raises(RuntimeError, match="Nothing ran"):
        without_api_key(lambda: run_comparison(["llm"], data=small_data(), reports_dir=tmp_path, verbose=False))
