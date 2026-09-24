import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import pytest

from churn.config import LEAKY_COLUMNS, TARGET, BusinessCosts
from churn.data import generate_churn_data
from churn.evaluate import baseline_profits, best_threshold, profit, summarize
from churn.features import select_features
from churn.leakage import find_leaky, single_feature_auc
from churn.predict import score_customers
from churn.train import run_training


@lru_cache(maxsize=1)
def small_data() -> pd.DataFrame:
    return generate_churn_data(n=2500, seed=1)


@lru_cache(maxsize=1)
def trained():
    """Train once (small data, 3-fold CV) and reuse the result across tests."""
    tmp = Path(tempfile.mkdtemp())
    metrics = run_training(
        small_data(),
        models_dir=tmp / "models",
        reports_dir=tmp / "reports",
        figures_dir=tmp / "reports" / "figures",
        cv_folds=3,
        verbose=False,
    )
    return tmp, metrics


# ---------- data ----------

def test_generated_data_is_imbalanced_reproducible_and_unique():
    df = small_data()
    assert 0.12 < df[TARGET].mean() < 0.25
    assert df["customer_id"].is_unique
    pd.testing.assert_frame_equal(df, generate_churn_data(n=2500, seed=1))


def test_generated_data_has_leaky_column_and_missing_values():
    df = small_data()
    assert all(col in df.columns for col in LEAKY_COLUMNS)
    assert df.isna().any().any()


def test_select_features_drops_target_id_and_leaky_columns():
    X = select_features(small_data())
    assert TARGET not in X.columns
    assert "customer_id" not in X.columns
    assert not set(LEAKY_COLUMNS) & set(X.columns)


# ---------- leakage ----------

def test_leakage_detector_flags_only_the_leaky_column():
    df = small_data()
    candidates = [c for c in df.columns if c not in (TARGET, "customer_id")]
    assert find_leaky(single_feature_auc(df, candidates)) == ["final_invoice_sent"]


# ---------- metrics and money ----------

def test_profit_math():
    costs = BusinessCosts(customer_value=250, contact_cost=20, save_rate=0.4)
    assert costs.gain_per_true_positive == 80
    assert profit(tp=10, fp=5, costs=costs) == 10 * 80 - 5 * 20


def test_summarize_counts_and_rates():
    result = summarize([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_best_threshold_separates_clearly_separable_scores():
    y = [1] * 20 + [0] * 80
    proba = [0.9] * 20 + [0.1] * 80
    threshold = best_threshold(y, proba, BusinessCosts())
    assert 0.1 < threshold <= 0.9


def test_contacting_everyone_loses_money_when_churn_is_rare():
    y = [1] * 18 + [0] * 82
    baselines = baseline_profits(y, BusinessCosts())
    assert baselines["contact_nobody"] == 0
    assert baselines["contact_everyone"] < 0


# ---------- end to end ----------

def test_training_creates_artifacts():
    tmp, _ = trained()
    assert (tmp / "models" / "churn_model.joblib").exists()
    for name in ["metrics.json", "report.md"]:
        assert (tmp / "reports" / name).exists()
    for name in ["pr_curve.png", "profit_curve.png", "feature_importance.png"]:
        assert (tmp / "reports" / "figures" / name).exists()


def test_model_beats_random_guessing_and_the_no_model_strategies():
    _, metrics = trained()
    base_rate = metrics["data"]["churn_rate"]
    assert metrics["test_at_tuned"]["pr_auc"] > base_rate + 0.10
    profits = metrics["profit_per_1000"]
    assert profits["tuned_threshold"] > profits["contact_nobody"]
    assert profits["tuned_threshold"] > profits["contact_everyone"]
    assert 0 < metrics["threshold"] < 1


def test_leaky_column_was_excluded_from_features():
    _, metrics = trained()
    assert "final_invoice_sent" not in metrics["data"]["features"]
    assert "final_invoice_sent" in metrics["data"]["excluded_leaky"]


def test_training_refuses_to_run_when_a_new_leaky_feature_appears():
    df = small_data().assign(cancellation_reason_code=lambda d: d[TARGET])
    with pytest.raises(ValueError, match="leakage"):
        run_training(df, models_dir=Path(tempfile.mkdtemp()), reports_dir=Path(tempfile.mkdtemp()),
                     figures_dir=Path(tempfile.mkdtemp()), cv_folds=3, verbose=False)


# ---------- scoring ----------

def test_score_customers_sorts_by_risk_and_applies_threshold():
    tmp, _ = trained()
    artifact = joblib.load(tmp / "models" / "churn_model.joblib")
    scored = score_customers(small_data().head(100), artifact)
    assert scored["churn_risk"].is_monotonic_decreasing
    assert (scored["contact"] == (scored["churn_risk"] >= artifact["threshold"])).all()
    assert len(scored) == 100


def test_score_customers_rejects_missing_columns():
    tmp, _ = trained()
    artifact = joblib.load(tmp / "models" / "churn_model.joblib")
    with pytest.raises(ValueError, match="missing required column"):
        score_customers(small_data()[["age"]], artifact)
