"""Steps 2-4: baseline vs. gradient boosting, proper metrics, cost-based threshold.

Run:  python -m churn.train

The order of operations matters and is the main lesson of this project:
  1. Split off a test set and do not touch it until the very end.
  2. Compare models with cross-validation on the training data only.
  3. Pick the decision threshold on out-of-fold training predictions
     (never on the test set).
  4. Report final numbers on the untouched test set.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from churn import plots
from churn.config import (
    CV_FOLDS,
    FIGURES_DIR,
    ID_COLUMN,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
    BusinessCosts,
)
from churn.data import load_data
from churn.evaluate import baseline_profits, best_threshold, profit, profit_curve, summarize
from churn.features import build_preprocessor, select_features
from churn.leakage import find_leaky, single_feature_auc
from churn.report import write_report


def candidate_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """A trivial baseline, a simple linear model, and a gradient-boosted tree model."""

    def make(estimator) -> Pipeline:
        return Pipeline([("prep", build_preprocessor(X)), ("model", estimator)])

    return {
        "Baseline (base rate)": make(DummyClassifier(strategy="prior")),
        "Logistic regression": make(LogisticRegression(max_iter=1000)),
        "Gradient boosting": make(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
        ),
    }


def run_training(
    df: pd.DataFrame,
    costs: BusinessCosts = BusinessCosts(),
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    cv_folds: int = CV_FOLDS,
    verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    for directory in (models_dir, reports_dir, figures_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)

    X = select_features(df)
    y = df[TARGET]

    # Safety net: refuse to train if a remaining feature looks like leakage.
    solo = single_feature_auc(df, list(X.columns))
    leaky = find_leaky(solo)
    if leaky:
        raise ValueError(f"Possible target leakage in features {leaky}. Add them to LEAKY_COLUMNS.")

    # 1. Hold out a test set. Stratify so both sets keep the same churn rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | churn rate {y.mean():.1%}\n")

    # 2. Compare models with cross-validation on the training set only.
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    models = candidate_models(X)
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}
        )
        cv_results[name] = {
            "pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "pr_auc_std": float(scores["test_pr_auc"].std()),
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
        }
    log("Cross-validation on training data (mean +/- std):")
    log(f"{'model':<24}{'PR-AUC':>16}{'ROC-AUC':>16}")
    for name, r in cv_results.items():
        log(f"{name:<24}{r['pr_auc_mean']:>9.3f} +/-{r['pr_auc_std']:.3f}{r['roc_auc_mean']:>9.3f} +/-{r['roc_auc_std']:.3f}")

    real_models = [n for n in cv_results if not n.startswith("Baseline")]
    selected = max(real_models, key=lambda n: cv_results[n]["pr_auc_mean"])
    log(f"\nSelected model (best CV PR-AUC): {selected}")

    # 3. Choose the threshold from out-of-fold predictions on the TRAINING set.
    oof = cross_val_predict(models[selected], X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof, costs)
    train_curve = profit_curve(y_train, oof, costs)
    log(f"Profit-maximizing threshold: {threshold:.2f} (the default 0.50 is rarely right)\n")

    # 4. Final evaluation on the untouched test set.
    fitted = {name: pipe.fit(X_train, y_train) for name, pipe in models.items()}
    test_proba = {name: pipe.predict_proba(X_test)[:, 1] for name, pipe in fitted.items()}
    proba = test_proba[selected]

    at_default = summarize(y_test, proba, 0.5)
    at_tuned = summarize(y_test, proba, threshold)
    scale = 1000 / len(y_test)  # report profit per 1,000 customers
    base = baseline_profits(y_test, costs)
    profits = {
        "contact_nobody": base["contact_nobody"] * scale,
        "contact_everyone": base["contact_everyone"] * scale,
        "default_threshold": profit(at_default["tp"], at_default["fp"], costs) * scale,
        "tuned_threshold": profit(at_tuned["tp"], at_tuned["fp"], costs) * scale,
    }

    # Lift: how much better than random are the customers the model ranks highest?
    order = np.argsort(-proba)
    top_n = max(1, len(proba) // 10)
    top_rate = float(np.asarray(y_test)[order][:top_n].mean())

    # How sensitive is the profit to our guess about the offer's success rate?
    sensitivity = []
    for rate in (0.10, 0.20, 0.40, 0.60):
        alt = BusinessCosts(costs.customer_value, costs.contact_cost, rate)
        sensitivity.append(
            {"save_rate": rate, "profit_per_1000": profit(at_tuned["tp"], at_tuned["fp"], alt) * scale}
        )

    # Which inputs matter? Shuffle each raw feature and see how much the score drops.
    imp = permutation_importance(
        fitted[selected], X_test, y_test, scoring="average_precision", n_repeats=5, random_state=RANDOM_STATE
    )
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    log(f"Test set, {selected}:")
    log(f"  PR-AUC {at_tuned['pr_auc']:.3f} (random guessing = {y_test.mean():.3f}) | ROC-AUC {at_tuned['roc_auc']:.3f}")
    log(f"  At threshold 0.50: precision {at_default['precision']:.2f}, recall {at_default['recall']:.2f}")
    log(f"  At threshold {threshold:.2f}: precision {at_tuned['precision']:.2f}, recall {at_tuned['recall']:.2f}, "
        f"contacts {at_tuned['share_contacted']:.0%} of customers")
    log(f"  Top 10% highest-risk customers: {top_rate:.0%} churn (vs {y_test.mean():.0%} overall)")
    log("  Profit per 1,000 customers:")
    for label, value in profits.items():
        log(f"    {label:<20}${value:>10,.0f}")

    metrics = {
        "data": {
            "n_customers": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "churn_rate": float(y.mean()),
            "features": list(X.columns),
            "excluded_leaky": [c for c in df.columns if c not in X.columns and c not in (TARGET, ID_COLUMN)],
        },
        "cv": cv_results,
        "selected_model": selected,
        "threshold": float(threshold),
        "test_by_model": {
            n: {
                "pr_auc": summarize(y_test, p, 0.5)["pr_auc"],
                "roc_auc": summarize(y_test, p, 0.5)["roc_auc"],
            }
            for n, p in test_proba.items()
        },
        "test_at_default": at_default,
        "test_at_tuned": at_tuned,
        "top_decile": {"churn_rate": top_rate, "lift": top_rate / float(y_test.mean())},
        "profit_per_1000": profits,
        "sensitivity": sensitivity,
        "importance": importance.head(10).to_dict(orient="records"),
        "costs": asdict(costs),
    }

    # Save artifacts: model, metrics, charts, and the plain-language report.
    joblib.dump(
        {
            "pipeline": fitted[selected],
            "threshold": float(threshold),
            "features": list(X.columns),
            "costs": asdict(costs),
            "model_name": selected,
        },
        Path(models_dir) / "churn_model.joblib",
    )
    (Path(reports_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plots.plot_pr_curves(y_test, {k: v for k, v in test_proba.items() if not k.startswith("Baseline")}, Path(figures_dir) / "pr_curve.png")
    plots.plot_profit_curve(train_curve, threshold, Path(figures_dir) / "profit_curve.png")
    plots.plot_importance(importance, Path(figures_dir) / "feature_importance.png")
    write_report(metrics, Path(reports_dir) / "report.md")
    log(f"\nSaved model, metrics.json, figures and report.md to {reports_dir} / {models_dir}")
    return metrics


def main() -> None:
    run_training(load_data())


if __name__ == "__main__":
    main()
