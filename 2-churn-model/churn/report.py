"""Step 5: turn the numbers into a short report a non-technical reader can act on."""
from __future__ import annotations

from pathlib import Path

from churn.config import IS_SYNTHETIC_DATA

FRIENDLY_NAMES = {
    "tenure_months": "How long the customer has been with us",
    "contract": "Contract type",
    "internet_service": "Internet service type",
    "payment_method": "Payment method",
    "tech_support": "Has a tech-support add-on",
    "num_products": "Number of products held",
    "age": "Customer age",
    "monthly_charges": "Monthly bill",
    "total_charges": "Total amount paid so far",
    "support_tickets_90d": "Support tickets in the last 90 days",
    "late_payments_12m": "Late payments in the last year",
    "avg_monthly_gb": "Monthly data usage",
}


def money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def write_report(m: dict, path: Path) -> None:
    tuned, default, top = m["test_at_tuned"], m["test_at_default"], m["top_decile"]
    profits, costs, base_rate = m["profit_per_1000"], m["costs"], m["data"]["churn_rate"]
    selected = m["selected_model"]

    drivers = "\n".join(
        f"{i}. {FRIENDLY_NAMES.get(row['feature'], row['feature'])}"
        for i, row in enumerate(m["importance"][:5], start=1)
    )
    cv_rows = "\n".join(
        f"| {name} | {r['pr_auc_mean']:.3f} | {r['roc_auc_mean']:.3f} | {m['test_by_model'][name]['pr_auc']:.3f} |"
        for name, r in m["cv"].items()
    )
    sens_rows = "\n".join(
        f"| {int(s['save_rate'] * 100)}% | {money(s['profit_per_1000'])} |" for s in m["sensitivity"]
    )
    breakeven = costs["contact_cost"] / (tuned["precision"] * costs["customer_value"]) if tuned["precision"] else float("nan")
    choice_note = (
        "The simpler model did at least as well as the more complex one, so we chose it: it is easier to explain and maintain."
        if selected == "Logistic regression"
        else "The more complex model picked up patterns the simple one missed, so we chose it."
    )
    synthetic_note = (
        "- **This report uses synthetic data.** Repeat the analysis on real company data before making decisions.\n"
        if IS_SYNTHETIC_DATA
        else ""
    )
    excluded = ", ".join(f"`{c}`" for c in m["data"]["excluded_leaky"]) or "none"

    text = f"""# Customer Churn Model: Summary for Decision Makers

*Data: {m['data']['n_customers']:,} customers ({base_rate:.0%} churned). Model: {selected}. All numbers below come from a held-out test set of {m['data']['n_test']:,} customers the model never saw during training.*

## The short version

- **The model finds at-risk customers far better than guessing.** Among the 10% of customers it ranks as highest risk, **{top['churn_rate']:.0%}** actually left, compared with {base_rate:.0%} across all customers ({top['lift']:.1f}x better than random).
- **Recommendation:** offer a retention deal to customers whose risk score is at least **{m['threshold']:.2f}**. That is about **{tuned['share_contacted']:.0%}** of customers. It reaches **{tuned['recall']:.0%}** of the customers who would have left, and **{tuned['precision']:.0%}** of the people contacted really were about to leave.
- **Expected value:** about **{money(profits['tuned_threshold'])} per 1,000 customers**, compared with {money(profits['contact_everyone'])} if we contacted everyone and $0 if we did nothing.

## Why not just report "accuracy"?

Only {base_rate:.0%} of customers churn, so a model that predicts "nobody leaves" would be {1 - base_rate:.0%} accurate and completely useless. Instead we looked at:

- **Recall**: of the customers who really leave, how many do we catch?
- **Precision**: of the customers we contact, how many were really about to leave?
- **PR-AUC**: one score summarizing the trade-off between the two (random guessing scores {base_rate:.2f}; higher is better).

| Model | PR-AUC (cross-validated) | ROC-AUC (cross-validated) | PR-AUC (test set) |
|---|---|---|---|
{cv_rows}

{choice_note} The default cut-off of 0.50 would flag too few people: recall {default['recall']:.0%} at 0.50 versus {tuned['recall']:.0%} at {m['threshold']:.2f}. The threshold was chosen to maximize profit using the training data only, not the test set.

## What drives churn

The five factors the model relies on most:

{drivers}

These show *association*, not cause. For example, removing a tech-support add-on will not necessarily make customers leave. Use them to guide investigation and experiments.

## Assumptions behind the money numbers

| Assumption | Value |
|---|---|
| Value of keeping a customer | {money(costs['customer_value'])} |
| Cost of each retention contact | {money(costs['contact_cost'])} |
| Chance an offer keeps a customer who would leave | {costs['save_rate']:.0%} |

The offer success rate is a guess, and the result is sensitive to it. **The campaign only breaks even if offers work about {breakeven:.0%} of the time or more.**

| If the offer success rate is actually... | Profit per 1,000 customers |
|---|---|
{sens_rows}

## Caveats and next steps

- **Test before scaling.** Run an A/B test (contact a random half of the flagged customers) to measure the real success rate. This model predicts who is at risk, not who can be saved.
- **A leaky column was excluded:** {excluded}. It is only recorded once a customer has left, so using it would make the model look almost perfect while being useless in real life.
- **Model quality is moderate.** Some churn is driven by things we do not observe. Better data (usage trends, complaints, competitor offers) is the most likely way to improve it.
- **Monitor over time.** Re-check performance monthly; customer behavior changes.
{synthetic_note}
*Charts: `figures/pr_curve.png`, `figures/profit_curve.png`, `figures/feature_importance.png`, `figures/eda.png`.*
"""
    Path(path).write_text(text)
