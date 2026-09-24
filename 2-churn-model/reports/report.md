# Customer Churn Model: Summary for Decision Makers

*Data: 8,000 customers (18% churned). Model: Logistic regression. All numbers below come from a held-out test set of 1,600 customers the model never saw during training.*

## The short version

- **The model finds at-risk customers far better than guessing.** Among the 10% of customers it ranks as highest risk, **52%** actually left, compared with 18% across all customers (2.9x better than random).
- **Recommendation:** offer a retention deal to customers whose risk score is at least **0.20**. That is about **38%** of customers. It reaches **72%** of the customers who would have left, and **35%** of the people contacted really were about to leave.
- **Expected value:** about **$5,800 per 1,000 customers**, compared with -$1,625 if we contacted everyone and $0 if we did nothing.

## Why not just report "accuracy"?

Only 18% of customers churn, so a model that predicts "nobody leaves" would be 82% accurate and completely useless. Instead we looked at:

- **Recall**: of the customers who really leave, how many do we catch?
- **Precision**: of the customers we contact, how many were really about to leave?
- **PR-AUC**: one score summarizing the trade-off between the two (random guessing scores 0.18; higher is better).

| Model | PR-AUC (cross-validated) | ROC-AUC (cross-validated) | PR-AUC (test set) |
|---|---|---|---|
| Baseline (base rate) | 0.184 | 0.500 | 0.184 |
| Logistic regression | 0.444 | 0.779 | 0.471 |
| Gradient boosting | 0.412 | 0.763 | 0.454 |

The simpler model did at least as well as the more complex one, so we chose it: it is easier to explain and maintain. The default cut-off of 0.50 would flag too few people: recall 19% at 0.50 versus 72% at 0.20. The threshold was chosen to maximize profit using the training data only, not the test set.

## What drives churn

The five factors the model relies on most:

1. Contract type
2. How long the customer has been with us
3. Support tickets in the last 90 days
4. Late payments in the last year
5. Internet service type

These show *association*, not cause. For example, removing a tech-support add-on will not necessarily make customers leave. Use them to guide investigation and experiments.

## Assumptions behind the money numbers

| Assumption | Value |
|---|---|
| Value of keeping a customer | $250 |
| Cost of each retention contact | $20 |
| Chance an offer keeps a customer who would leave | 40% |

The offer success rate is a guess, and the result is sensitive to it. **The campaign only breaks even if offers work about 23% of the time or more.**

| If the offer success rate is actually... | Profit per 1,000 customers |
|---|---|
| 10% | -$4,184 |
| 20% | -$856 |
| 40% | $5,800 |
| 60% | $12,456 |

## Caveats and next steps

- **Test before scaling.** Run an A/B test (contact a random half of the flagged customers) to measure the real success rate. This model predicts who is at risk, not who can be saved.
- **A leaky column was excluded:** `final_invoice_sent`. It is only recorded once a customer has left, so using it would make the model look almost perfect while being useless in real life.
- **Model quality is moderate.** Some churn is driven by things we do not observe. Better data (usage trends, complaints, competitor offers) is the most likely way to improve it.
- **Monitor over time.** Re-check performance monthly; customer behavior changes.
- **This report uses synthetic data.** Repeat the analysis on real company data before making decisions.

*Charts: `figures/pr_curve.png`, `figures/profit_curve.png`, `figures/feature_importance.png`, `figures/eda.png`.*
