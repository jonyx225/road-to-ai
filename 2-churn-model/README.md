# Customer Churn Model: Evaluate It Properly

A hands-on project about the part of machine learning that separates juniors from
seniors: **evaluating a model honestly** and **connecting it to a business decision**.

You will predict which subscription customers are about to cancel, and then answer
the questions a manager actually cares about: *Who should we contact? How much money
does it make? How sure are we?*

## What this project teaches

| Step | Skill | Where |
|------|-------|-------|
| 1 | Explore the data, spot class imbalance and **target leakage** | `python -m churn.explore` |
| 2 | Build a baseline, then compare models with cross-validation | `python -m churn.train` |
| 3 | Judge with **precision, recall, PR-AUC** instead of accuracy | `churn/evaluate.py` |
| 4 | Choose the decision threshold by **profit**, not by 0.5 | `churn/evaluate.py`, `churn/train.py` |
| 5 | Explain results to a non-technical reader | `reports/report.md` (auto-generated) |

## Project structure

```
churn-model/
├── churn/
│   ├── config.py      # paths, column names, business costs, leaky columns
│   ├── data.py        # synthetic data generator + loader
│   ├── features.py    # feature selection + preprocessing pipeline
│   ├── leakage.py     # single-feature leakage detector
│   ├── evaluate.py    # metrics, profit curve, threshold tuning
│   ├── explore.py     # step 1: EDA and leakage checks
│   ├── train.py       # steps 2-4: compare models, tune threshold, save artifacts
│   ├── report.py      # step 5: writes reports/report.md
│   ├── plots.py       # charts
│   └── predict.py     # score customers with the saved model
├── data/churn.csv     # 8,000 synthetic customers (re-create with `make data`)
├── reports/           # sample output: report.md, metrics.json, figures/
├── tests/test_churn.py
├── Makefile, requirements*.txt, pytest.ini
└── .github/workflows/ci.yml
```

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m churn.explore              # step 1: imbalance, missing values, leakage
python -m churn.train                # steps 2-5: models, threshold, report, charts
python -m churn.predict --input data/churn.csv --top 10
pytest -v
```

`data/churn.csv` is already included. Run `python -m churn.data` to regenerate it.
The `reports/` folder in this zip is sample output; running the commands above
overwrites it (numbers can shift slightly with different library versions).

## The workflow, and what to look for

**Step 1: Explore (`churn.explore`).** Only about 18% of customers churn, so a model
that always predicts "stays" is 82% accurate and worthless. The script also scores
every column *on its own*. One column, `final_invoice_sent`, predicts churn with
ROC-AUC 0.97 by itself. It is only sent when someone cancels, so it is **leakage**:
using it would inflate the model's score from about 0.75 to 0.98 while making it
useless in production. It is excluded in `churn/config.py`, and `train.py` refuses to
run if any remaining feature looks that suspicious.

**Step 2: Baseline vs. models (`churn.train`).** A `DummyClassifier` sets the floor
that any real model must beat. A logistic regression and a gradient-boosting model are
then compared with 5-fold cross-validation **on the training data only**.

**Step 3: The right metrics.** PR-AUC and precision/recall respond to the rare
class; accuracy does not. ROC-AUC is shown too, but on imbalanced data it can look
flattering.

**Step 4: Threshold by profit.** A model outputs a probability. Turning it into
"contact / don't contact" needs a cut-off, and 0.5 is rarely right. In
`config.BusinessCosts` we set the value of a saved customer ($250), the cost of a
contact ($20), and the offer's success rate (40%). The threshold is chosen from
**out-of-fold predictions on the training set**, so the test set stays untouched
until the final report.

**Step 5: Communicate (`reports/report.md`).** The report leads with the decision
and its expected value, explains the metrics in plain language, states the
assumptions, and shows how sensitive the result is to the one number nobody knows
(the offer success rate).

## Sample results (synthetic data, held-out test set)

| | |
|---|---|
| Churn rate | 18% |
| Selected model | Logistic regression (PR-AUC 0.47 vs. 0.18 for random guessing) |
| Top 10% highest-risk customers | 52% actually churn (2.9x lift) |
| Threshold chosen by profit | 0.20 (contacts ~38% of customers) |
| At that threshold | precision 35%, recall 72% |
| Profit per 1,000 customers | about $5,800, vs. -$1,625 if you contact everyone |
| Break-even offer success rate | about 23% |

**Why did the simpler model win?** The synthetic data was generated with a
logistic-style rule, so logistic regression fits it very well. On real, messier data
gradient boosting often wins. Do not assume the complex model is better: let
cross-validation decide, and prefer the simpler model when scores are close.

## Use your own data

1. Put your CSV in `data/` and point `DATA_PATH` in `churn/config.py` at it.
2. Make sure it has an ID column (`ID_COLUMN`) and a 0/1 target column (`TARGET`).
   For a typical "Telco Customer Churn" file that means renaming `customerID` to
   `customer_id`, mapping `Churn` from `Yes/No` to `1/0` as `churned`, and converting
   `TotalCharges` to numbers with `pd.to_numeric(..., errors="coerce")`.
3. Run `python -m churn.explore` and read the leakage table. Add any suspicious
   columns to `LEAKY_COLUMNS`.
4. Replace the numbers in `BusinessCosts` with your company's real economics.
5. Set `IS_SYNTHETIC_DATA = False`, then run `python -m churn.train`.

## Scoring new customers

```bash
python -m churn.predict --input new_customers.csv
```

The file needs the same feature columns as the training data. It writes
`reports/at_risk_customers.csv` sorted by risk with a `contact` flag. (Scoring the
training file itself, as the quickstart does, is only a demo: those customers were
used to fit the model, so their scores are optimistic.)

## Limitations to be aware of

- The split is random. Real churn models should usually be validated **by time**
  (train on the past, test on a later period), because that mimics production.
- Predicting *who will leave* is not the same as knowing *who can be saved*. Only an
  A/B test measures that.
- Feature importance here shows association, not cause.

## Exercises

1. Add `class_weight="balanced"` to the logistic regression. What happens to the
   probabilities, the best threshold, and the profit?
2. Change `save_rate` to 0.15 and re-run. Does the campaign still make money? What
   does the report say?
3. Add a probability-calibration check (`sklearn.calibration.calibration_curve`).
   Can you trust "0.30 means 30% chance"?
4. Make the split time-based by adding a `signup_date` column to the generator.
5. Tune the gradient-boosting model with `RandomizedSearchCV`. Does it beat logistic
   regression on cross-validated PR-AUC?
6. Add a `--model` option to `churn.predict` and write a test for it.
