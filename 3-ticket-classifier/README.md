# Fine-tune vs. Prompt: Classifying Support Tickets Three Ways

The most common design decision in applied AI: **do you train a small model on your
own labelled data, or just prompt a big hosted LLM?** This project makes you answer
it with evidence instead of opinion.

You will classify customer-support tickets into six categories (billing, refund/return,
shipping, technical issue, account access, feature request) using three approaches,
then compare them on accuracy, speed and cost.

| # | Approach | Training? | Where it runs |
|---|----------|-----------|---------------|
| 0 | **TF-IDF + logistic regression** (baseline) | seconds | your CPU |
| A | **Fine-tuned DistilBERT** (PyTorch + Hugging Face) | minutes | your machine or a GPU |
| B | **Prompted LLM** (few-shot, Anthropic API) | none | hosted API |

## What this project teaches

- Why you always build a cheap baseline first
- How fine-tuning works, using a plain PyTorch training loop with validation-based checkpointing
- How prompting works: label definitions, few-shot examples, output parsing
- How to evaluate approaches **fairly** (same test set, same metrics) and on **hard cases**, not just averages
- How to reason about **cost, latency and maintenance**, including a break-even calculation

## Project structure

```
ticket-classifier/
├── tickets/
│   ├── config.py       # labels, hyperparameters, model name, prices, hosting assumption
│   ├── data.py         # synthetic ticket generator + loader
│   ├── baseline.py     # TF-IDF + logistic regression
│   ├── finetune.py     # DistilBERT training loop + inference class
│   ├── llm.py          # prompt building, response parsing, cached API classifier
│   ├── evaluate.py     # shared metrics, latency stats, cost and break-even math
│   ├── compare.py      # runs everything on the same test set, writes the report
│   ├── report.py       # reports/comparison.md
│   └── plots.py        # charts
├── data/               # train.csv, val.csv, test.csv (re-create with `make data`)
├── tests/test_tickets.py
├── requirements.txt / requirements-core.txt
└── Makefile, pytest.ini, .github/workflows/ci.yml
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest # full project (installs PyTorch, ~2 GB)
```

Only want the baseline and the tests? `pip install -r requirements-core.txt` is enough.

For the LLM part, create an API key in the Anthropic Console and export it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Step by step

### 1. The data

```bash
python -m tickets.data
```

`data/` already contains the CSVs. Every ticket has a **wording** tag:

- `direct`: obvious keywords ("I want a refund", "I forgot my password")
- `indirect`: the same problems described without the keywords
- `novel`: indirect phrasing that **never appears in the training data**

Training data is 77% direct and 23% indirect, and the test set is 47% direct, 26%
indirect and 27% novel. That is deliberate: keyword-matching models look brilliant on
easy tickets, so the interesting question is what happens on the hard ones. This
mirrors production, where customers keep inventing new ways to say things.

The data is **synthetic** (template-generated), so it is a stand-in. Use your own
`train.csv`, `val.csv` and `test.csv` (columns `text`, `label`) when you have real tickets,
and edit `LABELS` and `LABEL_DESCRIPTIONS` in `tickets/config.py`.

### 2. The baseline

```bash
python -m tickets.compare --methods baseline
```

Measured on the included data (one run on the author's machine):

| Method | Overall | Direct | Indirect | Novel | Median latency |
|---|---|---|---|---|---|
| TF-IDF + LR | 92.7% | 100% | 100% | 72.8% | ~0.4 ms |

It is perfect on wording it has seen and drops sharply on wording it has not. That
gap is exactly what the next two approaches must beat, and it is why you should never
trust a single average accuracy number.

### 3. Fine-tune DistilBERT

```bash
python -m tickets.finetune
```

The first run downloads the pretrained weights (about 250 MB). `finetune.py` walks
through the standard recipe: tokenize, batch with dynamic padding, AdamW with linear
warm-up and decay, gradient clipping, and evaluation after every epoch. It keeps the
best epoch by **validation** macro-F1 and never looks at the test set. The model lands
in `models/distilbert-tickets/`. Training takes longer on a CPU than on a GPU
or Apple-silicon chip, and the run's actual time is recorded in the report.

### 4. Prompt an LLM

Start small, because every ticket is a paid API call:

```bash
python -m tickets.compare --methods llm --limit 50
```

`llm.py` builds a system prompt with a definition of each label plus 3 examples per
label, sends each ticket, and parses the one-word reply. Replies are cached in
`reports/llm_cache.jsonl`, so re-running is free and a crashed run resumes where it stopped.

A rough estimate (characters divided by four) puts the 3-shot prompt at about 850 tokens
per ticket, so all 300 test tickets should cost on the order of **$0.30** at Claude
Haiku 4.5's list price of $1 / $5 per million input / output tokens. The report
computes the real figure from the token counts the API returns. Prices change, so check
the current rates and update `LLM_PRICE_*` in `tickets/config.py` (or the environment
variables of the same name).

Try `--shots 0` (zero-shot) to see how much the examples help, and set `LLM_MODEL` to try
another model. If a model rejects the temperature setting, run with `LLM_TEMPERATURE=none`.

### 5. Compare everything

```bash
python -m tickets.compare
```

This writes `reports/comparison.md` with accuracy overall and by wording, median and p95
latency, cost per 1,000 tickets, and a break-even estimate. It also writes charts
in `reports/figures/`, per-ticket predictions in `reports/predictions_*.csv`, and
raw numbers in `reports/results.json`. Open the predictions CSVs and read the mistakes.
That is where you learn the most.

## How to decide (the real-world part)

Numbers from this project feed a decision, and accuracy is only one input:

| Factor | Favors fine-tuning | Favors prompting an LLM |
|---|---|---|
| Volume | Millions of requests/month | Low or spiky volume |
| Latency | Tight budgets (milliseconds) | Seconds are acceptable |
| Labelled data | Hundreds+ examples available | Few or none |
| Labels change often | Retraining on every change is a chore | Edit the prompt |
| Data privacy | Data must stay in your infrastructure | A hosted API is allowed |
| Novel phrasing | Only if training data is broad | Usually more forgiving |
| Ops burden | You own hosting, monitoring, retraining | Provider handles it |

The break-even line in the report uses `SELF_HOST_COST_PER_MONTH` (default $150), which is
**an assumption**, so replace it with your actual hosting cost. A common production pattern
combines the two: use a fast, cheap model for the bulk of traffic and send low-confidence
tickets to an LLM. Try building that as an exercise.

## Tests

```bash
pytest -v
```

The tests cover data generation, prompt building, response parsing, the on-disk cache,
metrics, cost math, and the whole comparison flow. The LLM tests use a **fake client**, so they
cost nothing and need no API key or internet. The fine-tuning code needs PyTorch and a
model download, so it is not part of the automated tests. Check it by running `make finetune`.

## Limitations to be aware of

- Synthetic data. Real tickets contain multiple issues, sarcasm, other languages, and
  ambiguous labels. Expect lower scores and more disagreement on real data.
- One test set of 300 tickets: differences of a few points are within noise. Use more
  data or repeated runs before drawing strong conclusions.
- The LLM latency includes your network and depends on API load at that moment.
- Prompt changes matter. A better prompt can change the LLM result a lot, so treat the
  default prompt as a first attempt, not a ceiling.

## Exercises

1. Compare `--shots 0`, `1`, `3` and `8`. Where do returns diminish?
2. Rewrite the label descriptions in `config.py`. How much does the LLM's accuracy move?
3. Fine-tune with `--epochs 1` versus `--epochs 6`. Check the validation curve in
   `models/distilbert-tickets/training_summary.json`.
4. Add a confidence score to the fine-tuned model (softmax probability) and send tickets
   below a threshold to the LLM. Measure accuracy, latency and cost of the combined system.
5. Add a fourth contender: a sentence-embedding model plus logistic regression.
6. Replace the synthetic data with a real public support-ticket dataset and see which
   conclusions survive.
