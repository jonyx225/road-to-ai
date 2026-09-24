# Sentiment API

A small but production-shaped ML service: a scikit-learn sentiment classifier
(TF-IDF + logistic regression) served through a FastAPI REST API, with tests,
Docker, and CI on GitHub Actions.

**What this project teaches:** packaging a model as a service, validating input,
loading the model once at startup, testing an ML API, containerizing it, and
running checks automatically on every push.

## Project structure

```
sentiment-api/
├── app/
│   ├── config.py      # settings (model path, input limits) from env vars
│   ├── model.py       # loads the saved pipeline and runs predictions
│   ├── schemas.py     # request/response models + input validation
│   ├── main.py        # FastAPI app and endpoints
│   └── training.py    # trains the model and saves it to models/
├── data/reviews.csv   # sample training data (text,label)
├── tests/             # pytest suite for the API and the training code
├── Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt / requirements-dev.txt
├── pytest.ini
└── Makefile
```

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m app.training             # trains and saves models/sentiment.joblib
uvicorn app.main:app --reload      # serves on http://127.0.0.1:8000
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI, or use curl:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product, works perfectly!"}'

curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Loved it", "Terrible, it broke in a day"]}'
```

Example response:

```json
{
  "label": "positive",
  "confidence": 0.9814,
  "probabilities": {"negative": 0.0186, "positive": 0.9814}
}
```

## Endpoints

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/health`        | `ok` if the model is loaded, else `degraded`  |
| POST   | `/predict`       | Classify one text                             |
| POST   | `/predict/batch` | Classify up to 32 texts (order is preserved)  |

Invalid input (empty or whitespace-only text, missing field, over 5,000
characters, empty or oversized batch) returns `422`. If the model file is
missing, `/predict` returns `503` instead of crashing.

## Run the tests

```bash
pytest -v
```

The test fixtures train a throwaway model into a temp folder, so the tests never
depend on your local `models/` directory.

## Docker

```bash
docker build -t sentiment-api .
docker run --rm -p 8000:8000 sentiment-api
```

The image trains the model at build time, so it always matches the installed
scikit-learn version, and it runs as a non-root user.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: sentiment API"
git branch -M main
git remote add origin https://github.com/<your-username>/sentiment-api.git
git push -u origin main
```

Or, with the GitHub CLI: `gh repo create sentiment-api --public --source=. --push`

Once pushed, the workflow in `.github/workflows/ci.yml` runs the tests and a
Docker smoke test on every push and pull request.

## Important: the sample data is synthetic

`data/reviews.csv` is a small, partly templated dataset so the project runs
offline and the tests are fast. Its held-out accuracy (~96%) is **inflated**
because the examples are repetitive. Treat it as a placeholder, and use real
data to see honest numbers.

To train on real reviews, produce a CSV with `text` and `label` columns
(labels `positive` / `negative`) and run:

```bash
python -m app.training --data path/to/your_reviews.csv
```

One way to get real data is the IMDB dataset from Hugging Face
(`pip install datasets`):

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))
df = ds.to_pandas()
df["label"] = df["label"].map({0: "negative", 1: "positive"})
df[["text", "label"]].to_csv("data/imdb_5k.csv", index=False)
```

(Shuffle before slicing: the raw IMDB file is sorted by label.)

## Exercises to go further

1. Train on real data and compare the metrics with the synthetic sample.
2. Add a `neutral` class and see how the API and tests need to change.
3. Add a `/model-info` endpoint that returns the training metadata (accuracy, date, classes).
4. Add request logging with a request ID.
5. Try `LinearSVC` or a small transformer and compare accuracy and latency.
