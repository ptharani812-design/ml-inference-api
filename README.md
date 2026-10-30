# Real-Time ML Inference API

A production-style REST API that serves a **champion machine-learning model** with FastAPI,
packaged in Docker, covered by tests, and documented end to end.

The demo model classifies **Iris flowers**. A small tournament (`training/train.py`) trains four
candidate models, picks the best by cross-validation, and ships only the winner.
Open `http://localhost:8000/` for the **Iris Field Lab**, a live page where you drag flower
measurements and watch the model's probabilities move in real time.

## Architecture

```
 training/train.py                    models/                         app/ (FastAPI)                 Docker
┌──────────────────────┐   saves   ┌───────────────────┐   loads    ┌───────────────────┐   ┌─────────────────┐
│ 4 candidate models   │ ────────▶ │ model.joblib      │ ─────────▶ │ ModelService      │   │ python:3.11-slim│
│ 5-fold CV tournament │           │ model_meta.json   │  at start  │ Pydantic schemas  │──▶│ non-root user   │
│ champion + metrics   │           │ (features, classes│            │ /predict /health  │   │ HEALTHCHECK     │
└──────────────────────┘           │  metrics, ranking)│            │ /model-info /docs │   │ uvicorn :8000   │
                                   └───────────────────┘            └───────────────────┘   └─────────────────┘
```

Design decisions:

- **Model loaded once** at process start, never per request.
- **Schema-driven validation:** the request model is generated from `model_meta.json`, so a new model
  with different features needs no code changes. Unknown fields are rejected (`extra="forbid"`), and NaN/Infinity are refused.
- **Pinned dependencies** in `requirements.txt`. The scikit-learn version used to train must match the one that serves.
- **Container hygiene:** layer-cached installs, non-root user, built-in health check.

## Endpoints

| Method | Path             | Purpose                                              |
|--------|------------------|------------------------------------------------------|
| POST   | `/predict`       | One prediction with class probabilities              |
| POST   | `/predict/batch` | 1 to 100 predictions in one request                  |
| GET    | `/health`        | Liveness probe (used by Docker `HEALTHCHECK`)        |
| GET    | `/model-info`    | Features, classes, metrics, tournament leaderboard   |
| GET    | `/docs`          | Interactive Swagger UI (auto-generated)              |
| GET    | `/`              | Iris Field Lab playground                            |

### Example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm": 6.0, "sepal_width_cm": 2.8, "petal_length_cm": 4.3, "petal_width_cm": 1.3}'
```

```json
{
  "prediction": "versicolor",
  "prediction_index": 1,
  "confidence": 0.9,
  "probabilities": {"setosa": 0.02, "versicolor": 0.9, "virginica": 0.08},
  "model_version": "1.0.0-logistic_regression",
  "latency_ms": 1.2
}
```
(Values above are illustrative; your numbers will differ slightly.)

### Status codes

| Code | When                                                                  |
|------|-----------------------------------------------------------------------|
| 200  | Valid request                                                         |
| 405  | Wrong HTTP method (e.g. `GET /predict`)                               |
| 404  | Unknown route                                                         |
| 422  | Missing field, wrong type, null, unknown field, bad JSON, empty/oversized batch |

## Run it

### Docker

```bash
docker build -t ml-inference-api .
docker run --rm -p 8000:8000 ml-inference-api
# open http://localhost:8000  (playground)  or  http://localhost:8000/docs
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python training/train.py          # trains the champion, writes ./models
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -v
```

22 tests cover: response schema, probabilities summing to 1, confidence matching the top class,
known flowers per species, determinism, every 422 path (missing, wrong type, null, unknown field,
malformed JSON, empty body), 404/405, batch limits, the OpenAPI contract, and `/health`.
GitHub Actions (`.github/workflows/ci.yml`) runs the tests, builds the image, and smoke-tests the container.

## Model performance

Run `python training/train.py` to reproduce. Results are stored in `models/model_meta.json`
and served at `GET /model-info`.

| Model               | 5-fold CV accuracy |
|---------------------|--------------------|
| logistic_regression | 0.965 (champion)   |
| svm_rbf             | 0.965              |
| random_forest       | 0.964              |
| gradient_boosting   | 0.956              |

Held-out test (38 samples): accuracy 0.921, macro F1 0.923. The shipped model is then refit on all 150 samples.

## Use your own model

1. Edit `load_data()` in `training/train.py` to return your data, feature names and class names.
2. Add or remove candidates in `candidates()`. Any estimator with `predict_proba` works.
3. Run `python training/train.py`, then `pytest`. Update the sample values in the `curl` example and the tests that name Iris features.

## Project layout

```
app/            FastAPI service (main, schemas, model loader, playground page)
training/       Champion tournament script
models/         model.joblib + model_meta.json (committed so Docker can copy them)
tests/          pytest suite
Dockerfile      Slim, non-root, health-checked image
```
