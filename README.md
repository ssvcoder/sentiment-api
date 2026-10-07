# Sentiment API

A text sentiment classification service: a scikit-learn NLP pipeline served
behind a FastAPI REST API, with a small browser demo UI and a Dockerfile.

Given a product or movie review, it predicts **positive** or **negative**
sentiment and returns a confidence score.

## Tech stack

- **ML:** scikit-learn (TF-IDF + Logistic Regression), joblib
- **API:** FastAPI, pydantic, uvicorn
- **Demo UI:** plain HTML + JavaScript (no build step)
- **Packaging:** Docker

## How it works

1. **Preprocessing** — text is lowercased and tokenized by scikit-learn's
   `TfidfVectorizer`; English stop words are removed.
2. **Featurization** — word unigrams + bigrams (up to 5,000 features) are
   weighted with sublinear TF-IDF.
3. **Classification** — a `LogisticRegression` model (balanced class weights)
   outputs `P(positive)` / `P(negative)`; the higher one wins.
4. **Serving** — the fitted pipeline is serialized with joblib to `model.pkl`
   and loaded once at API startup; `/predict` returns the label, confidence,
   and full probability distribution.

## Training

```bash
pip install -r requirements.txt
python train.py
```

This trains on the 144 hand-labeled reviews in `train.py`, prints train/test
accuracy, a classification report, a confusion matrix, and the most indicative
tokens per class, then saves `model.pkl`:

```
Dataset: 144 examples (72 positive, 72 negative)
Train: 115  Test: 29 (stratified split, seed=42)

Train accuracy: 0.991
Test  accuracy: 0.862
...
Top positive indicators: perfect, amazing, excellent, wonderful, love, ...
Top negative indicators: regret, terrible, rude, worst, overpriced, ...
Saved model to /path/to/model.pkl
```

Options: `python train.py --test-size 0.2 --seed 42 --model-out model.pkl`

## Running the API

```bash
uvicorn app:app --reload --port 8000
```

The model must exist first (`python train.py`) — the service fails fast at
startup with a clear error if `model.pkl` is missing.

## API docs

Interactive docs: `http://localhost:8000/docs`

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### `POST /predict` — classify one text

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was fantastic, I loved every minute of it!"}'
```

```json
{
  "text": "The movie was fantastic, I loved every minute of it!",
  "label": "positive",
  "confidence": 0.9234,
  "probabilities": {"negative": 0.0766, "positive": 0.9234}
}
```

### `POST /predict/batch` — classify up to 32 texts

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I loved it!", "Terrible, boring, awful."]}'
```

```json
{
  "predictions": [
    {"text": "I loved it!", "label": "positive", "confidence": 0.81, "probabilities": {"negative": 0.19, "positive": 0.81}},
    {"text": "Terrible, boring, awful.", "label": "negative", "confidence": 0.93, "probabilities": {"negative": 0.93, "positive": 0.07}}
  ],
  "count": 2
}
```

Validation: empty/blank text → `422`; batch over 32 items or any text over
5000 chars → `422`; missing model → `503` (or startup failure).

## Demo UI

Open `ui/index.html` in a browser while the API runs on `http://localhost:8000`
(the base URL is editable in the page). Type a review, hit **Analyze sentiment**,
and see the label plus a confidence bar. `Ctrl/Cmd+Enter` submits.

## Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

The image trains the model during `docker build`, so the container serves
predictions immediately. API at `http://localhost:8000`, docs at `/docs`.

## Project layout

```
sentiment-api/
├── train.py           # dataset + training pipeline, saves model.pkl
├── app.py             # FastAPI service (/health, /predict, /predict/batch)
├── ui/index.html      # demo single-page UI
├── requirements.txt   # pinned dependencies
├── Dockerfile         # trains + serves the model
└── README.md
```

## Limitations (honest notes)

- **Tiny dataset** — 144 hand-written examples. Great for demonstrating the
  pipeline end to end; not a production-grade corpus. ~0.86 test accuracy on 29
  held-out examples is encouraging but noisy — expect variance across splits.
- **Binary, English only** — no neutral/mixed class, no other languages.
- **Bag-of-words blind spots** — negation ("not good"), sarcasm ("oh great,
  another delay"), and domain shifts (e.g. medical text) will fool it.
- **Short texts** — tuned for review-length snippets; very long documents are
  truncated at 5000 chars per request.
- Natural next steps: a larger labeled corpus (e.g. IMDb/SST-2), a neutral
  class, or fine-tuning a transformer for harder cases.
