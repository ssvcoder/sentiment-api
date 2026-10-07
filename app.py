#!/usr/bin/env python3
"""FastAPI service for sentiment classification.

Loads the scikit-learn pipeline saved by train.py (model.pkl) and exposes:
    GET  /health          - liveness check (includes model load status)
    POST /predict         - classify a single text
    POST /predict/batch   - classify up to 32 texts in one call

Run locally:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.pkl"))
MAX_TEXT_LEN = 5000
MAX_BATCH = 32

_state: dict[str, Any] = {"pipeline": None, "labels": {0: "negative", 1: "positive"}}


def load_model() -> None:
    """Load the trained pipeline into process state; raise RuntimeError if missing."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python train.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    _state["pipeline"] = bundle["pipeline"]
    _state["labels"] = bundle.get("labels", {0: "negative", 1: "positive"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast at startup if the model is missing so misconfigurations
    # surface immediately instead of as 503s on first request.
    load_model()
    yield


app = FastAPI(
    title="Sentiment API",
    description="Binary sentiment classification (positive/negative) over short English text.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN,
                      description="Text to classify (1-5000 chars).")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             description="Up to 32 texts to classify.")


class Prediction(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _predict_one(text: str) -> Prediction:
    pipeline = _state["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    proba = pipeline.predict_proba([cleaned])[0]
    classes = list(pipeline.classes_)
    best_idx = int(proba.argmax())
    label_id = int(classes[best_idx])
    probs = {
        _state["labels"].get(int(c), str(c)): round(float(p), 4)
        for c, p in zip(classes, proba)
    }
    return Prediction(
        text=text,
        label=_state["labels"].get(label_id, str(label_id)),
        confidence=round(float(proba[best_idx]), 4),
        probabilities=probs,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["pipeline"] is not None else "degraded",
        model_loaded=_state["pipeline"] is not None,
    )


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(req: PredictRequest) -> Prediction:
    """Classify a single piece of text as positive or negative."""
    return _predict_one(req.text)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Classify up to 32 texts in a single request."""
    for t in req.texts:
        if len(t) > MAX_TEXT_LEN:
            raise HTTPException(
                status_code=422,
                detail=f"Each text must be at most {MAX_TEXT_LEN} characters.",
            )
    preds = [_predict_one(t) for t in req.texts]
    return BatchResponse(predictions=preds, count=len(preds))
