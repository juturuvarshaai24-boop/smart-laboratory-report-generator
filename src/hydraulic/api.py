"""FastAPI inference service for hydraulic pump leakage detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import FEATURE_STATISTICS, PHYSICAL_SENSOR_NAMES, feature_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models/hydraulic/comparisons/"
    / "hydraulic-compare-20261021-2104d/random_forest.joblib"
)
MODEL_PATH = Path(os.getenv("HYDRAULIC_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DECISION_THRESHOLD = 0.13628977735799333

FEATURE_NAMES = [
    feature_name(sensor, statistic)
    for sensor in PHYSICAL_SENSOR_NAMES
    for statistic in FEATURE_STATISTICS
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="The 112 engineered hydraulic sensor features.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    leakage_probability: float
    decision_threshold: float


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load and validate the saved estimator."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    estimator = joblib.load(path)

    if getattr(estimator, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("Saved model does not match the 112-feature schema.")

    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Saved model does not support probability predictions.")

    if list(getattr(estimator, "feature_names_in_", [])) != FEATURE_NAMES:
        raise ValueError("Saved model feature order does not match the API schema.")

    return estimator


app = FastAPI(
    title="Hydraulic Pump Leakage Detection API",
    version="1.0.0",
    description="Predicts pump leakage from 112 engineered hydraulic sensor features.",
)

try:
    model = load_model()
    model_error: str | None = None
except (FileNotFoundError, ValueError, OSError) as exc:
    model = None
    model_error = str(exc)


@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {model_error}")
    return {"status": "healthy", "model_status": "loaded"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model": "RandomForestClassifier",
        "feature_count": len(FEATURE_NAMES),
        "decision_threshold": DECISION_THRESHOLD,
        "positive_class": "pump leakage present",
        "negative_class": "no pump leakage",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {model_error}")

    supplied = set(request.features)
    expected = set(FEATURE_NAMES)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )

    try:
        values = pd.DataFrame(
            [[request.features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=422, detail="All feature values must be numeric.")

    if not np.isfinite(values.to_numpy()).all():
        raise HTTPException(status_code=422, detail="Feature values must be finite.")

    probability = float(model.predict_proba(values)[0, 1])
    prediction = int(probability >= DECISION_THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        result="leakage_detected" if prediction else "no_leakage_detected",
        leakage_probability=probability,
        decision_threshold=DECISION_THRESHOLD,
    )
