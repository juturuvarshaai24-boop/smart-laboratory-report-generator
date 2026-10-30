"""Leakage-safe model construction and metrics for pump-leakage detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RANDOM_STATE = 42


def build_candidate_models(random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Pipeline]:
    """Return a majority baseline and interpretable regularized classifier."""
    return {
        "majority_dummy": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def calculate_binary_metrics(
    estimator: Any,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate threshold metrics and ROC-AUC when both classes exist."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("decision threshold must be a finite value in [0, 1]")
    truth = _validated_binary_labels(features, labels)
    probabilities = None
    positive_indexes = np.array([], dtype=int)
    if hasattr(estimator, "predict_proba"):
        classes = np.asarray(estimator.classes_)
        positive_indexes = np.flatnonzero(classes == 1)
        if len(positive_indexes) == 1:
            all_probabilities = np.asarray(estimator.predict_proba(features))
            if all_probabilities.shape != (len(features), len(classes)):
                raise ValueError("predicted probabilities do not align with features")
            candidate = all_probabilities[:, int(positive_indexes[0])]
            if np.isfinite(candidate).all() and np.all(
                (candidate >= 0) & (candidate <= 1)
            ):
                probabilities = candidate
    if probabilities is None:
        predictions = np.asarray(estimator.predict(features), dtype=np.int64)
    else:
        predictions = (probabilities >= threshold).astype(np.int64)
    if predictions.shape != truth.shape:
        raise ValueError("predictions do not align with evaluation labels")
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    roc_auc = None
    if probabilities is not None and np.unique(truth).size == 2:
        roc_auc = float(roc_auc_score(truth, probabilities))
    return {
        "cycle_count": len(truth),
        "class_counts": {
            "0": int(np.sum(truth == 0)),
            "1": int(np.sum(truth == 1)),
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "roc_auc": roc_auc,
        "decision_threshold": float(threshold),
    }


def select_probability_threshold(
    estimator: Any,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
) -> float:
    """Choose a probability threshold on validation data only."""
    truth = _validated_binary_labels(features, labels)
    if np.unique(truth).size != 2:
        raise ValueError("threshold selection requires both validation classes")
    if not hasattr(estimator, "predict_proba"):
        return 0.5
    classes = np.asarray(estimator.classes_)
    positive_indexes = np.flatnonzero(classes == 1)
    if len(positive_indexes) != 1:
        return 0.5
    all_probabilities = np.asarray(estimator.predict_proba(features))
    if all_probabilities.shape != (len(features), len(classes)):
        raise ValueError("validation probabilities do not align with features")
    probabilities = all_probabilities[:, int(positive_indexes[0])]
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("validation probabilities must be finite values in [0, 1]")
    unique = np.unique(probabilities)
    midpoints = (unique[:-1] + unique[1:]) / 2
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
    scored = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        scored.append(
            (
                float(balanced_accuracy_score(truth, predictions)),
                float(f1_score(truth, predictions, zero_division=0)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return max(scored)[3]


def _validated_binary_labels(
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
) -> np.ndarray:
    if isinstance(labels, pd.Series) and not features.index.equals(labels.index):
        raise ValueError("feature and label cycle indexes do not match")
    truth = np.asarray(labels)
    if truth.ndim != 1 or len(truth) != len(features):
        raise ValueError("evaluation labels must be one-dimensional and align with features")
    if not np.isfinite(truth).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("evaluation labels must contain only finite binary values 0 and 1")
    return truth.astype(np.int64, copy=False)


def select_candidate(validation_metrics: dict[str, dict[str, Any]]) -> str:
    """Select by validation balanced accuracy, F1, ROC-AUC, then model name."""
    if not validation_metrics:
        raise ValueError("at least one validation result is required")
    return max(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["balanced_accuracy"],
            validation_metrics[name]["f1"],
            (
                validation_metrics[name]["roc_auc"]
                if validation_metrics[name]["roc_auc"] is not None
                else -1.0
            ),
            name == "majority_dummy",
            name == "logistic_regression",
        ),
    )
