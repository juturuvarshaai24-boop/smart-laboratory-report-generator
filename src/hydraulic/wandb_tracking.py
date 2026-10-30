
"""Optional Weights & Biases tracking for saved hydraulic comparisons.

MLflow remains the primary tracker. This module logs existing comparison
results and never trains models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import wandb

from src.hydraulic.artifacts import validate_run_id
from src.hydraulic.data import ARCHIVE_SHA256
from src.hydraulic.mlflow_tracking import (
    DATASET_DOI,
    DATASET_NAME,
    DATASET_TAG,
    DATASET_URL,
    EXPERIMENT_NAME,
    TASK_TAG,
)


def log_comparison_to_wandb(
    project_root: Path,
    comparison_id: str,
    *,
    project: str = "hydraulic-condition-monitoring",
    entity: str | None = None,
) -> dict[str, str]:
    """Log saved model comparison metrics and metadata to W&B.

    Requires a W&B API login or WANDB_API_KEY environment variable.
    This function does not fit or retrain any model.
    """
    safe_id = validate_run_id(comparison_id)
    project_root = Path(project_root)

    metrics_path = (
        project_root
        / "metrics"
        / "hydraulic"
        / "comparisons"
        / safe_id
        / "comparison.json"
    )
    metadata_path = (
        project_root
        / "models"
        / "hydraulic"
        / "comparisons"
        / safe_id
        / "comparison_metadata.json"
    )

    for path in (metrics_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison file is missing: {path}")

    results: dict[str, Any] = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    metadata: dict[str, Any] = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if (
        results.get("comparison_id") != safe_id
        or metadata.get("comparison_id") != safe_id
    ):
        raise ValueError("Comparison ID does not match saved metadata")

    if (
        results.get("dataset_doi") != DATASET_DOI
        or metadata.get("dataset_doi") != DATASET_DOI
        or results.get("archive_sha256") != ARCHIVE_SHA256
        or metadata.get("archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("Dataset provenance validation failed")

    if not os.environ.get("WANDB_API_KEY") and not wandb.api.api_key:
        # W&B may already be authenticated through `wandb login`.
        raise RuntimeError(
                "W&B is not authenticated. Run `wandb login` in your terminal."
            )

    run_ids: dict[str, str] = {}
    candidates = results.get("candidate_results", {})
    versions = metadata.get("model_versions", {})

    if not candidates or set(candidates) != set(versions):
        raise ValueError("Candidate results and model metadata do not agree")

    for family, candidate in candidates.items():
        run = wandb.init(
            project=project,
            entity=entity,
            name=f"{safe_id}-{family}",
            group=safe_id,
            job_type="model-comparison",
            tags=[
                DATASET_TAG,
                TASK_TAG,
                "model_comparison",
                "chronological",
                family,
            ],
            config={
                "comparison_id": safe_id,
                "experiment": EXPERIMENT_NAME,
                "dataset": DATASET_NAME,
                "dataset_url": DATASET_URL,
                "dataset_doi": DATASET_DOI,
                "dataset_archive_sha256": ARCHIVE_SHA256,
                "model_family": versions[family]["family"],
                "model_parameters": versions[family]["parameters"],
                "random_state": versions[family]["random_state"],
                "feature_count": results.get("feature_count"),
                "split_method": results.get("split_method"),
                "validation_method": results.get("validation_method"),
                "selected_candidate": results.get("selected_candidate") == family,
                "test_used_for_selection": results.get("test_used_for_selection"),
            },
            reinit=True,
        )

        try:
            logged_metrics: dict[str, float] = {}

            for split_name in ("training_in_sample", "validation", "test"):
                split = candidate.get(split_name, {})
                for threshold_name, metric_set in split.items():
                    if not isinstance(metric_set, dict):
                        continue
                    for metric_name in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                    ):
                        value = metric_set.get(metric_name)
                        if isinstance(value, (int, float)):
                            logged_metrics[
                                f"{split_name}.{threshold_name}.{metric_name}"
                            ] = float(value)

            blocked = candidate.get("blocked_training_validation", {})
            for metric_name in (
                "mean_balanced_accuracy",
                "std_balanced_accuracy",
                "minimum_balanced_accuracy",
            ):
                value = blocked.get(metric_name)
                if isinstance(value, (int, float)):
                    logged_metrics[f"training_blocked_cv.{metric_name}"] = float(value)

            if logged_metrics:
                run.log(logged_metrics)

            run.summary["comparison_id"] = safe_id
            run.summary["selected_candidate"] = (
                results.get("selected_candidate") == family
            )
            run.summary["dataset_archive_sha256"] = ARCHIVE_SHA256

            run_ids[family] = run.id
        finally:
            run.finish()

    return run_ids
