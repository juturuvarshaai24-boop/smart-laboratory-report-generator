"""MLflow tracking for audited hydraulic model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sqlalchemy.exc import SQLAlchemyError

from .artifacts import validate_run_id
from .data import (
    ARCHIVE_SHA256,
    DATASET_DOI,
    DATASET_NAME,
    DATASET_URL,
    FEATURE_RECIPE_VERSION,
)

EXPERIMENT_NAME = "hydraulic-condition-monitoring"
DATASET_TAG = "uci-hydraulic-condition-monitoring"
TASK_TAG = "binary-pump-leakage-classification"
DEFAULT_TRACKING_ENV = "MLFLOW_TRACKING_URI"
REGISTERED_MODEL_NAME = "hydraulic-pump-leakage-classifier"


def resolve_tracking_uri(
    project_root: Path,
    tracking_uri: str | None = None,
) -> str:
    """Resolve explicit, environment, then repository-local SQLite tracking."""
    if tracking_uri:
        return tracking_uri
    configured = os.environ.get(DEFAULT_TRACKING_ENV)
    if configured:
        return configured
    return f"sqlite:///{(project_root / 'mlflow.db').resolve()}"


def _get_or_create_experiment(
    client: MlflowClient,
    project_root: Path,
) -> str:
    experiments = client.search_experiments(view_type=ViewType.ALL)
    for experiment in experiments:
        if experiment.name == EXPERIMENT_NAME:
            if experiment.lifecycle_stage != "active":
                raise RuntimeError(
                    f"MLflow experiment {EXPERIMENT_NAME!r} exists but is "
                    "deleted; restore it explicitly before logging"
                )
            return experiment.experiment_id

    artifact_location = (
        project_root / "mlruns" / EXPERIMENT_NAME
    ).resolve().as_uri()

    return client.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=artifact_location,
    )


def _read_comparison(
    project_root: Path,
    comparison_id: str,
) -> tuple[dict, dict, Path, Path]:
    safe_id = validate_run_id(comparison_id)

    model_dir = (
        project_root / "models" / "hydraulic" / "comparisons" / safe_id
    )
    metrics_dir = (
        project_root / "metrics" / "hydraulic" / "comparisons" / safe_id
    )

    metrics_path = metrics_dir / "comparison.json"
    metadata_path = model_dir / "comparison_metadata.json"
    predictions_path = metrics_dir / "test_predictions.csv"

    required_files = [
        metrics_path,
        metadata_path,
        predictions_path,
    ]

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"comparison model directory is missing: {model_dir}"
        )

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"required comparison artifact is missing: {path}"
            )

    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if (
        results.get("comparison_id") != safe_id
        or metadata.get("comparison_id") != safe_id
    ):
        raise ValueError(
            "comparison ID does not match its saved provenance metadata"
        )

    checksums = {
        results.get("archive_sha256"),
        metadata.get("archive_sha256"),
        ARCHIVE_SHA256,
    }

    if len(checksums) != 1:
        raise ValueError(
            "comparison archive checksum does not match the audited dataset"
        )

    if (
        results.get("dataset_doi") != DATASET_DOI
        or metadata.get("dataset_doi") != DATASET_DOI
    ):
        raise ValueError(
            "comparison dataset DOI does not match hydraulic provenance"
        )

    if results.get("dataset") != DATASET_NAME:
        raise ValueError(
            "comparison dataset name does not match hydraulic provenance"
        )

    candidates = results.get("candidate_results", {})
    versions = metadata.get("model_versions", {})

    if not candidates or set(candidates) != set(versions):
        raise ValueError(
            "candidate metrics and model metadata do not agree"
        )

    if results.get("selected_candidate") not in candidates:
        raise ValueError(
            "selected candidate is absent from comparison results"
        )

    if results.get("feature_count") != 112:
        raise ValueError(
            "comparison does not use the audited 112-feature model schema"
        )

    if results.get("test_used_for_selection") is not False:
        raise ValueError(
            "comparison metadata does not confirm an untouched test split"
        )

    for candidate in candidates:
        model_path = model_dir / f"{candidate}.joblib"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"saved candidate estimator is missing: {model_path}"
            )

        estimator = joblib.load(model_path)

        if (
            getattr(estimator, "n_features_in_", None)
            != results["feature_count"]
        ):
            raise ValueError(
                f"saved {candidate} estimator does not use "
                "the audited feature count"
            )

        classifier = (
            estimator.named_steps.get("classifier")
            if isinstance(estimator, Pipeline)
            else estimator
        )

        expected_types = {
            "majority_dummy": DummyClassifier,
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "hist_gradient_boosting": HistGradientBoostingClassifier,
        }

        expected_type = expected_types.get(candidate)

        if expected_type is None or not isinstance(
            classifier, expected_type
        ):
            raise ValueError(
                f"saved estimator does not match candidate family {candidate}"
            )

        actual_parameters = classifier.get_params(deep=False)

        for name, expected_value in versions[candidate]["parameters"].items():
            if (
                name not in actual_parameters
                or actual_parameters[name] != expected_value
            ):
                raise ValueError(
                    f"saved {candidate} estimator parameter {name!r} "
                    "differs from comparison metadata"
                )

    return results, metadata, model_dir, metrics_dir


def _metric_values(
    metrics: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    values: dict[str, float] = {}

    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        value = metrics.get(name)

        if value is not None:
            numeric = float(value)

            if not math.isfinite(numeric):
                raise ValueError(f"non-finite {prefix}.{name} metric")

            values[f"{prefix}.{name}"] = numeric

    confusion = metrics.get("confusion_matrix")

    if confusion is not None:
        if len(confusion) != 2 or any(len(row) != 2 for row in confusion):
            raise ValueError(f"invalid {prefix} confusion matrix")

        for row_name, row in zip(
            ("negative", "positive"),
            confusion,
        ):
            for column_name, value in zip(
                ("negative", "positive"),
                row,
            ):
                values[
                    f"{prefix}.confusion_matrix."
                    f"{row_name}_{column_name}"
                ] = float(value)

    return values


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() if result.returncode == 0 else None


def _log_candidate(
    *,
    client: MlflowClient,
    experiment_id: str,
    project_root: Path,
    comparison_id: str,
    family: str,
    results: dict[str, Any],
    metadata: dict[str, Any],
    model_dir: Path,
    metrics_dir: Path,
) -> str:
    matching = client.search_runs(
        [experiment_id],
        filter_string=f"tags.comparison_id = '{comparison_id}'",
        max_results=100,
    )

    existing = [
        run for run in matching
        if (
            run.data.tags.get("candidate") == family
            and run.data.tags.get("superseded_by_retry") != "true"
        )
    ]

    if len(existing) > 1:
        raise RuntimeError(
            f"duplicate MLflow records already exist for "
            f"{comparison_id}/{family}"
        )

    if existing:
        run = existing[0]

        if run.info.status == "FAILED":
            # Preserve the failed attempt for audit, but exclude it from
            # future idempotency lookups and create a clean replacement run.
            client.set_tag(run.info.run_id, "superseded_by_retry", "true")
            client.set_tag(
                run.info.run_id,
                "retry_note",
                "Incomplete native MLflow model; retained raw joblib and comparison artifacts",
            )
            existing = []
        elif run.info.status == "FINISHED":
            # Older runs may contain only a raw .joblib file. Repair those
            # runs by adding the standard MLflow sklearn model directory.
            model_artifacts = client.list_artifacts(run.info.run_id, "model")
            has_mlflow_model = any(
                artifact.path == "model/MLmodel"
                for artifact in model_artifacts
            )
            if not has_mlflow_model:
                estimator = joblib.load(model_dir / f"{family}.joblib")
                with mlflow.start_run(run_id=run.info.run_id):
                    mlflow_sklearn.log_model(
                        sk_model=estimator,
                        artifact_path="model",
                        # The local, audited tree-based estimators were trained
                        # by this project. Trust only sklearn's tree storage type.
                        skops_trusted_types=[
                            "sklearn.tree._tree.Tree",
                            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                        ],
                    )
                    mlflow.log_artifact(
                        str(model_dir / f"{family}.joblib"),
                        artifact_path="joblib",
                    )
            return run.info.run_id

        elif run.info.status != "FAILED":
            raise RuntimeError(
                f"existing {run.info.status} run {run.info.run_id} for "
                f"{comparison_id}/{family}; inspect it before retrying"
            )

    candidate = results["candidate_results"][family]
    model_metadata = metadata["model_versions"][family]
    distributions = results["class_distributions"]
    validation = candidate["validation"]
    test = candidate["test"]

    tags = {
        "dataset": DATASET_TAG,
        "dataset_name": DATASET_NAME,
        "dataset_source_url": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "dataset_archive_sha256": ARCHIVE_SHA256,
        "dataset_version": f"archive-sha256:{ARCHIVE_SHA256}",
        "task": TASK_TAG,
        "stage": "model_comparison",
        "evaluation": "chronological",
        "comparison_id": comparison_id,
        "candidate": family,
        "model_family": model_metadata["family"],
        "selected_candidate": str(
            results["selected_candidate"] == family
        ).lower(),
        "positive_class": (
            "1=pump leakage present; original pump grades 1 or 2"
        ),
        "test_used_for_selection": "false",
        "probability_interpretation": (
            "uncalibrated model scores, not calibrated confidence"
        ),
        "validation_method": results["validation_method"],
        "limitations": (
            "Sequential cycles; shifted cooler and accumulator-pressure "
            "conditions; no independent run/rig identifiers; "
            "retrospective single-rig evidence only."
        ),
    }

    commit = _git_commit(project_root)

    if commit:
        tags["git_commit"] = commit

    params = {
        "model_family": family,
        "model_parameters": json.dumps(
            model_metadata["parameters"],
            sort_keys=True,
        ),
        "random_state": str(model_metadata["random_state"]),
        "feature_recipe": FEATURE_RECIPE_VERSION,
        "feature_count": str(results["feature_count"]),
        "dataset_name": DATASET_NAME,
        "dataset_source_url": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "dataset_archive_sha256": ARCHIVE_SHA256,
        "target_definition": (
            "0=no pump leakage; 1=pump leakage present (grades 1 or 2)"
        ),
        "split_method": results["split_method"],
        "embargo_cycles": "60",
        "boundary_purge_cycles_per_side": "60",
        "validation_method": results["validation_method"],
        "validation_selected_threshold": str(
            validation["validation_tuned_threshold"]
        ),
    }

    for split_name in ("train", "validation", "test"):
        split = distributions[split_name]
        start, stop = split["cycle_id_range"]

        params[f"{split_name}_cycle_id_range"] = f"{start}-{stop}"
        params[f"{split_name}_cycle_count"] = str(split["sample_count"])

        for label, count in split["binary_class_counts"].items():
            params[f"{split_name}_class_{label}_count"] = str(count)

    for name, value in model_metadata["parameters"].items():
        params[f"model.{name}"] = str(value)

    model_path = model_dir / f"{family}.joblib"
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    tags["model_artifact_sha256"] = model_digest
    run_name = f"{comparison_id}-{family}"

    try:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        ) as active_run:
            mlflow.log_params(params)

            mlflow.log_metrics(
                _metric_values(
                    candidate["training_in_sample"]["threshold_0_5"],
                    "train.in_sample.threshold_0_5",
                )
            )

            mlflow.log_metrics(
                _metric_values(
                    validation["threshold_0_5"],
                    "validation.threshold_0_5",
                )
            )

            mlflow.log_metrics(
                _metric_values(
                    validation["validation_tuned_metrics"],
                    "validation.validation_selected",
                )
            )

            mlflow.log_metrics(
                _metric_values(
                    test["threshold_0_5"],
                    "test.threshold_0_5",
                )
            )

            mlflow.log_metrics(
                _metric_values(
                    test["validation_tuned_threshold"],
                    "test.validation_selected",
                )
            )

            blocked = candidate.get("blocked_training_validation")

            if blocked:
                mlflow.log_metrics(
                    {
                        "training_blocked_cv.mean_balanced_accuracy": float(
                            blocked["mean_balanced_accuracy"]
                        ),
                        "training_blocked_cv.std_balanced_accuracy": float(
                            blocked["std_balanced_accuracy"]
                        ),
                        "training_blocked_cv.minimum_balanced_accuracy": float(
                            blocked["minimum_balanced_accuracy"]
                        ),
                    }
                )

            # Log a native MLflow sklearn model so the registered
            # runs:/.../model URI contains MLmodel and can be loaded with
            # mlflow.sklearn.load_model().
            estimator = joblib.load(model_path)
            mlflow_sklearn.log_model(
                sk_model=estimator,
                artifact_path="model",
                # The local, audited tree-based estimators were trained
                # by this project. Trust only sklearn's tree storage type.
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                ],
            )
            # Retain the original audited .joblib artifact and its path.
            mlflow.log_artifact(
                str(model_path),
                artifact_path="joblib",
            )

            mlflow.log_artifact(
                str(metrics_dir / "comparison.json"),
                artifact_path="comparison",
            )

            mlflow.log_artifact(
                str(metrics_dir / "test_predictions.csv"),
                artifact_path="comparison",
            )

            mlflow.log_artifact(
                str(model_dir / "comparison_metadata.json"),
                artifact_path="comparison",
            )

            return active_run.info.run_id

    except (MlflowException, SQLAlchemyError, OSError) as exc:
        raise RuntimeError(
            f"MLflow logging failed for {comparison_id}/{family} using "
            f"{resolve_tracking_uri(project_root)!r}: {exc}"
        ) from exc


def log_comparison(
    project_root: Path,
    comparison_id: str,
    *,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Log saved comparison candidates, skipping finished candidate records."""
    safe_id = validate_run_id(comparison_id)
    uri = resolve_tracking_uri(project_root, tracking_uri)

    results, metadata, model_dir, metrics_dir = _read_comparison(
        project_root,
        safe_id,
    )

    mlflow.set_tracking_uri(uri)

    try:
        client = MlflowClient(tracking_uri=uri)

        experiment_id = _get_or_create_experiment(
            client,
            project_root,
        )

        return {
            family: _log_candidate(
                client=client,
                experiment_id=experiment_id,
                project_root=project_root,
                comparison_id=safe_id,
                family=family,
                results=results,
                metadata=metadata,
                model_dir=model_dir,
                metrics_dir=metrics_dir,
            )
            for family in results["candidate_results"]
        }

    except (MlflowException, SQLAlchemyError, OSError) as exc:
        raise RuntimeError(
            f"MLflow tracking store {uri!r} is unavailable or failed: {exc}"
        ) from exc


def register_selected_model(
    project_root: Path,
    comparison_id: str,
    *,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Register the selected comparison candidate in MLflow Model Registry."""
    safe_id = validate_run_id(comparison_id)
    uri = resolve_tracking_uri(project_root, tracking_uri)

    results, metadata, model_dir, _ = _read_comparison(
        project_root,
        safe_id,
    )

    selected = results["selected_candidate"]

    if metadata.get("selected_candidate") != selected:
        raise ValueError(
            "Selected candidate differs between comparison files"
        )

    model_path = model_dir / f"{selected}.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Selected model is missing: {model_path}"
        )

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            "Hydraulic MLflow experiment does not exist. "
            "Log the comparison first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.comparison_id = '{safe_id}'",
        max_results=100,
    )

    selected_runs = [
        run for run in runs
        if run.data.tags.get("candidate") == selected
        and run.data.tags.get("selected_candidate") == "true"
    ]

    if len(selected_runs) != 1:
        raise RuntimeError(
            f"Expected exactly one selected {selected} run; "
            f"found {len(selected_runs)}"
        )

    run = selected_runs[0]

    if run.info.status != "FINISHED":
        raise RuntimeError("Selected MLflow run is not FINISHED")

    if run.data.tags.get("model_artifact_sha256") != digest:
        raise RuntimeError(
            "Model SHA-256 does not match the logged MLflow run"
        )

    model_artifacts = client.list_artifacts(run.info.run_id, "model")
    joblib_artifacts = client.list_artifacts(run.info.run_id, "joblib")

    available_paths = {
        item.path for item in [*model_artifacts, *joblib_artifacts]
    }

    expected_paths = {
        f"model/{selected}.joblib",
        f"joblib/{selected}.joblib",
    }

    if not (available_paths & expected_paths):
        raise RuntimeError("Selected Joblib model artifact is missing from MLflow")

    if "model/MLmodel" not in available_paths:
        raise RuntimeError("Selected run has no MLflow MLmodel descriptor.")

    registered_models = client.search_registered_models()

    registered = next(
        (
            model for model in registered_models
            if model.name == REGISTERED_MODEL_NAME
        ),
        None,
    )

    if registered is None:
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description=(
                "Random Forest classifier for binary hydraulic "
                "pump leakage detection."
            ),
        )

    existing_versions = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )

    for version in existing_versions:
        if version.run_id == run.info.run_id:
            if version.source != f"runs:/{run.info.run_id}/model":
                raise RuntimeError(
                    "Existing registered version has an unexpected source"
                )

            existing_digest = version.tags.get("model_artifact_sha256")

            if existing_digest and existing_digest != digest:
                raise RuntimeError(
                    "Existing registered version has a different SHA-256"
                )

            if not existing_digest:
                client.set_model_version_tag(
                    REGISTERED_MODEL_NAME,
                    version.version,
                    "model_artifact_sha256",
                    digest,
                )

            return {
                "name": REGISTERED_MODEL_NAME,
                "version": str(version.version),
                "status": version.status,
                "run_id": run.info.run_id,
            }

    version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"runs:/{run.info.run_id}/model",
        run_id=run.info.run_id,
        tags={
            "model_family": selected,
            "comparison_id": safe_id,
            "dataset_doi": DATASET_DOI,
            "model_artifact_sha256": digest,
            "selection_status": "selected_candidate",
        },
    )

    return {
        "name": REGISTERED_MODEL_NAME,
        "version": str(version.version),
        "status": version.status,
        "run_id": run.info.run_id,
    }



def load_registered_model(
    project_root: Path,
    version: str,
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
) -> Any:
    """Load a registered MLflow model, including nested artifact layouts."""
    uri = resolve_tracking_uri(project_root, tracking_uri)
    mlflow.set_tracking_uri(uri)

    client = MlflowClient(tracking_uri=uri)
    registered_version = client.get_model_version(model_name, str(version))

    if registered_version.status != "READY":
        raise RuntimeError(
            f"Registered model {model_name!r} version {version} "
            f"is not READY: {registered_version.status}"
        )

    model_uri = f"models:/{model_name}/{version}"
    downloaded_path = Path(
        mlflow.artifacts.download_artifacts(model_uri)
    )

    candidates = [
        downloaded_path,
        downloaded_path / "model",
    ]

    model_path = next(
        (candidate for candidate in candidates
         if (candidate / "MLmodel").is_file()),
        None,
    )

    if model_path is None:
        raise FileNotFoundError(
            f"No MLmodel descriptor found in downloaded registry artifact: "
            f"{downloaded_path}"
        )

    return mlflow.pyfunc.load_model(str(model_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--register-selected",
        action="store_true",
        help="Register the selected comparison candidate in MLflow",
    )

    args = parser.parse_args()

    project_root = (
        args.project_root
        or Path(__file__).resolve().parents[2]
    )

    run_ids = log_comparison(
        project_root,
        args.comparison_id,
        tracking_uri=args.tracking_uri,
    )

    output = {
        "experiment": EXPERIMENT_NAME,
        "run_ids": run_ids,
    }

    if args.register_selected:
        output["registered_model"] = register_selected_model(
            project_root,
            args.comparison_id,
            tracking_uri=args.tracking_uri,
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()