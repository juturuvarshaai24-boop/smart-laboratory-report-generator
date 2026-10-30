"""Controlled, cycle-aware model comparison for hydraulic leakage detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import validate_run_id
from .data import DATASET_DOI, DATASET_NAME, DATASET_URL, load_hydraulic_dataset
from .evaluate import select_evaluation_partition
from .mlflow_tracking import log_comparison
from .modeling import (
    DEFAULT_RANDOM_STATE,
    build_candidate_models,
    calculate_binary_metrics,
    select_probability_threshold,
)
from .splitting import (
    CycleSplits,
    chronological_splits,
    validate_no_cross_split_duplicates,
)

BLOCKED_FOLDS: tuple[tuple[int, int], ...] = (
    (400, 700),
    (700, 950),
    (950, 1200),
    (1200, 1483),
)
FOLD_EMBARGO_CYCLES = 60


def comparison_pipeline(
    family: str,
    parameters: dict[str, Any],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create one named candidate with preprocessing scoped inside its pipeline."""
    if family == "majority_dummy":
        if parameters:
            raise ValueError("majority DummyClassifier does not accept tuned parameters")
        return build_candidate_models(random_state)["majority_dummy"]
    if family == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_state,
                        **parameters,
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=random_state,
                        **parameters,
                    ),
                )
            ]
        )
    if family == "hist_gradient_boosting":
        return Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        early_stopping=False,
                        random_state=random_state,
                        **parameters,
                    ),
                )
            ]
        )
    raise ValueError(f"unsupported comparison model family: {family}")


def candidate_grid() -> dict[str, tuple[dict[str, Any], ...]]:
    """Return a small predeclared grid for training-blocked tuning."""
    return {
        "majority_dummy": ({},),
        "logistic_regression": tuple({"C": value} for value in (0.01, 0.1, 1.0, 10.0)),
        "random_forest": tuple(
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (8, None)
            for leaf in (5, 15)
        ),
        "hist_gradient_boosting": tuple(
            {"max_iter": iterations, "learning_rate": rate, "max_leaf_nodes": 15}
            for iterations in (100, 200)
            for rate in (0.05, 0.1)
        ),
    }


def select_comparison_candidate(selection_rows: list[dict[str, Any]]) -> str:
    """Rank candidates using blocked-training and fixed-validation evidence."""
    if not selection_rows:
        raise ValueError("at least one candidate is required")

    def selection_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
        blocked_cv = item["blocked_cv"]
        cv_balanced = blocked_cv["mean_balanced_accuracy"]
        validation_balanced = item["validation_balanced_accuracy_threshold_0_5"]
        validation_auc = (
            item["validation_roc_auc"]
            if item["validation_roc_auc"] is not None
            else -1.0
        )
        stability = -blocked_cv["std_balanced_accuracy"]
        return (
            (cv_balanced + validation_balanced) / 2,
            stability,
            validation_auc,
            item["family"],
        )

    return max(selection_rows, key=selection_key)["family"]


def blocked_temporal_folds(
    training_indices: np.ndarray,
    *,
    embargo_cycles: int = FOLD_EMBARGO_CYCLES,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create expanding-window folds entirely inside the designated train set.

    The windows are ordered source-row blocks, not independent runs. Each
    validation window has an explicit 60-cycle buffer from its training data.
    """
    allowed = np.asarray(training_indices, dtype=np.int64)
    if allowed.ndim != 1 or len(allowed) == 0:
        raise ValueError("training indices must be a non-empty one-dimensional array")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in BLOCKED_FOLDS:
        train_stop = start - embargo_cycles
        if train_stop <= 0 or stop > len(allowed):
            raise ValueError("fixed blocked validation window is outside training data")
        fold_train = allowed[:train_stop]
        fold_validation = allowed[start:stop]
        if np.intersect1d(fold_train, fold_validation).size:
            raise ValueError("blocked training and validation rows overlap")
        folds.append((fold_train, fold_validation))
    return tuple(folds)


def _profile_distribution(
    dataset: Any,
    indices: np.ndarray,
) -> dict[str, dict[str, int]]:
    profile_columns = (
        "cooler_condition_percent",
        "valve_condition_percent",
        "pump_leakage_grade",
        "accumulator_pressure_bar",
        "stable_flag",
    )
    profile = dataset.profile_targets.iloc[indices]
    return {
        column: {
            str(value): int(count)
            for value, count in profile[column].value_counts().sort_index().items()
        }
        for column in profile_columns
    }


def _feature_shift_summary(
    features: pd.DataFrame,
    splits: CycleSplits,
    split_name: str,
) -> list[dict[str, Any]]:
    train = features.iloc[splits.train]
    held_out = features.iloc[getattr(splits, split_name)]
    scale = train.std(ddof=0).replace(0, np.nan)
    standardized_shift = (held_out.mean() - train.mean()) / scale
    ordered = standardized_shift.abs().replace([np.inf, -np.inf], np.nan).dropna()
    return [
        {
            "feature": str(name),
            "train_mean": float(train[name].mean()),
            "split_mean": float(held_out[name].mean()),
            "train_std": float(scale[name]),
            "standardized_mean_shift": float(standardized_shift[name]),
        }
        for name in ordered.nlargest(12).index
    ]


def _blocked_cv_results(
    *,
    family: str,
    parameters: dict[str, Any],
    model_features: pd.DataFrame,
    labels: pd.Series,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    random_state: int,
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        estimator = comparison_pipeline(
            family, parameters, random_state=random_state
        )
        estimator.fit(
            model_features.iloc[train_indices], labels.iloc[train_indices]
        )
        validation_x = model_features.iloc[validation_indices]
        validation_y = labels.iloc[validation_indices]
        result = calculate_binary_metrics(estimator, validation_x, validation_y)
        fold_results.append(
            {
                "fold": fold_number,
                "train_cycle_id_range": [
                    int(model_features.index[train_indices[0]]),
                    int(model_features.index[train_indices[-1]]),
                ],
                "validation_cycle_id_range": [
                    int(model_features.index[validation_indices[0]]),
                    int(model_features.index[validation_indices[-1]]),
                ],
                "validation_metrics_threshold_0_5": result,
            }
        )
    balanced_scores = [
        result["validation_metrics_threshold_0_5"]["balanced_accuracy"]
        for result in fold_results
    ]
    auc_scores = [
        result["validation_metrics_threshold_0_5"]["roc_auc"]
        for result in fold_results
        if result["validation_metrics_threshold_0_5"]["roc_auc"] is not None
    ]
    return {
        "folds": fold_results,
        "mean_balanced_accuracy": float(np.mean(balanced_scores)),
        "std_balanced_accuracy": float(np.std(balanced_scores, ddof=0)),
        "minimum_balanced_accuracy": float(np.min(balanced_scores)),
        "mean_roc_auc": float(np.mean(auc_scores)) if auc_scores else None,
    }


def compare_models(
    *,
    data_dir: Path,
    project_root: Path,
    comparison_id: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Tune within training blocks, select on validation, then assess test once."""
    safe_id = validate_run_id(comparison_id)
    output_paths = {
        "model": project_root / "models" / "hydraulic" / "comparisons" / safe_id,
        "metrics": project_root / "metrics" / "hydraulic" / "comparisons" / safe_id,
        "report": project_root / "reports" / "hydraulic" / "comparisons" / safe_id,
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite comparison outputs: " + ", ".join(existing)
        )
    dataset = load_hydraulic_dataset(data_dir)
    splits = chronological_splits(len(dataset.cycle_ids))
    all_features = dataset.model_features()
    validate_no_cross_split_duplicates(all_features.to_numpy(), splits)
    labels = dataset.pump_leakage_binary
    model_features = all_features.copy()
    model_features.index = pd.Index(dataset.cycle_ids, name="cycle_id")
    folds = blocked_temporal_folds(splits.train)
    folds_by_family: dict[str, list[dict[str, Any]]] = {}
    best_parameters: dict[str, dict[str, Any]] = {}
    best_inner_cv: dict[str, dict[str, Any]] = {}

    for family, configurations in candidate_grid().items():
        trials = []
        for parameters in configurations:
            cv = _blocked_cv_results(
                family=family,
                parameters=parameters,
                model_features=model_features,
                labels=labels,
                folds=folds,
                random_state=random_state,
            )
            trials.append({"parameters": parameters, "blocked_cv": cv})
        best = max(
            trials,
            key=lambda trial: (
                trial["blocked_cv"]["mean_balanced_accuracy"],
                trial["blocked_cv"]["minimum_balanced_accuracy"],
                trial["blocked_cv"]["mean_roc_auc"]
                if trial["blocked_cv"]["mean_roc_auc"] is not None
                else -1.0,
                json.dumps(trial["parameters"], sort_keys=True),
            ),
        )
        best_parameters[family] = best["parameters"]
        best_inner_cv[family] = best["blocked_cv"]
        folds_by_family[family] = trials

    fitted: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    tuned_thresholds: dict[str, float] = {}
    selection_rows: list[dict[str, Any]] = []
    train_features = model_features.iloc[splits.train]
    train_labels = labels.iloc[splits.train]
    validation_features = model_features.iloc[splits.validation]
    validation_labels = labels.iloc[splits.validation]

    for family, parameters in best_parameters.items():
        estimator = comparison_pipeline(family, parameters, random_state=random_state)
        estimator.fit(train_features, train_labels)
        fitted[family] = estimator
        threshold = (
            select_probability_threshold(
                estimator, validation_features, validation_labels
            )
            if family != "majority_dummy" and hasattr(estimator, "predict_proba")
            else 0.5
        )
        tuned_thresholds[family] = threshold
        validation_results[family] = {
            "threshold_0_5": calculate_binary_metrics(
                estimator, validation_features, validation_labels
            ),
            "validation_tuned_threshold": threshold,
            "validation_tuned_metrics": calculate_binary_metrics(
                estimator,
                validation_features,
                validation_labels,
                threshold=threshold,
            ),
        }
        selection_rows.append(
            {
                "family": family,
                "selected_parameters": parameters,
                "blocked_cv": best_inner_cv[family],
                "validation_balanced_accuracy_threshold_0_5": validation_results[
                    family
                ]["threshold_0_5"]["balanced_accuracy"],
                "validation_roc_auc": validation_results[family][
                    "threshold_0_5"
                ]["roc_auc"],
            }
        )

    selected_family = select_comparison_candidate(selection_rows)

    # Test partition is first touched after family, parameters, and validation
    # thresholds have been fixed using train and validation data only.
    split_results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for family, estimator in fitted.items():
        split_results[family] = {
            "training_in_sample": {
                "threshold_0_5": calculate_binary_metrics(
                    estimator, train_features, train_labels
                ),
                "validation_tuned_threshold_metrics": calculate_binary_metrics(
                    estimator,
                    train_features,
                    train_labels,
                    threshold=tuned_thresholds[family],
                ),
            },
            "validation": validation_results[family],
            "blocked_training_validation": best_inner_cv.get(family),
            "test": {},
        }
        tuned_threshold = tuned_thresholds.get(family, 0.5)
        for threshold_name, threshold in (
            ("threshold_0_5", 0.5),
            ("validation_tuned_threshold", tuned_threshold),
        ):
            _, test_features, test_labels = select_evaluation_partition(
                features=model_features,
                labels=labels,
                cycle_ids=dataset.cycle_ids,
                splits=splits,
                split_name="test",
            )
            split_results[family]["test"][threshold_name] = calculate_binary_metrics(
                estimator, test_features, test_labels, threshold=threshold
            )
        test_indices = splits.test
        test_features = model_features.iloc[test_indices]
        test_labels = labels.iloc[test_indices]
        probabilities = (
            np.asarray(estimator.predict_proba(test_features))[:, 1]
            if hasattr(estimator, "predict_proba")
            else None
        )
        predictions = (
            (probabilities >= tuned_threshold).astype(np.int64)
            if probabilities is not None
            else np.asarray(estimator.predict(test_features), dtype=np.int64)
        )
        for offset, index in enumerate(test_indices):
            prediction_rows.append(
                {
                    "model": family,
                    "cycle_id": int(dataset.cycle_ids[index]),
                    "true_pump_leakage_binary": int(test_labels.iloc[offset]),
                    "positive_class_probability": (
                        float(probabilities[offset])
                        if probabilities is not None
                        else None
                    ),
                    "prediction_threshold_0_5": (
                        int(probabilities[offset] >= 0.5)
                        if probabilities is not None
                        else int(estimator.predict(test_features.iloc[[offset]])[0])
                    ),
                    "prediction_validation_threshold": int(predictions[offset]),
                    "validation_selected_threshold": float(tuned_threshold),
                }
            )

    class_distributions = {}
    for split_name in ("train", "validation", "test"):
        indices = getattr(splits, split_name)
        split_labels = labels.iloc[indices]
        class_distributions[split_name] = {
            "cycle_id_range": [
                int(dataset.cycle_ids[indices[0]]),
                int(dataset.cycle_ids[indices[-1]]),
            ],
            "sample_count": len(indices),
            "binary_class_counts": {
                "0": int((split_labels == 0).sum()),
                "1": int((split_labels == 1).sum()),
            },
            "profile_distributions": _profile_distribution(dataset, indices),
        }

    results: dict[str, Any] = {
        "comparison_id": safe_id,
        "dataset": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "archive_sha256": dataset.archive_sha256,
        "positive_class": "1 = pump leakage present (original grades 1 or 2)",
        "random_state": random_state,
        "scikit_learn_version": sklearn.__version__,
        "feature_count": int(model_features.shape[1]),
        "model_input_features": list(model_features.columns),
        "split_method": "forward chronological 70/15/15 with 60-cycle boundary purge",
        "test_used_for_selection": False,
        "model_selection_rule": (
            "Highest mean of training-only expanding blocked-CV balanced accuracy "
            "and fixed-validation balanced accuracy at threshold 0.5; ties favor "
            "lower blocked-CV fold standard deviation, then fixed-validation "
            "ROC-AUC. Test results are excluded from selection."
        ),
        "class_distributions": class_distributions,
        "feature_distribution_shifts": {
            split_name: _feature_shift_summary(
                all_features, splits, split_name
            )
            for split_name in ("validation", "test")
        },
        "hyperparameter_trials": folds_by_family,
        "selected_parameters": best_parameters,
        "selection_scores": selection_rows,
        "selected_candidate": selected_family,
        "candidate_results": split_results,
        "limitations": [
            (
                "The source has no experimental run or rig identifiers; "
                "blocked folds are temporal blocks, not independent runs."
            ),
            "Validation and test cooler/accumulator-pressure conditions differ from training conditions.",
            "The fixed validation interval contains long single-class temporal blocks.",
            (
                "Per-cycle summaries use measurements from the complete labelled "
                "cycle; this is retrospective classification, not early warning."
            ),
            "Model probabilities are uncalibrated scores and are not reported as confidence.",
            "No candidate is established as generalizing to independent runs or rigs.",
        ],
        "validation_method": (
            "Four expanding, forward temporal blocks inside the nominal training "
            "partition, each with a 60-cycle gap, followed by the fixed chronological "
            "validation partition. These blocks do not establish independent runs."
        ),
    }

    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=False)
    for family, estimator in fitted.items():
        joblib.dump(estimator, output_paths["model"] / f"{family}.joblib")
    (output_paths["metrics"] / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        output_paths["metrics"] / "test_predictions.csv", index=False
    )
    metadata = {
        "comparison_id": safe_id,
        "selected_candidate": selected_family,
        "model_versions": {
            family: {
                "family": family,
                "parameters": best_parameters.get(family, {}),
                "random_state": random_state,
            }
            for family in fitted
        },
        "archive_sha256": dataset.archive_sha256,
        "dataset_doi": DATASET_DOI,
        "validation_method": results["validation_method"],
    }
    (output_paths["model"] / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results["mlflow_run_ids"] = log_comparison(
        project_root,
        safe_id,
        tracking_uri=tracking_uri,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or project_root / "data" / "inspection" / (
        "uci-hydraulic-condition-monitoring"
    ) / "extracted"
    results = compare_models(
        data_dir=data_dir,
        project_root=project_root,
        comparison_id=args.comparison_id,
        random_state=args.random_state,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
