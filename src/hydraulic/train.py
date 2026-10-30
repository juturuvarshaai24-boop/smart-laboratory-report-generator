"""Train the isolated, cycle-aware hydraulic pump-leakage pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from .artifacts import (
    create_run_directories,
    new_run_id,
    run_paths,
    save_dataset_artifacts,
)
from .data import (
    DATASET_DOI,
    DATASET_NAME,
    DATASET_URL,
    FEATURE_RECIPE_VERSION,
    PHYSICAL_SENSOR_NAMES,
    SENSOR_SPECS,
    VIRTUAL_SENSOR_NAMES,
    load_hydraulic_dataset,
)
from .modeling import (
    DEFAULT_RANDOM_STATE,
    build_candidate_models,
    calculate_binary_metrics,
    select_candidate,
    select_probability_threshold,
)
from .splitting import (
    DEFAULT_EMBARGO_CYCLES,
    chronological_splits,
    validate_no_cross_split_duplicates,
)


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_pipeline(
    *,
    data_dir: Path,
    project_root: Path,
    run_id: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[str, dict[str, Any]]:
    """Fit on the chronological training partition and assess held-out sets."""
    dataset = load_hydraulic_dataset(data_dir)
    splits = chronological_splits(len(dataset.cycle_ids))
    model_features = dataset.model_features()
    validate_no_cross_split_duplicates(model_features.to_numpy(), splits)
    labels = dataset.pump_leakage_binary.to_numpy()

    run_id = run_id or new_run_id()
    paths = run_paths(project_root, run_id)
    create_run_directories(paths)
    save_dataset_artifacts(dataset, paths["processed_dir"], splits)

    candidates = build_candidate_models(random_state)
    train_indices = splits.train
    validation_features = model_features.iloc[splits.validation]
    validation_labels = dataset.pump_leakage_binary.iloc[splits.validation]
    validation_results: dict[str, dict[str, Any]] = {}

    for estimator in candidates.values():
        estimator.fit(model_features.iloc[train_indices], labels[train_indices])
    for name, estimator in candidates.items():
        threshold = (
            select_probability_threshold(
                estimator, validation_features, validation_labels
            )
            if name == "logistic_regression"
            else 0.5
        )
        validation_results[name] = calculate_binary_metrics(
            estimator,
            validation_features,
            validation_labels,
            threshold=threshold,
        )
        validation_results[name]["decision_threshold"] = threshold
    selected_name = select_candidate(validation_results)
    selected_model = candidates[selected_name]
    selected_threshold = validation_results[selected_name]["decision_threshold"]

    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        indices = getattr(splits, split_name)
        split_features = model_features.iloc[indices]
        split_labels = dataset.pump_leakage_binary.iloc[indices]
        split_metrics[split_name] = {
            "cycle_id_start": int(dataset.cycle_ids[indices[0]]),
            "cycle_id_end": int(dataset.cycle_ids[indices[-1]]),
            "selected_model": calculate_binary_metrics(
                selected_model,
                split_features,
                split_labels,
                threshold=selected_threshold,
            ),
            "majority_dummy_baseline": calculate_binary_metrics(
                candidates["majority_dummy"], split_features, split_labels
            ),
        }

    class_counts = {
        split_name: {
            "0": int(np.sum(labels[getattr(splits, split_name)] == 0)),
            "1": int(np.sum(labels[getattr(splits, split_name)] == 1)),
        }
        for split_name in ("train", "validation", "test")
    }
    grade_counts = {
        split_name: {
            str(grade): int(
                np.sum(
                    dataset.pump_leakage_grade.to_numpy()[
                        getattr(splits, split_name)
                    ]
                    == grade
                )
            )
            for grade in (0, 1, 2)
        }
        for split_name in ("train", "validation", "test")
    }
    split_manifest = {
        "split_method": "forward_chronological_70_15_15_with_purge",
        "cycle_duration_seconds_documented": 60,
        "embargo_cycles_each_side_of_boundary": DEFAULT_EMBARGO_CYCLES,
        "train_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.train[0]]),
            int(dataset.cycle_ids[splits.train[-1]]),
        ],
        "validation_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.validation[0]]),
            int(dataset.cycle_ids[splits.validation[-1]]),
        ],
        "test_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.test[0]]),
            int(dataset.cycle_ids[splits.test[-1]]),
        ],
        "purged_cycle_count": len(splits.purged),
        "class_counts": class_counts,
        "original_pump_grade_counts": grade_counts,
        "limitations": [
            "Source files provide row ordering but no run, rig, or timestamp identifiers.",
            "The one-hour boundary purge cannot establish statistical independence.",
            "Results measure later-cycle performance within one experimental test rig only.",
        ],
    }
    metrics = {
        "dataset": DATASET_NAME,
        "doi": DATASET_DOI,
        "target": "pump_leakage_binary: 0=no leakage; 1=weak or severe leakage",
        "selection_metric": (
            "validation balanced accuracy, then F1 and ROC-AUC; logistic "
            "threshold selected on validation only"
        ),
        "selected_model": selected_name,
        "split_metrics": split_metrics,
        "validation_candidate_metrics": validation_results,
    }
    metadata = {
        "run_id": run_id,
        "dataset": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "archive_sha256": dataset.archive_sha256,
        "feature_recipe_version": FEATURE_RECIPE_VERSION,
        "feature_statistics": [
            "mean",
            "population_std_ddof_0",
            "minimum",
            "maximum",
            "root_mean_square",
            "25th_percentile",
            "median",
            "75th_percentile",
        ],
        "all_sensors_summarized": list(SENSOR_SPECS),
        "model_input_sensors": list(PHYSICAL_SENSOR_NAMES),
        "excluded_virtual_sensors": list(VIRTUAL_SENSOR_NAMES),
        "model_feature_names": list(model_features.columns),
        "model_feature_count": int(model_features.shape[1]),
        "target_mapping": {"0": "no pump leakage", "1": "pump leakage present"},
        "original_pump_grade_preserved": True,
        "random_state": random_state,
        "estimator": selected_name,
        "selection_metric": metrics["selection_metric"],
        "validation_threshold": float(selected_threshold),
        "sklearn_version": sklearn.__version__,
        "cycle_alignment": "positional 1-based row index per UCI documentation",
        "report_warning": (
            "Predictions are model outputs, not direct measurements or certified diagnoses."
        ),
    }
    joblib.dump(selected_model, paths["model_dir"] / "pipeline.joblib")
    _json_write(paths["model_dir"] / "training_metadata.json", metadata)
    _json_write(paths["model_dir"] / "split_manifest.json", split_manifest)
    _json_write(paths["metrics_dir"] / "metrics.json", metrics)
    return run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or project_root / "data" / "inspection" / (
        "uci-hydraulic-condition-monitoring"
    ) / "extracted"
    run_id, metrics = train_pipeline(
        data_dir=data_dir,
        project_root=project_root,
        run_id=args.run_id,
        random_state=args.random_state,
    )
    print(f"Hydraulic run ID: {run_id}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Model: models/hydraulic/runs/{run_id}/pipeline.joblib")


if __name__ == "__main__":
    main()
