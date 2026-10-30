"""Hydraulic-specific report adapter and per-cycle report command."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model.config import ModelConfig
from model.generator import LabReportGenerator
from model.inference import MockModelBackend, ModelBackend
from model.schemas import ExperimentInput, LabReport, Measurement, SectionName
from model.validator import validate_report

from .artifacts import run_paths
from .data import (
    DATASET_DOI,
    DATASET_NAME,
    DATASET_URL,
    FEATURE_RECIPE_VERSION,
    FEATURE_STATISTICS,
    PHYSICAL_SENSOR_NAMES,
    SENSOR_SPECS,
    VIRTUAL_SENSOR_NAMES,
    load_hydraulic_dataset,
)
from .evaluate import evaluate_run

ATTRIBUTION = (
    "Helwig, Pignanelli, and Schütze, “Condition Monitoring of a Complex "
    "Hydraulic System Using Multivariate Statistics,” IEEE I2MTC 2015, "
    "DOI: 10.1109/I2MTC.2015.7151267."
)
MODEL_WARNING = (
    "Predictions are model outputs, not direct laboratory measurements, "
    "certified diagnoses, or maintenance instructions."
)


class HydraulicReportBackend(ModelBackend):
    """Generate structured cycle reports and a physical-sensor prediction."""

    def __init__(
        self,
        config: ModelConfig,
        model: Any,
        evaluation: dict[str, Any],
        decision_threshold: float,
        ground_truth: int | None = None,
        include_probability: bool = True,
    ) -> None:
        super().__init__(config)
        if not np.isfinite(decision_threshold) or not 0 <= decision_threshold <= 1:
            raise ValueError("decision threshold must be a finite value in [0, 1]")
        self._model = model
        self._evaluation = evaluation
        self._decision_threshold = decision_threshold
        self._ground_truth = ground_truth
        self._include_probability = include_probability
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            raise ValueError("hydraulic model must expose its fitted feature names")
        self._model_feature_names = tuple(str(name) for name in feature_names)
        expected = tuple(
            f"{sensor}__{statistic}"
            for sensor in PHYSICAL_SENSOR_NAMES
            for statistic in FEATURE_STATISTICS
        )
        if self._model_feature_names != expected:
            raise ValueError("hydraulic model feature names/order are not supported")
        self._fallback = MockModelBackend(config)

    def _predict(self, experiment: ExperimentInput) -> tuple[int, float | None]:
        values = {measurement.name: measurement.value for measurement in experiment.measurements}
        missing = [name for name in self._model_feature_names if name not in values]
        if missing:
            raise ValueError(f"report is missing model features: {missing}")
        frame = pd.DataFrame(
            [[float(values[name]) for name in self._model_feature_names]],
            columns=list(self._model_feature_names),
        )
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("report contains non-finite model features")
        probability = None
        if hasattr(self._model, "predict_proba"):
            classes = np.asarray(self._model.classes_)
            positive = np.flatnonzero(classes == 1)
            if len(positive) == 1:
                candidate = float(
                    self._model.predict_proba(frame)[0, int(positive[0])]
                )
                if np.isfinite(candidate) and 0 <= candidate <= 1:
                    predicted = int(candidate >= self._decision_threshold)
                    probability = (
                        candidate if predicted == 1 else 1.0 - candidate
                    )
                else:
                    predicted = int(self._model.predict(frame)[0])
            else:
                predicted = int(self._model.predict(frame)[0])
        else:
            predicted = int(self._model.predict(frame)[0])
        return predicted, probability

    def generate(self, section: SectionName, experiment: ExperimentInput) -> str:
        if section not in {SectionName.OBJECTIVE, SectionName.MATERIALS,
                           SectionName.PROCEDURE, SectionName.RESULTS,
                           SectionName.CONCLUSION}:
            raise ValueError(f"unsupported report section: {section.value}")
        if section == SectionName.OBJECTIVE:
            return (
                "Retrospectively estimate whether the documented hydraulic "
                "cycle shows pump leakage from cycle-level physical sensor features."
            )
        if section == SectionName.MATERIALS:
            return (
                "Smart Laboratory Report Generator; UCI hydraulic "
                "condition-monitoring experimental test rig; "
                "17 documented sensor channels; 60-second cycles. "
                f"Source: {DATASET_NAME} (DOI {DATASET_DOI}). "
                f"Attribution requested: {ATTRIBUTION}"
            )
        if section == SectionName.PROCEDURE:
            return (
                "Sensor rows were aligned positionally using the UCI documented "
                "row-as-cycle convention. Per-cycle mean, population standard "
                "deviation (ddof=0), minimum, maximum, RMS, 25th percentile, "
                "median, and 75th percentile were computed independently at each "
                "channel's native sampling rate. The fitted pipeline uses the "
                "14 physical channels only; profile targets and virtual channels "
                f"{', '.join(VIRTUAL_SENSOR_NAMES)} are excluded from prediction. "
                "Scaling was fitted on the training partition only; the "
                "classification threshold was selected on validation data. "
                "This is retrospective classification using full-cycle sensor "
                "summaries, not a pre-cycle forecast or early-warning result. "
                "Native rates: PS1-PS6/EPS1 100 Hz, FS1/FS2 10 Hz, "
                "TS1-TS4/VS1/CE/CP/SE 1 Hz."
            )
        if section == SectionName.RESULTS:
            cycle_id = experiment.metadata["cycle_id"]
            lines = [
                f"Cycle identifier: {cycle_id} (1-based source row index; no timestamp).",
                (f"Model: {experiment.metadata['model_name']} "
                 f"({experiment.metadata['model_version']})."),
                ("Validation method: "
                 f"{str(experiment.metadata['validation_method']).rstrip('.')}."),
                ("Positive class definition: label 1 = pump leakage present "
                 "(original pump grade 1 or 2); label 0 = original grade 0."),
                "Derived sensor features (each retains its source channel unit):",
            ]
            lines.extend(
                f"- {measurement.name}: {measurement.value:.8g} {measurement.unit}"
                for measurement in experiment.measurements
            )
            prediction, probability = self._predict(experiment)
            label = "pump leakage present" if prediction == 1 else "no pump leakage"
            lines.append(f"Predicted class: {prediction} ({label}).")
            if probability is not None and self._include_probability:
                lines.append(
                    "Estimated model probability for predicted class "
                    f"(not calibrated uncertainty): {probability:.6g}."
                )
            if self._ground_truth is not None:
                lines.append(
                    "Ground-truth pump-leakage class (retrospective only): "
                    f"{self._ground_truth}."
                )
            lines.append(
                "Decision threshold for positive class 1, selected on validation data: "
                f"{self._decision_threshold:.6g}."
            )
            for split_name in ("train", "validation", "test"):
                if split_name not in self._evaluation["splits"]:
                    continue
                split = self._evaluation["splits"][split_name]
                metrics = split["selected_model"]
                support = metrics["class_counts"]
                auc = (
                    f"{metrics['roc_auc']:.4f}"
                    if metrics["roc_auc"] is not None
                    else "unavailable"
                )
                split_label = (
                    "training (in-sample)"
                    if split_name == "train"
                    else (
                        "validation (selection)"
                        if split_name == "validation"
                        else "locked chronological test"
                    )
                )
                lines.append(
                    f"{split_label} cycles {split['cycle_id_start']}-"
                    f"{split['cycle_id_end']}: support 0/1 "
                    f"{support['0']}/{support['1']}; confusion matrix "
                    f"{metrics['confusion_matrix']}; accuracy "
                    f"{metrics['accuracy']:.4f}, precision "
                    f"{metrics['precision']:.4f}, recall "
                    f"{metrics['recall']:.4f}, F1 {metrics['f1']:.4f}, "
                    f"balanced accuracy {metrics['balanced_accuracy']:.4f}, "
                    f"ROC-AUC {auc}."
                )
            return "\n".join(lines)
        if section == SectionName.CONCLUSION:
            return (
                f"{MODEL_WARNING} Data come from one experimental rig; source "
                "files have no timestamp/run identifier, and the time purge "
                "does not establish independent cycles. Sensor feature "
                "summaries are derived values, not raw measurements."
            )
        return self._fallback.generate(section, experiment)


def build_cycle_report(
    *,
    dataset: Any,
    model: Any,
    evaluation: dict[str, Any],
    cycle_id: int,
    decision_threshold: float = 0.5,
    include_ground_truth: bool = False,
    model_name: str = "Logistic Regression",
    model_version: str = "hydraulic-pump-leakage-v1",
    validation_method: str = (
        "forward chronological 70/15/15 split with a 60-cycle purge"
    ),
    include_probability: bool = True,
    comparison_id: str | None = None,
) -> LabReport:
    """Build and validate a report for one 1-based cycle row."""
    if cycle_id not in set(dataset.cycle_ids.tolist()):
        raise ValueError(f"cycle_id must be between 1 and {len(dataset.cycle_ids)}")
    cycle_features = dataset.features.loc[cycle_id]
    measurements = []
    for sensor_name, spec in SENSOR_SPECS.items():
        for statistic in FEATURE_STATISTICS:
            name = f"{sensor_name}__{statistic}"
            measurements.append(
                Measurement(
                    name=name,
                    value=float(cycle_features[name]),
                    unit=str(spec["unit"]),
                )
            )
    metadata = {
        "project_name": "Smart Laboratory Report Generator",
        "model_name": model_name,
        "model_version": model_version,
        "validation_method": validation_method,
        "cycle_id": str(cycle_id),
        "dataset": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "archive_sha256": dataset.archive_sha256,
        "feature_recipe_version": FEATURE_RECIPE_VERSION,
        "target": (
            "pump leakage: 0=no leakage; 1=leakage present "
            "(original pump grade 1 or 2)"
        ),
        "positive_class_definition": (
            "label 1 = pump leakage present (original pump grade 1 or 2)"
        ),
        "model_validation": validation_method,
        "validation_decision_threshold": str(decision_threshold),
        "prediction_warning": MODEL_WARNING,
        "ground_truth_note": "retrospective only; never a model input",
    }
    if comparison_id is not None:
        metadata["comparison_id"] = comparison_id
    ground_truth = None
    if include_ground_truth:
        ground_truth = int(dataset.pump_leakage_binary.loc[cycle_id])
        metadata["ground_truth_pump_leakage_binary"] = str(ground_truth)
        metadata["ground_truth_original_pump_grade"] = str(
            int(dataset.pump_leakage_grade.loc[cycle_id])
        )
    config = ModelConfig(backend="hydraulic", model_id="hydraulic-pump-leakage-v1")
    config.model_id = model_version
    backend = HydraulicReportBackend(
        config,
        model,
        evaluation,
        decision_threshold,
        ground_truth,
        include_probability,
    )
    experiment = ExperimentInput(
        title=f"Hydraulic condition report — cycle {cycle_id}",
        objective="assess cycle-level pump-leakage condition retrospectively",
        materials=["UCI hydraulic test rig", "17-channel sensor system"],
        procedure=[
            "Validate audited source and positional row alignment.",
            "Summarize each sensor cycle using the documented feature recipe.",
            "Apply the training-fitted pipeline to physical sensor features.",
        ],
        measurements=measurements,
        metadata=metadata,
    )
    report = LabReportGenerator(backend=backend, config=config).generate_from_input(
        experiment
    )
    result = validate_report(report)
    if not result.is_valid:
        issues = "; ".join(issue.message for issue in result.issues)
        raise ValueError(f"generated hydraulic report failed validation: {issues}")
    return report


def serialize_report(report: LabReport) -> dict[str, Any]:
    """Convert report dataclasses and enum values to JSON-safe primitives."""
    payload = asdict(report)
    payload["status"] = report.status.value
    payload["sections"] = [
        {"name": section.name.value, "content": section.content}
        for section in report.sections
    ]
    return payload


def comparison_cycle_split(
    cycle_id: int,
    class_distributions: dict[str, Any],
) -> str:
    """Resolve a source cycle's saved comparison split or purged membership."""
    if cycle_id < 1:
        raise ValueError("cycle ID must be a positive source-row index")
    for split_name in ("train", "validation", "test"):
        cycle_range = class_distributions[split_name]["cycle_id_range"]
        if cycle_range[0] <= cycle_id <= cycle_range[1]:
            return split_name
    return "purged"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--comparison-id")
    parser.add_argument("--candidate")
    parser.add_argument("--cycle-id", required=True, type=int)
    parser.add_argument("--include-ground-truth", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or project_root / "data" / "inspection" / (
        "uci-hydraulic-condition-monitoring"
    ) / "extracted"
    dataset = load_hydraulic_dataset(data_dir)
    if args.comparison_id:
        comparison_id = args.comparison_id
        model_dir = (
            project_root / "models" / "hydraulic" / "comparisons" / comparison_id
        )
        metrics_path = (
            project_root / "metrics" / "hydraulic" / "comparisons" / comparison_id
            / "comparison.json"
        )
        metadata_path = model_dir / "comparison_metadata.json"
        if not metadata_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"no complete hydraulic comparison found for {comparison_id}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metadata["archive_sha256"] != dataset.archive_sha256:
            raise ValueError("current source archive differs from comparison provenance")
        candidate = args.candidate or metadata["selected_candidate"]
        if candidate not in metadata["model_versions"]:
            raise ValueError(f"candidate is not present in comparison: {candidate}")
        model_path = model_dir / f"{candidate}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"comparison model file is missing: {model_path}")
        model = joblib.load(model_path)
        candidate_results = results["candidate_results"][candidate]
        training = candidate_results["training_in_sample"]
        validation = candidate_results["validation"]
        test = candidate_results["test"]
        evaluation = {
            "splits": {
                "train": {
                    "cycle_id_start": results["class_distributions"]["train"][
                        "cycle_id_range"
                    ][0],
                    "cycle_id_end": results["class_distributions"]["train"][
                        "cycle_id_range"
                    ][1],
                    "selected_model": training[
                        "validation_tuned_threshold_metrics"
                    ],
                },
                "validation": {
                    "cycle_id_start": results["class_distributions"][
                        "validation"
                    ]["cycle_id_range"][0],
                    "cycle_id_end": results["class_distributions"][
                        "validation"
                    ]["cycle_id_range"][1],
                    "selected_model": validation["validation_tuned_metrics"],
                },
                "test": {
                    "cycle_id_start": results["class_distributions"]["test"][
                        "cycle_id_range"
                    ][0],
                    "cycle_id_end": results["class_distributions"]["test"][
                        "cycle_id_range"
                    ][1],
                    "selected_model": test["validation_tuned_threshold"],
                },
            }
        }
        family_metadata = metadata["model_versions"][candidate]
        model_version = f"hydraulic-comparison-v1:{candidate}"
        validation_method = results["validation_method"]
        threshold = float(
            candidate_results["validation"]["validation_tuned_threshold"]
        )
        report = build_cycle_report(
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            decision_threshold=threshold,
            cycle_id=args.cycle_id,
            model_name=candidate.replace("_", " ").title(),
            model_version=model_version,
            validation_method=validation_method,
            include_probability=False,
            include_ground_truth=args.include_ground_truth,
            comparison_id=comparison_id,
        )
        report.metadata.update(
            {
                "model_parameters": json.dumps(
                    family_metadata["parameters"], sort_keys=True
                ),
                "model_random_state": str(family_metadata["random_state"]),
                "probability_disclosure": (
                    "Omitted: candidate probabilities are uncalibrated model scores."
                ),
                "cycle_split_membership": comparison_cycle_split(
                    args.cycle_id, results["class_distributions"]
                ),
                "generalization_warning": (
                    "Validation and test are cooler condition 100%, unlike most "
                    "training cycles; accumulator-pressure conditions also shift. "
                    "The source is sequential and provides no independent run or "
                    "rig identifiers, so independent-run generalization has not "
                    "been established. This is retrospective classification of "
                    "a completed cycle, not early warning."
                ),
            }
        )
        validation_result = validate_report(report)
        if not validation_result.is_valid:
            issues = "; ".join(
                issue.message for issue in validation_result.issues
            )
            raise ValueError(
                f"comparison report failed validation after metadata updates: {issues}"
            )
        output_dir = (
            project_root / "reports" / "hydraulic" / "comparisons" / comparison_id
        )
        output_name = f"cycle_{args.cycle_id:04d}_{candidate}_report.json"
    else:
        if args.candidate:
            raise ValueError("--candidate requires --comparison-id")
        run_id = args.run_id
        paths = run_paths(project_root, run_id)
        model_path = paths["model_dir"] / "pipeline.joblib"
        metadata_path = paths["model_dir"] / "training_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"no complete hydraulic model run found for {run_id}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["archive_sha256"] != dataset.archive_sha256:
            raise ValueError("current source archive differs from the model's recorded source")
        model = joblib.load(model_path)
        evaluation = evaluate_run(
            run_id=run_id,
            data_dir=data_dir,
            project_root=project_root,
        )
        report = build_cycle_report(
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            decision_threshold=float(metadata["validation_threshold"]),
            cycle_id=args.cycle_id,
            include_ground_truth=args.include_ground_truth,
        )
        output_dir = project_root / "reports" / "hydraulic" / "runs" / run_id
        output_name = f"cycle_{args.cycle_id:04d}_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    output_path.write_text(
        json.dumps(serialize_report(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated report saved to {output_path}")


if __name__ == "__main__":
    main()
