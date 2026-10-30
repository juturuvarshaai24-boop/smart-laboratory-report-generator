
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.hydraulic.data import ARCHIVE_SHA256, DATASET_DOI
from src.hydraulic.mlflow_tracking import (
    DATASET_TAG,
    EXPERIMENT_NAME,
    log_comparison,
)


def _metric_set(*, threshold: float) -> dict:
    return {
        "accuracy": 0.8,
        "balanced_accuracy": 0.75,
        "precision": 0.82,
        "recall": 0.7,
        "f1": 0.75,
        "roc_auc": 0.84,
        "confusion_matrix": [[8, 2], [3, 7]],
        "class_counts": {"0": 10, "1": 10},
        "cycle_count": 20,
        "decision_threshold": threshold,
    }


class HydraulicMlflowTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.uri = f"sqlite:///{self.root / 'tracking.db'}"

        mlflow.set_tracking_uri(self.uri)

        self.comparison_id = "hydraulic-fixture-01"

        self.model_dir = (
            self.root
            / "models"
            / "hydraulic"
            / "comparisons"
            / self.comparison_id
        )

        self.metrics_dir = (
            self.root
            / "metrics"
            / "hydraulic"
            / "comparisons"
            / self.comparison_id
        )

        self.model_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        estimator = Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=2,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                )
            ]
        )

        fixture_features = pd.DataFrame(
            np.arange(20 * 112, dtype=float).reshape(20, 112),
            columns=[f"feature_{index}" for index in range(112)],
        )

        fixture_labels = pd.Series([0, 1] * 10)
        estimator.fit(fixture_features, fixture_labels)

        joblib.dump(
            estimator,
            self.model_dir / "random_forest.joblib",
        )

        (self.model_dir / "comparison_metadata.json").write_text(
            json.dumps(
                {
                    "comparison_id": self.comparison_id,
                    "archive_sha256": ARCHIVE_SHA256,
                    "dataset_doi": DATASET_DOI,
                    "selected_candidate": "random_forest",
                    "model_versions": {
                        "random_forest": {
                            "family": "random_forest",
                            "parameters": {
                                "max_depth": None,
                                "min_samples_leaf": 5,
                            },
                            "random_state": 42,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        self._write_comparison()
        self.client = MlflowClient(tracking_uri=self.uri)

    def tearDown(self) -> None:
        for uri, engine in list(SqlAlchemyStore._engine_map.items()):
            if uri == self.uri:
                SqlAlchemyStore._engine_map.pop(uri).dispose()

        mlflow.set_tracking_uri("sqlite:///:memory:")
        self.temporary_directory.cleanup()

    def _write_comparison(self) -> None:
        candidate = {
            "training_in_sample": {
                "threshold_0_5": _metric_set(threshold=0.5),
            },
            "validation": {
                "threshold_0_5": _metric_set(threshold=0.5),
                "validation_tuned_threshold": 0.2,
                "validation_tuned_metrics": _metric_set(threshold=0.2),
            },
            "test": {
                "threshold_0_5": _metric_set(threshold=0.5),
                "validation_tuned_threshold": _metric_set(threshold=0.2),
            },
            "blocked_training_validation": {
                "mean_balanced_accuracy": 0.7,
                "std_balanced_accuracy": 0.05,
                "minimum_balanced_accuracy": 0.6,
            },
        }

        results = {
            "comparison_id": self.comparison_id,
            "dataset": "UCI Condition Monitoring of Hydraulic Systems",
            "dataset_url": (
                "https://archive.ics.uci.edu/dataset/447/"
                "condition+monitoring+of+hydraulic+systems"
            ),
            "dataset_doi": DATASET_DOI,
            "archive_sha256": ARCHIVE_SHA256,
            "positive_class": (
                "1 = pump leakage present (original grades 1 or 2)"
            ),
            "random_state": 42,
            "feature_count": 112,
            "split_method": (
                "forward chronological 70/15/15 with "
                "60-cycle boundary purge"
            ),
            "test_used_for_selection": False,
            "validation_method": (
                "forward blocked validation, then fixed chronology"
            ),
            "selected_candidate": "random_forest",
            "class_distributions": {
                "train": {
                    "cycle_id_range": [1, 1483],
                    "sample_count": 1483,
                    "binary_class_counts": {"0": 827, "1": 656},
                },
                "validation": {
                    "cycle_id_range": [1604, 1814],
                    "sample_count": 211,
                    "binary_class_counts": {"0": 122, "1": 89},
                },
                "test": {
                    "cycle_id_range": [1935, 2205],
                    "sample_count": 271,
                    "binary_class_counts": {"0": 107, "1": 164},
                },
            },
            "candidate_results": {
                "random_forest": candidate,
            },
        }

        (self.metrics_dir / "comparison.json").write_text(
            json.dumps(results),
            encoding="utf-8",
        )

        (self.metrics_dir / "test_predictions.csv").write_text(
            "model,cycle_id,true_pump_leakage_binary\n"
            "random_forest,2000,1\n",
            encoding="utf-8",
        )

    def test_hydraulic_experiment_reuse_tags_metrics_and_artifacts(
        self,
    ) -> None:
        first = log_comparison(
            self.root,
            self.comparison_id,
            tracking_uri=self.uri,
        )

        second = log_comparison(
            self.root,
            self.comparison_id,
            tracking_uri=self.uri,
        )

        self.assertEqual(first, second)

        experiment = self.client.get_experiment_by_name(EXPERIMENT_NAME)
        self.assertIsNotNone(experiment)

        runs = self.client.search_runs(
            [experiment.experiment_id],
            max_results=10,
        )

        self.assertEqual(len(runs), 1)
        run = runs[0]

        # Parse the file URI instead of comparing URI strings.
        # On macOS, /var and /private/var can refer to the same path.
        artifact_uri = urlparse(run.info.artifact_uri)
        self.assertEqual(artifact_uri.scheme, "file")

        artifact_path = Path(unquote(artifact_uri.path))

        self.assertTrue(
            artifact_path.resolve().is_relative_to(self.root.resolve()),
            (
                f"Artifact path {artifact_path.resolve()} is outside "
                f"test root {self.root.resolve()}"
            ),
        )

        self.assertEqual(
            run.data.tags["dataset"],
            DATASET_TAG,
        )
        self.assertEqual(
            run.data.tags["task"],
            "binary-pump-leakage-classification",
        )
        self.assertEqual(
            run.data.tags["stage"],
            "model_comparison",
        )
        self.assertEqual(
            run.data.tags["evaluation"],
            "chronological",
        )
        self.assertEqual(
            run.data.tags["comparison_id"],
            self.comparison_id,
        )
        self.assertEqual(
            run.data.tags["probability_interpretation"],
            "uncalibrated model scores, not calibrated confidence",
        )

        self.assertEqual(
            run.data.params["validation_selected_threshold"],
            "0.2",
        )

        self.assertAlmostEqual(
            run.data.metrics["validation.threshold_0_5.accuracy"],
            0.8,
        )
        self.assertAlmostEqual(
            run.data.metrics["test.validation_selected.accuracy"],
            0.8,
        )

        self.assertIn(
            "validation.validation_selected.confusion_matrix.negative_negative",
            run.data.metrics,
        )

        artifact_paths = {
            artifact.path
            for artifact in self.client.list_artifacts(
                run.info.run_id,
                "comparison",
            )
        }

        self.assertIn(
            "comparison/comparison.json",
            artifact_paths,
        )
        self.assertIn(
            "comparison/test_predictions.csv",
            artifact_paths,
        )

        self.assertEqual(
            self.client.get_run(run.info.run_id).info.status,
            "FINISHED",
        )

    def test_duplicate_candidate_records_are_rejected(self) -> None:
        log_comparison(
            self.root,
            self.comparison_id,
            tracking_uri=self.uri,
        )

        experiment = self.client.get_experiment_by_name(EXPERIMENT_NAME)

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            tags={
                "comparison_id": self.comparison_id,
                "candidate": "random_forest",
            },
        ):
            pass

        with self.assertRaisesRegex(
            RuntimeError,
            "duplicate MLflow records",
        ):
            log_comparison(
                self.root,
                self.comparison_id,
                tracking_uri=self.uri,
            )

    def test_missing_artifact_is_reported_before_logging(self) -> None:
        (self.model_dir / "random_forest.joblib").unlink()

        with self.assertRaisesRegex(
            FileNotFoundError,
            "candidate estimator is missing",
        ):
            log_comparison(
                self.root,
                self.comparison_id,
                tracking_uri=self.uri,
            )

        self.assertIsNone(
            self.client.get_experiment_by_name(EXPERIMENT_NAME)
        )

    def test_missing_comparison_summary_is_reported(self) -> None:
        (self.metrics_dir / "comparison.json").unlink()

        with self.assertRaisesRegex(
            FileNotFoundError,
            "comparison.json",
        ):
            log_comparison(
                self.root,
                self.comparison_id,
                tracking_uri=self.uri,
            )

    def test_unavailable_tracking_store_has_contextual_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "tracking store"):
            log_comparison(
                self.root,
                self.comparison_id,
                tracking_uri="missing-store-scheme://local",
            )

    def test_deleted_wine_runs_are_not_active_in_isolated_store(self) -> None:
        wine_experiment_id = self.client.create_experiment(
            "smart-lab-report",
            artifact_location=(
                self.root / "mlruns" / "legacy-wine"
            ).as_uri(),
        )

        with mlflow.start_run(
            experiment_id=wine_experiment_id,
            tags={
                "dataset": "winequality-red",
                "stage": "baseline",
            },
        ):
            mlflow.log_metric("test.f1", 0.7)

        self.client.delete_experiment(wine_experiment_id)

        active = self.client.search_runs(
            [wine_experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=10,
        )
        self.assertEqual(active, [])

        all_runs = self.client.search_runs(
            [wine_experiment_id],
            run_view_type=ViewType.ALL,
            max_results=10,
        )
        self.assertEqual(len(all_runs), 1)
        self.assertEqual(
            all_runs[0].info.lifecycle_stage,
            "deleted",
        )

        environment = os.environ.copy()
        environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gc",
                "--backend-store-uri",
                self.uri,
                "--experiment-ids",
                str(wine_experiment_id),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )

        with sqlite3.connect(self.root / "tracking.db") as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
                    (wine_experiment_id,),
                ).fetchone()[0],
                0,
            )

            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE experiment_id = ?",
                    (wine_experiment_id,),
                ).fetchone()[0],
                0,
            )


if __name__ == "__main__":
    unittest.main()
