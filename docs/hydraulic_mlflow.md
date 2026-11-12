# Hydraulic MLflow tracking

## Active tracking setup

Hydraulic comparisons are logged to the MLflow experiment
`hydraulic-condition-monitoring`. The default tracking store is the project
SQLite database. From the repository root, its relative URI is:

```text
sqlite:///mlflow.db
```

`src/hydraulic/mlflow_tracking.py` centralizes URI resolution. An explicit
`--tracking-uri` takes precedence, then `MLFLOW_TRACKING_URI`, then the
repository-local `mlflow.db`. The Python API accepts the same optional
`tracking_uri` argument. The database and `mlruns/` artifacts are ignored by
Git. A fresh clone creates its own local store when a comparison is logged;
neither this machine's database nor a remote service is required.

## Log a comparison

Every normal comparison command writes its models and metrics to unique
comparison output directories and then logs one MLflow run per candidate.
For example:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.comparison `
  --comparison-id hydraulic-compare-new-01 `
  --random-state 42
```

To log an already-saved comparison without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.mlflow_tracking `
  --comparison-id hydraulic-compare-20261021-2104d
```

The logger verifies the saved comparison ID, dataset DOI and audited archive
SHA-256, 112-feature schema, test-selection flag, metadata, candidate
estimators, metrics summary, and test predictions before tracking them.
Each candidate run records model parameters and seed; dataset name, DOI,
archive checksum/version; target mapping; feature recipe and count; cycle
range/class counts by split; split and validation methodology; the 60-cycle
boundary purge and embargo; the validation-selected threshold; and training,
validation, and test metrics under separate metric namespaces. Confusion
matrix cells are logged as metrics, and the model plus comparison JSON,
metadata, and test prediction table are logged as artifacts.

Runs use stable `comparison_id` and `candidate` tags. A completed candidate
run is reused on repeated logging, missing candidates can be added after a
partial logging failure, and an existing failed/running record or duplicate
candidate records produce an explicit error rather than silently creating a
second run. A changed comparison must use a new comparison ID; comparison
output directories themselves refuse overwrites.

Core tags include:

- `dataset=uci-hydraulic-condition-monitoring`
- `task=binary-pump-leakage-classification`
- `stage=model_comparison`
- `evaluation=chronological`
- `comparison_id=<saved comparison ID>`
- `candidate=<model family>`

Validation metrics use `validation.*`; untouched chronological test metrics
use `test.*`. The decision threshold is selected only on validation data and
is separately recorded. ROC-AUC is omitted when unavailable. Probabilities
are not logged as calibrated confidence. Run tags and comparison artifacts
state that the observations are sequential, operating conditions shift, and
the dataset lacks independent run/rig identifiers. Perfect retrospective
scores do not establish independent-run or real-world generalization.

## Inspect runs

Open the local tracking UI from the repository root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Or use the MLflow Python client:

```python
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("hydraulic-condition-monitoring")
runs = client.search_runs([experiment.experiment_id], max_results=100)
for run in runs:
    print(run.info.run_name, run.data.tags, run.data.metrics)
```

Set `MLFLOW_TRACKING_URI` to another supported MLflow URI to use a different
tracking backend. Unit tests create isolated temporary SQLite stores and do
not access the project database.

The historical Wine tracking data was backed up outside the repository before
its removal from the active store. The archive covered 14 scoped run IDs and
the experiment's logged-model artifact directory (68 files total) and passed
ZIP CRC validation. The legacy Wine DVC stages and local model/report outputs
remain available, but those scripts no longer write Wine runs to MLflow.
`dvc.yaml`, `.dvc/config`, and DVC-tracked data were not changed by the
tracking migration.

## Historical tracking data

Local tracking history is deliberately not distributed with the source
repository. No setup command deletes or migrates an existing database. Keep
private database backups outside Git and do not replace an existing tracking
store as part of setup. The retained Wine scripts and DVC metadata are a
separate legacy workflow and do not write to this hydraulic experiment.
