"""Run-specific output paths and dataset artifacts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import HydraulicDataset

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def new_run_id() -> str:
    """Return a UTC timestamp run ID with microsecond precision."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def validate_run_id(run_id: str) -> str:
    """Reject run IDs that could escape or alias a run directory."""
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(f"invalid run ID: {run_id!r}")
    return run_id


def run_paths(project_root: Path, run_id: str) -> dict[str, Path]:
    """Return isolated, collision-safe artifact locations for a run."""
    safe_id = validate_run_id(run_id)
    return {
        "model_dir": project_root / "models" / "hydraulic" / "runs" / safe_id,
        "metrics_dir": project_root / "metrics" / "hydraulic" / "runs" / safe_id,
        "processed_dir": project_root
        / "data"
        / "processed"
        / "hydraulic"
        / "runs"
        / safe_id,
    }


def create_run_directories(paths: dict[str, Path]) -> None:
    """Create new run directories and refuse to reuse existing output paths."""
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing hydraulic run outputs: "
            + ", ".join(existing)
        )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)


def split_membership_frame(
    cycle_ids: Any,
    splits: dict[str, Any],
) -> pd.DataFrame:
    """Build the canonical row-to-partition table for saved and checked runs."""
    row_positions: list[int] = []
    split_names: list[str] = []
    for split_name in ("train", "validation", "test", "purged"):
        indices = np.asarray(getattr(splits, split_name), dtype=np.int64)
        split_names.extend([split_name] * len(indices))
        row_positions.extend(int(index) for index in indices)
    positions = np.asarray(row_positions, dtype=np.int64)
    ids = np.asarray(cycle_ids, dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= len(ids)):
        raise ValueError("split membership contains an out-of-range row position")
    if len(np.unique(positions)) != len(ids) or len(positions) != len(ids):
        raise ValueError("split membership must assign each cycle exactly once")
    return pd.DataFrame(
        {
            "split": split_names,
            "row_position_zero_based": positions,
            "cycle_id_one_based": ids[positions],
        }
    )


def validate_split_membership_file(
    path: Path,
    cycle_ids: Any,
    splits: dict[str, Any],
) -> None:
    """Reject changed, misaligned, missing, or duplicated saved split rows."""
    actual = pd.read_csv(path)
    expected = split_membership_frame(cycle_ids, splits)
    if not actual.equals(expected):
        raise ValueError("saved split membership does not match the fixed cycle split")


def save_dataset_artifacts(
    dataset: HydraulicDataset,
    processed_dir: Path,
    splits: dict[str, Any],
) -> None:
    """Persist features and profile labels in separate, run-specific files."""
    feature_table = dataset.features.copy()
    feature_table.insert(0, "cycle_id", dataset.cycle_ids)
    feature_table.to_csv(processed_dir / "cycle_features.csv", index=False)

    targets = dataset.profile_targets.copy()
    targets.insert(0, "cycle_id", dataset.cycle_ids)
    targets["pump_leakage_binary"] = dataset.pump_leakage_binary.to_numpy()
    targets.to_csv(processed_dir / "profile_targets.csv", index=False)
    split_membership_frame(dataset.cycle_ids, splits).to_csv(
        processed_dir / "split_membership.csv", index=False
    )
