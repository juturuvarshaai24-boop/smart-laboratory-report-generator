"""Tests for safe, explicit hydraulic dataset archive setup."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.setup_hydraulic_dataset import validate_archive
from src.hydraulic.data import DatasetIntegrityError, load_hydraulic_dataset


class HydraulicDatasetSetupTests(unittest.TestCase):
    def test_missing_dataset_points_to_setup_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "extracted"
            with self.assertRaisesRegex(
                DatasetIntegrityError, "scripts/setup_hydraulic_dataset.py"
            ):
                load_hydraulic_dataset(data_dir)

    def test_verified_archive_contract_accepts_expected_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile.txt", b"profile")
                archive.writestr("PS1.txt", b"sensor")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

            self.assertEqual(
                validate_archive(
                    archive_path,
                    expected_sha256=digest,
                    expected_files={"profile.txt": 7, "PS1.txt": 6},
                ),
                digest,
            )

    def test_archive_digest_and_members_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile.txt", b"profile")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(DatasetIntegrityError, "SHA-256 mismatch"):
                validate_archive(
                    archive_path,
                    expected_sha256="0" * 64,
                    expected_files={"profile.txt": 7},
                )
            with self.assertRaisesRegex(DatasetIntegrityError, "unexpected UCI archive"):
                validate_archive(
                    archive_path,
                    expected_sha256=digest,
                    expected_files={"profile.txt": 7, "PS1.txt": 6},
                )

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../profile.txt", b"profile")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(DatasetIntegrityError, "unsafe path"):
                validate_archive(
                    archive_path,
                    expected_sha256=digest,
                    expected_files={"../profile.txt": 7},
                )


if __name__ == "__main__":
    unittest.main()