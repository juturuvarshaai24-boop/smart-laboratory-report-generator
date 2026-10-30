"""Download, verify, and safely extract the official UCI hydraulic dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hydraulic.data import (
    ARCHIVE_SHA256,
    DATASET_DOI,
    DATASET_URL,
    DEFAULT_DATA_DIR,
    PROFILE_FILE_BYTES,
    SENSOR_SPECS,
    DatasetIntegrityError,
    sha256_file,
)

ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition+monitoring+of+hydraulic+systems.zip"
)
ARCHIVE_PATH = DEFAULT_DATA_DIR.parent / "uci_dataset_447.zip"
EXPECTED_ARCHIVE_FILES = {
    **{f"{name}.txt": int(spec["bytes"]) for name, spec in SENSOR_SPECS.items()},
    "profile.txt": PROFILE_FILE_BYTES,
    "description.txt": 4044,
    "documentation.txt": 4719,
}


def validate_archive(
    archive_path: Path,
    *,
    expected_sha256: str = ARCHIVE_SHA256,
    expected_files: dict[str, int] = EXPECTED_ARCHIVE_FILES,
) -> str:
    """Check digest, exact member names/sizes, safe paths, and ZIP CRCs."""
    archive_path = Path(archive_path)
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise DatasetIntegrityError(
            f"UCI archive SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise DatasetIntegrityError("UCI archive contains duplicate file names")
            for entry in entries:
                posix_path = PurePosixPath(entry.filename)
                windows_path = PureWindowsPath(entry.filename)
                if (
                    entry.is_dir()
                    or len(posix_path.parts) != 1
                    or posix_path.name in {"", ".", ".."}
                    or windows_path.is_absolute()
                    or windows_path.drive
                    or "\\" in entry.filename
                ):
                    raise DatasetIntegrityError(
                        f"unsafe path in UCI archive: {entry.filename!r}"
                    )
            if set(names) != set(expected_files):
                missing = sorted(set(expected_files) - set(names))
                unexpected = sorted(set(names) - set(expected_files))
                raise DatasetIntegrityError(
                    f"unexpected UCI archive contents; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            for entry in entries:
                expected_size = expected_files[entry.filename]
                if entry.file_size != expected_size:
                    raise DatasetIntegrityError(
                        f"{entry.filename} has {entry.file_size} bytes; "
                        f"expected {expected_size}"
                    )
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise DatasetIntegrityError(
                    f"UCI archive CRC check failed for {corrupt_member}"
                )
    except zipfile.BadZipFile as exc:
        raise DatasetIntegrityError(f"invalid UCI ZIP archive: {exc}") from exc
    return actual_sha256


def _verify_extracted_files(archive_path: Path, extracted_dir: Path) -> None:
    """Require an existing extraction to exactly match the verified archive."""
    if not extracted_dir.is_dir():
        raise DatasetIntegrityError(f"extraction directory is missing: {extracted_dir}")
    actual_names = {path.name for path in extracted_dir.iterdir() if path.is_file()}
    expected_names = set(EXPECTED_ARCHIVE_FILES)
    if actual_names != expected_names or any(
        path.is_dir() for path in extracted_dir.iterdir()
    ):
        raise DatasetIntegrityError(
            f"existing extraction contents differ from the archive: {extracted_dir}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(expected_names):
            source_path = extracted_dir / name
            if source_path.stat().st_size != EXPECTED_ARCHIVE_FILES[name]:
                raise DatasetIntegrityError(
                    f"existing extracted file has unexpected size: {source_path}"
                )
            with archive.open(name) as source, source_path.open("rb") as target:
                while True:
                    source_chunk = source.read(1024 * 1024)
                    target_chunk = target.read(1024 * 1024)
                    if source_chunk != target_chunk:
                        raise DatasetIntegrityError(
                            f"existing extracted file differs from archive: {source_path}"
                        )
                    if not source_chunk:
                        break


def _download_archive(archive_path: Path) -> None:
    """Download to a temporary sibling and publish only after digest checks."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        validate_archive(archive_path)
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="hydraulic-dataset-",
            suffix=".download",
            dir=archive_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = urllib.request.Request(
                ARCHIVE_URL,
                headers={"User-Agent": "Smart-Laboratory-Report-Generator/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary_file, length=1024 * 1024)
        validate_archive(temporary_path)
        temporary_path.rename(archive_path)
        temporary_path = None
    except OSError as exc:
        raise RuntimeError(
            f"could not download the UCI dataset from {ARCHIVE_URL}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def setup_dataset(archive_path: Path, extracted_dir: Path) -> str:
    """Verify or download the archive and extract it without overwriting files."""
    archive_path = Path(archive_path)
    extracted_dir = Path(extracted_dir)
    if not archive_path.is_file():
        _download_archive(archive_path)
    digest = validate_archive(archive_path)

    if extracted_dir.exists():
        _verify_extracted_files(archive_path, extracted_dir)
        return digest

    extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hydraulic-extract-", dir=extracted_dir.parent
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary_path)
        _verify_extracted_files(archive_path, temporary_path)
        temporary_path.rename(extracted_dir)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    try:
        digest = setup_dataset(args.archive, args.extracted_dir)
    except (DatasetIntegrityError, RuntimeError, OSError) as exc:
        print(f"Dataset setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: UCI Condition Monitoring of Hydraulic Systems ({DATASET_DOI})")
    print(f"Source: {DATASET_URL}")
    print(f"Archive SHA-256: {digest}")
    print(f"Extracted files: {args.extracted_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())