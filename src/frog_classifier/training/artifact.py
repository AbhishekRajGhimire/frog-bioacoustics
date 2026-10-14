"""The saved run folder: staged creation, checksums, and reading back."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_SCHEMA_VERSION = 1
RUN_METADATA = "run.json"
MODEL_FILE = "model.pt"
PREDICTIONS_FILE = "predictions.csv"
PREDICTION_COLUMNS = (
    "example_id", "image_path", "image_sha256", "recording_id", "fold", "split",
    "label", "faint", "probability", "baseline_probability",
)
_STAGING_SUFFIX = ".partial"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_run(models_root: Path, name: str) -> Path:
    """Create the staging folder for a new run. Refuses a name already used or staged,
    so a run never overwrites another and an interrupted run never looks finished."""
    final = models_root / name
    staging = models_root / f".{name}{_STAGING_SUFFIX}"
    for path in (final, staging):
        if path.exists():
            raise FileExistsError(f"run folder already exists: {path}")
    staging.mkdir(parents=True)
    return staging


def finish_run(staging: Path, metadata: Mapping[str, Any]) -> Path:
    """Checksum every file, write run.json, and move the folder to its final name."""
    if not (staging.name.startswith(".") and staging.name.endswith(_STAGING_SUFFIX)):
        raise ValueError(f"not a staging folder: {staging}")
    files = {
        path.name: file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != RUN_METADATA
    }
    write_json(staging / RUN_METADATA, {**metadata, "files": files})
    final = staging.with_name(staging.name[1 : -len(_STAGING_SUFFIX)])
    staging.rename(final)
    return final


def read_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_METADATA).read_text(encoding="utf-8"))


def check_files(run_dir: Path, metadata: Mapping[str, Any]) -> list[str]:
    problems = []
    for name, expected in sorted(metadata["files"].items()):
        path = run_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing from the run folder")
        elif file_sha256(path) != expected:
            problems.append(f"{name}: checksum differs from run.json")
    return problems


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
