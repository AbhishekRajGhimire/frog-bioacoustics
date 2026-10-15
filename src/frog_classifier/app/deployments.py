"""Deployments: named scans of one folder of recordings, each with its own detection record."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path, PurePosixPath

from frog_classifier.files import replace_file
from frog_classifier.inference.recordings import find_recordings, start_time_from_name
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, StoreError

from .models import ModelInfo
from .paths import AppPaths
from .scans import (
    ACTIVE_STATES,
    ScanStatus,
    clear_stop_request,
    current_status,
    launch_scan_process,
    read_status,
    utc_timestamp,
    write_status,
)


DEPLOYMENT_FILE = "deployment.json"
NAME_LIMIT = 80
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9 _.()-]+")
_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
})


class DeploymentError(ValueError):
    pass


@dataclass(frozen=True)
class Deployment:
    name: str
    record_dir: Path
    input_root: str
    model_run: str
    created_at: str
    status: ScanStatus | None = None

    @property
    def scanning(self) -> bool:
        return self.status is not None and self.status.state in ACTIVE_STATES

    @property
    def results_ready(self) -> bool:
        """Whether the record holds complete results; a record made by the detection command has no status."""
        if self.status is None:
            return True
        return not self.scanning and self.status.results_ready


@dataclass(frozen=True)
class RecordingsPreview:
    count: int
    first_day: date | None
    last_day: date | None
    undated: int


def clean_name(text: str) -> str:
    """A deployment name that is safe as part of a folder name on every system."""
    # Windows drops trailing dots and spaces from folder names.
    name = _UNSAFE_CHARACTERS.sub("-", text).strip(" .")
    if not name:
        raise DeploymentError("give these recordings a name, such as the site and card number")
    if len(name) > NAME_LIMIT:
        raise DeploymentError(f"keep the name to {NAME_LIMIT} characters or fewer")
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        raise DeploymentError(f"{name!r} is reserved by Windows; choose another name")
    return name


def deployment_dir(detections_root: Path, name: str, model_name: str) -> Path:
    return detections_root / f"{clean_name(name)}-{model_name}"


def list_deployments(detections_root: Path, *, now: datetime) -> list[Deployment]:
    """Every deployment with a record under detections_root, newest first."""
    if not detections_root.is_dir():
        return []
    deployments = []
    for record_dir in detections_root.iterdir():
        deployment = _read_deployment(record_dir) if record_dir.is_dir() else None
        if deployment is not None:
            deployments.append(replace(deployment, status=current_status(record_dir, now)))
    return sorted(deployments, key=lambda deployment: deployment.created_at, reverse=True)


def live_scan(deployments: list[Deployment]) -> Deployment | None:
    return next((deployment for deployment in deployments if deployment.scanning), None)


def preview_recordings(folder: Path) -> RecordingsPreview:
    """How many recordings a folder holds, and the dates their file names give."""
    if not folder.is_dir():
        raise DeploymentError(f"there is no folder {folder}")
    paths = find_recordings(folder)
    days = [
        start.date()
        for path in paths
        if (start := start_time_from_name(PurePosixPath(path).stem)) is not None
    ]
    return RecordingsPreview(len(paths), min(days, default=None), max(days, default=None), len(paths) - len(days))


def start_scan(
    paths: AppPaths,
    model: ModelInfo,
    folder: Path,
    name: str,
    *,
    now: datetime,
    workers: int | None = None,
) -> Path:
    """Start scanning folder as the deployment called name, or continue that deployment.

    Returns the deployment's record folder; the scan runs in its own process.
    """
    name = clean_name(name)
    folder = folder.resolve()
    if not folder.is_dir():
        raise DeploymentError(f"there is no folder {folder}")
    deployments = list_deployments(paths.detections, now=now)
    running = live_scan(deployments)
    if running is not None:
        raise DeploymentError(f"the scan of {running.name} is still running; wait for it to finish or stop it first")
    record_dir = deployment_dir(paths.detections, name, model.name)
    existing = next((deployment for deployment in deployments if deployment.record_dir == record_dir), None)
    if existing is not None and Path(existing.input_root) != folder:
        raise DeploymentError(
            f"the name {name!r} is already used for the recordings in {existing.input_root}; choose another name"
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    if existing is None:
        _write_deployment(Deployment(name, record_dir, folder.as_posix(), model.name, utc_timestamp(now)))
    previous = read_status(record_dir)
    started = utc_timestamp(now)
    clear_stop_request(record_dir)
    write_status(record_dir, ScanStatus(
        state="starting",
        started_at=started,
        updated_at=started,
        results_ready=previous.results_ready if previous is not None else (record_dir / DATABASE_FILE).is_file(),
    ))
    launch_scan_process(model.run_dir, folder, record_dir, workers=workers)
    return record_dir


def _read_deployment(record_dir: Path) -> Deployment | None:
    """The deployment the app recorded, or else one made by the detection command."""
    try:
        data = json.loads((record_dir / DEPLOYMENT_FILE).read_bytes())
        return Deployment(record_dir=record_dir, **data)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        return None
    if not (record_dir / DATABASE_FILE).is_file():
        return None
    try:
        with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
            identity, created_at = store.identity, store.created_at
    except StoreError:
        return None
    name = record_dir.name.removesuffix(f"-{identity.model_run}")
    return Deployment(name, record_dir, identity.input_root, identity.model_run, created_at)


def _write_deployment(deployment: Deployment) -> None:
    content = json.dumps({
        "name": deployment.name,
        "input_root": deployment.input_root,
        "model_run": deployment.model_run,
        "created_at": deployment.created_at,
    }, indent=2).encode("utf-8")
    replace_file(deployment.record_dir / DEPLOYMENT_FILE, lambda staging: staging.write_bytes(content))
