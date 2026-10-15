"""Scan processes and the status file through which they report.

A scan runs in its own process, so it survives a page refresh and keeps PyTorch
out of the app. It reports through scan_status.json in its record folder, and a
stop is requested by creating stop_requested there.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from frog_classifier.files import FileInUseError, replace_file


STATUS_FILE = "scan_status.json"
STOP_FILE = "stop_requested"
LOG_FILE = "scan.log"
HEARTBEAT_SECONDS = 2.0
# A starting or running scan rewrites its status every two seconds, so one this
# old belongs to a process that ended without saying so.
STALE_AFTER = timedelta(seconds=30)
ACTIVE_STATES = frozenset({"starting", "running"})
_ATTEMPTS = 50
_RETRY_SECONDS = 0.1

# Kept so a scan process still running is never reported as abandoned.
_launched: list[subprocess.Popen] = []


@dataclass(frozen=True)
class ScanStatus:
    """A scan's progress; results_ready says whether the record holds complete results."""

    state: str
    started_at: str
    updated_at: str
    results_ready: bool
    found: int = 0
    already_done: int = 0
    processed: int = 0
    failed: int = 0
    device: str | None = None
    message: str | None = None

    @property
    def done(self) -> int:
        return self.already_done + self.processed


def utc_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(record_dir: Path) -> ScanStatus | None:
    """The last status written in record_dir, or None when there is none."""
    path = record_dir / STATUS_FILE
    for attempt in range(_ATTEMPTS):
        try:
            content = path.read_bytes()
            break
        except FileNotFoundError:
            return None
        except PermissionError:
            # The scan process is swapping in a new status at this moment.
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_SECONDS)
    try:
        return ScanStatus(**json.loads(content))
    except (ValueError, TypeError):
        return None


def write_status(record_dir: Path, status: ScanStatus) -> None:
    """Swap in a new status, retrying while a reader holds the old one open."""
    content = json.dumps(asdict(status), indent=2).encode("utf-8")
    for attempt in range(_ATTEMPTS):
        try:
            replace_file(record_dir / STATUS_FILE, lambda staging: staging.write_bytes(content))
            return
        except FileInUseError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_SECONDS)


def current_status(record_dir: Path, now: datetime) -> ScanStatus | None:
    """The status as it stands at now; an active scan that stopped reporting is interrupted."""
    status = read_status(record_dir)
    if status is not None and status.state in ACTIVE_STATES:
        if now - datetime.fromisoformat(status.updated_at) > STALE_AFTER:
            return replace(status, state="interrupted")
    return status


def stop_requested(record_dir: Path) -> bool:
    return (record_dir / STOP_FILE).exists()


def request_stop(record_dir: Path) -> None:
    (record_dir / STOP_FILE).touch()


def clear_stop_request(record_dir: Path) -> None:
    (record_dir / STOP_FILE).unlink(missing_ok=True)


def launch_scan_process(
    run_dir: Path,
    input_root: Path,
    record_dir: Path,
    *,
    workers: int | None = None,
) -> subprocess.Popen:
    """Start the scan process; its output goes to scan.log in record_dir."""
    command = [sys.executable, "-m", "frog_classifier.app.scan_job", str(run_dir), str(input_root), str(record_dir)]
    if workers is not None:
        command += ["--workers", str(workers)]
    with (record_dir / LOG_FILE).open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    _launched[:] = [*(running for running in _launched if running.poll() is None), process]
    return process
