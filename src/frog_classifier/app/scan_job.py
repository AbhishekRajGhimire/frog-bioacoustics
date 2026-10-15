"""The scan process: detect calls in one folder and keep its status file current.

The app starts it as python -m frog_classifier.app.scan_job RUN INPUT RECORD.
PyTorch loads only after the status thread is running, so the app sees the scan
within seconds of starting it.
"""

from __future__ import annotations

import _thread
import argparse
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from frog_classifier.files import FileInUseError
from frog_classifier.inference.store import DATABASE_FILE

from .scans import HEARTBEAT_SECONDS, ScanStatus, read_status, stop_requested, utc_timestamp, write_status


STOPPED = "Stopped on request; finished recordings are kept."
_EXIT_CODES = {"finished": 0, "stopped": 130}


class ScanStopped(Exception):
    """A stop was requested between two recordings."""


class StatusFile:
    """The scan's status, rewritten on every change; safe to update from two threads."""

    def __init__(self, record_dir: Path, status: ScanStatus) -> None:
        self.record_dir = record_dir
        self._status = status
        self._lock = threading.Lock()
        write_status(record_dir, status)

    @property
    def status(self) -> ScanStatus:
        with self._lock:
            return self._status

    def update(self, **changes: object) -> None:
        with self._lock:
            self._status = replace(self._status, **changes, updated_at=utc_timestamp(datetime.now(timezone.utc)))
            write_status(self.record_dir, self._status)

    def beat(self) -> None:
        """Show that the scan is alive; a reader holding the file only delays this to the next beat."""
        try:
            self.update()
        except FileInUseError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("run", type=Path, help="Saved model run folder.")
    parser.add_argument("input", type=Path, help="Folder of recordings.")
    parser.add_argument("record", type=Path, help="The deployment's detection record folder.")
    parser.add_argument("--workers", type=int, default=None, help="Rendering processes. Default: as the detection command.")
    args = parser.parse_args(argv)

    previous = read_status(args.record)
    started = utc_timestamp(datetime.now(timezone.utc))
    status = StatusFile(args.record, ScanStatus(
        state="starting",
        started_at=started,
        updated_at=started,
        results_ready=previous.results_ready if previous is not None else (args.record / DATABASE_FILE).is_file(),
    ))
    finished = threading.Event()
    heartbeat = threading.Thread(target=_keep_alive, args=(status, finished), daemon=True)
    heartbeat.start()
    try:
        try:
            outcome = _scan(args, status)
        finally:
            finished.set()
            heartbeat.join()
    except KeyboardInterrupt:
        outcome = {"state": "stopped", "message": STOPPED}
    except Exception as error:
        traceback.print_exc()
        outcome = {"state": "failed", "message": f"{type(error).__name__}: {error}"}
    status.update(**outcome)
    return _EXIT_CODES.get(str(outcome["state"]), 1)


def _scan(args: argparse.Namespace, status: StatusFile) -> dict[str, object]:
    # Imported here: the status thread is already running while PyTorch loads.
    from frog_classifier.inference.detect import DetectionError, default_workers, detect_calls
    from frog_classifier.inference.store import StoreError
    from frog_classifier.training.fit import select_device
    from frog_classifier.training.saved import SavedRunError

    device = select_device()
    status.update(device=device.type)

    def started(pending: int, done: int) -> None:
        # The record is complete only while nothing new is waiting to be scanned.
        status.update(
            state="running",
            found=pending + done,
            already_done=done,
            processed=0,
            failed=0,
            results_ready=status.status.results_ready and pending == 0,
        )

    def stored(outcome) -> None:
        current = status.status
        status.update(processed=current.processed + 1, failed=current.failed + int(outcome.failed))
        if stop_requested(status.record_dir):
            raise ScanStopped

    try:
        summary = detect_calls(
            args.run,
            args.input,
            args.record,
            device=device,
            workers=args.workers or default_workers(),
            on_start=started,
            on_recording=stored,
        )
    except ScanStopped:
        return {"state": "stopped", "message": STOPPED}
    except (DetectionError, SavedRunError, StoreError) as error:
        # Refused before anything was scanned, so the record is as complete as before.
        return {"state": "failed", "message": str(error)}
    return {"state": "finished", "results_ready": True, "failed": summary.failed, "message": summary.csv_problem}


def _keep_alive(status: StatusFile, finished: threading.Event) -> None:
    interrupted = False
    while not finished.wait(HEARTBEAT_SECONDS):
        status.beat()
        if not interrupted and stop_requested(status.record_dir):
            # Reaches a long render too; the main thread treats it like Ctrl+C.
            interrupted = True
            _thread.interrupt_main()


if __name__ == "__main__":
    raise SystemExit(main())
