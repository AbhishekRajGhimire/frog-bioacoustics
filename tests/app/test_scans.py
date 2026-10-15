from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.scans import (
    STATUS_FILE,
    ScanStatus,
    clear_stop_request,
    current_status,
    read_status,
    request_stop,
    stop_requested,
    write_status,
)
from frog_classifier.files import FileInUseError, replace_file


MOMENT = datetime(2026, 10, 15, tzinfo=timezone.utc)


def status(**changes: object) -> ScanStatus:
    base = ScanStatus(
        state="running",
        started_at="2026-10-15T00:00:00Z",
        updated_at="2026-10-15T00:00:00Z",
        results_ready=False,
        found=10,
        already_done=2,
        processed=3,
    )
    return replace(base, **changes)


class ScanStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = Path(directory.name)

    def test_a_written_status_reads_back_unchanged(self) -> None:
        write_status(self.record, status(device="cuda"))

        self.assertEqual(read_status(self.record), status(device="cuda"))
        self.assertEqual(read_status(self.record).done, 5)

    def test_a_missing_or_damaged_status_reads_as_none(self) -> None:
        self.assertIsNone(read_status(self.record))
        (self.record / STATUS_FILE).write_bytes(b"{not json")
        self.assertIsNone(read_status(self.record))

    def test_an_active_status_that_stopped_updating_reads_as_interrupted(self) -> None:
        write_status(self.record, status())

        self.assertEqual(current_status(self.record, MOMENT + timedelta(seconds=29)).state, "running")
        self.assertEqual(current_status(self.record, MOMENT + timedelta(seconds=31)).state, "interrupted")
        write_status(self.record, status(state="finished"))
        self.assertEqual(current_status(self.record, MOMENT + timedelta(hours=1)).state, "finished")

    def test_a_status_being_swapped_in_is_read_again(self) -> None:
        write_status(self.record, status())
        content = (self.record / STATUS_FILE).read_bytes()

        with patch.object(Path, "read_bytes", side_effect=[PermissionError(13, "in use"), content]):
            with patch("frog_classifier.app.scans.time.sleep"):
                self.assertEqual(read_status(self.record), status())

    def test_a_status_file_held_open_is_written_again(self) -> None:
        attempts = []

        def held_open_twice(path, write):
            attempts.append(path)
            if len(attempts) < 3:
                raise FileInUseError("in use")
            replace_file(path, write)

        with patch("frog_classifier.app.scans.replace_file", side_effect=held_open_twice):
            with patch("frog_classifier.app.scans.time.sleep"):
                write_status(self.record, status())

        self.assertEqual(len(attempts), 3)
        self.assertEqual(read_status(self.record), status())

    def test_a_stop_request_is_seen_and_cleared(self) -> None:
        self.assertFalse(stop_requested(self.record))
        request_stop(self.record)
        self.assertTrue(stop_requested(self.record))
        clear_stop_request(self.record)
        self.assertFalse(stop_requested(self.record))


if __name__ == "__main__":
    unittest.main()
