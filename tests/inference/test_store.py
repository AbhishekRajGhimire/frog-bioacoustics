from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, ScanIdentity, StoreError
from tests.inference.helpers import SCAN_IDENTITY, recording_row


class DetectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / DATABASE_FILE

    def _open(self, identity: ScanIdentity = SCAN_IDENTITY, now: str = "2026-10-15T00:00:00Z") -> DetectionStore:
        store = DetectionStore.open(self.path, identity, now=now)
        self.addCleanup(store.close)
        return store

    def _execute(self, statement: str) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(statement)
            connection.commit()
        finally:
            connection.close()

    def test_a_reopened_database_keeps_its_identity_and_creation_time(self) -> None:
        self._open().close()

        reopened = self._open(now="2026-10-16T00:00:00Z")

        self.assertEqual(reopened.identity, SCAN_IDENTITY)
        self.assertEqual(reopened.created_at, "2026-10-15T00:00:00Z")

    def test_a_database_made_for_another_scan_is_refused(self) -> None:
        self._open().close()
        changes = {
            "input_root": "E:/elsewhere",
            "model_sha256": "e" * 64,
            "threshold": 0.75,
            "preprocessing_sha256": "f" * 64,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(StoreError, f"different {field}; choose another output folder"):
                    DetectionStore.open(self.path, replace(SCAN_IDENTITY, **{field: value}), now="later")

    def test_a_database_from_another_schema_version_is_refused(self) -> None:
        self._open().close()
        self._execute("UPDATE scan SET schema_version = 2")

        with self.assertRaisesRegex(StoreError, "schema 2.*; choose another output folder"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_a_file_that_is_not_a_database_is_refused(self) -> None:
        self.path.write_bytes(b"these bytes are not a database " * 40)

        with self.assertRaisesRegex(StoreError, "is not a detection database.*; choose another output folder"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_another_sqlite_database_is_refused(self) -> None:
        self._execute("CREATE TABLE other (value INTEGER)")

        with self.assertRaisesRegex(StoreError, "is not a detection database"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_a_read_only_store_takes_its_identity_from_the_database_and_cannot_write(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))
        store.close()
        before = self.path.read_bytes()

        with DetectionStore.open_read_only(self.path) as reader:
            self.assertEqual(reader.identity, SCAN_IDENTITY)
            self.assertEqual(reader.created_at, "2026-10-15T00:00:00Z")
            self.assertEqual(reader.recording_rows(), [recording_row("site/a.wav")])
            with self.assertRaises(sqlite3.OperationalError):
                reader.save(recording_row("site/b.wav"), (0, 5), (0.1, 0.2))

        self.assertEqual(self.path.read_bytes(), before)

    def test_reading_a_missing_database_says_to_run_detection_first(self) -> None:
        with self.assertRaisesRegex(StoreError, "run the detection command first"):
            DetectionStore.open_read_only(self.path)

    def test_reading_a_database_that_is_mid_write_is_refused(self) -> None:
        self._open().close()
        self.path.with_name(f"{self.path.name}-journal").write_bytes(bytes(512))

        with self.assertRaisesRegex(StoreError, "still writing .* or was interrupted"):
            DetectionStore.open_read_only(self.path)

    def test_reading_a_file_that_is_not_a_detection_database_is_refused(self) -> None:
        for name, make in (
            ("not a database", lambda: self.path.write_bytes(b"these bytes are not a database " * 40)),
            ("another database", lambda: self._execute("CREATE TABLE other (value INTEGER)")),
        ):
            with self.subTest(file=name):
                self.path.unlink(missing_ok=True)
                make()
                with self.assertRaisesRegex(StoreError, "is not a detection database"):
                    DetectionStore.open_read_only(self.path)

    def test_a_recording_is_stored_with_all_of_its_windows(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))
        store.close()

        reopened = self._open()

        self.assertEqual(reopened.recording_rows(), [recording_row("site/a.wav")])
        self.assertEqual(reopened.windows(), [("site/a.wav", 0, 0.25), ("site/a.wav", 5, 0.75)])
        self.assertEqual(reopened.finished_recordings(), {"site/a.wav": "d" * 64})

    def test_a_save_that_fails_partway_stores_nothing(self) -> None:
        store = self._open()

        with self.assertRaises(sqlite3.IntegrityError):
            store.save(recording_row("site/a.wav"), (0, 0), (0.25, 0.75))

        self.assertEqual(store.recording_rows(), [])
        self.assertEqual(store.windows(), [])

    def test_a_retried_recording_replaces_its_failed_attempt(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav", windows=0, status="failed", message="OSError: busy"), (), ())
        self.assertEqual(store.finished_recordings(), {})

        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))

        self.assertEqual(store.recording_rows(), [recording_row("site/a.wav")])
        self.assertEqual(len(store.windows()), 2)

    def test_windows_can_be_limited_to_a_minimum_probability(self) -> None:
        store = self._open()
        store.save(recording_row("site/b.wav", windows=3), (0, 5, 10), (0.2, 0.5, 0.9))
        store.save(recording_row("site/a.wav", windows=1), (0,), (0.6,))

        self.assertEqual(
            store.windows(minimum_probability=0.5),
            [("site/a.wav", 0, 0.6), ("site/b.wav", 5, 0.5), ("site/b.wav", 10, 0.9)],
        )

    def test_a_window_count_that_disagrees_with_the_windows_is_rejected(self) -> None:
        store = self._open()

        with self.assertRaises(ValueError):
            store.save(recording_row("site/a.wav", windows=3), (0, 5), (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
