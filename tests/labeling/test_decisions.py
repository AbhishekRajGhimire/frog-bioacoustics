from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from frog_classifier.labeling.decisions import (
    LOG_COLUMNS,
    DecisionLog,
    confirm_decision,
    current_folder,
    record_decision,
    summarize_labels,
    undo_move,
)


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labeled_root = self.root / "labeled"
        self.queue_root = self.root / "queue"
        self.log = DecisionLog(self.labeled_root / "decisions.csv")
        self.clip = self._write(self.queue_root / "site" / "20220101_220000_start5s.png")

    def test_label_moves_the_clip_and_logs_a_label_row(self) -> None:
        move = record_decision(self.clip, "litoria_aurea", labeled_root=self.labeled_root, log=self.log, faint=True)

        self.assertFalse(self.clip.exists())
        self.assertEqual(move.destination, self.labeled_root / "litoria_aurea" / self.clip.name)
        self.assertTrue(move.destination.is_file())
        rows = self.log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].example_id, rows[0].decision, rows[0].action, rows[0].faint), ("20220101_220000_start5s", "litoria_aurea", "label", True))
        self.assertRegex(rows[0].decided_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_log_is_created_with_a_header_once(self) -> None:
        self.log.append("a_start0s", "unsure", "label")
        self.log.append("b_start0s", "non_target", "label")

        with self.log.path.open(newline="", encoding="utf-8") as handle:
            lines = list(csv.reader(handle))
        self.assertEqual(lines[0], list(LOG_COLUMNS))
        self.assertEqual(len(lines), 3)

    def test_log_header_is_written_when_the_file_exists_but_is_empty(self) -> None:
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        self.log.path.write_text("", encoding="utf-8")

        self.log.append("a_start0s", "unsure", "label")

        with self.log.path.open(newline="", encoding="utf-8") as handle:
            first_line = next(csv.reader(handle))
        self.assertEqual(first_line, list(LOG_COLUMNS))

    def test_change_moves_between_labeled_folders_and_logs_change(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        changed = record_decision(move.destination, "unsure", labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(changed.destination, self.labeled_root / "unsure" / self.clip.name)
        self.assertFalse(move.destination.exists())
        self.assertEqual([row.action for row in self.log.rows()], ["label", "change"])

    def test_confirm_logs_without_moving(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        row = confirm_decision(move.destination, labeled_root=self.labeled_root, log=self.log)

        self.assertTrue(move.destination.is_file())
        self.assertEqual((row.decision, row.action), ("non_target", "confirm"))

    def test_confirm_leaves_the_faint_flag_unrecorded_unless_reaffirmed(self) -> None:
        move = record_decision(self.clip, "litoria_aurea", labeled_root=self.labeled_root, log=self.log, faint=True)

        confirm_decision(move.destination, labeled_root=self.labeled_root, log=self.log)
        confirm_decision(move.destination, labeled_root=self.labeled_root, log=self.log, faint=True)

        self.assertEqual([row.faint for row in self.log.rows()], [True, None, True])

    def test_faint_confirm_is_refused_outside_the_frog_folder(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        with self.assertRaisesRegex(ValueError, "faint"):
            confirm_decision(move.destination, labeled_root=self.labeled_root, log=self.log, faint=True)

    def test_moving_into_the_current_folder_is_refused(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        with self.assertRaisesRegex(ValueError, "confirm"):
            record_decision(move.destination, "non_target", labeled_root=self.labeled_root, log=self.log)
        self.assertEqual(len(self.log.rows()), 1)

    def test_existing_destination_is_never_overwritten(self) -> None:
        existing = self._write(self.labeled_root / "litoria_aurea" / self.clip.name, b"existing")

        with self.assertRaises(FileExistsError):
            record_decision(self.clip, "litoria_aurea", labeled_root=self.labeled_root, log=self.log)
        self.assertTrue(self.clip.exists())
        self.assertEqual(existing.read_bytes(), b"existing")
        self.assertFalse(self.log.path.exists())

    def test_log_failure_moves_the_clip_back(self) -> None:
        with mock.patch.object(self.log, "append", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                record_decision(self.clip, "unsure", labeled_root=self.labeled_root, log=self.log)

        self.assertTrue(self.clip.is_file())
        self.assertFalse((self.labeled_root / "unsure" / self.clip.name).exists())

    def test_faint_only_applies_to_frogs(self) -> None:
        with self.assertRaisesRegex(ValueError, "faint"):
            record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log, faint=True)

    def test_confirm_refuses_a_clip_outside_the_labeled_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "labeled"):
            confirm_decision(self.clip, labeled_root=self.labeled_root, log=self.log)

    def test_undo_returns_the_clip_and_logs_where_it_went(self) -> None:
        move = record_decision(self.clip, "unsure", labeled_root=self.labeled_root, log=self.log)

        returned = undo_move(move, labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(returned, self.clip)
        self.assertTrue(self.clip.is_file())
        self.assertFalse(move.destination.exists())
        last = self.log.rows()[-1]
        self.assertEqual((last.decision, last.action), ("queue", "undo"))
        self.assertIsNone(last.faint)

    def test_undo_of_a_change_logs_the_folder_it_returned_to(self) -> None:
        first = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)
        second = record_decision(first.destination, "litoria_aurea", labeled_root=self.labeled_root, log=self.log)

        undo_move(second, labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(self.log.rows()[-1].decision, "non_target")
        self.assertEqual(current_folder(first.destination, self.labeled_root), "non_target")

    def test_summarize_counts_clips_and_distinct_recordings(self) -> None:
        self._write(self.labeled_root / "litoria_aurea" / "20220101_220000_start5s.png")
        self._write(self.labeled_root / "litoria_aurea" / "20220101_220000_start10s.png")
        self._write(self.labeled_root / "non_target" / "20220102_100000_start0s.png")
        self._write(self.labeled_root / "unsure" / "20220103_010000_start0s.png")

        summary = summarize_labels(self.labeled_root)

        self.assertEqual((summary["litoria_aurea"].clips, summary["litoria_aurea"].recordings), (2, 1))
        self.assertEqual((summary["non_target"].clips, summary["non_target"].recordings), (1, 1))
        self.assertEqual((summary["unsure"].clips, summary["unsure"].recordings), (1, 1))

    def _write(self, path: Path, content: bytes = b"png") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


if __name__ == "__main__":
    unittest.main()
