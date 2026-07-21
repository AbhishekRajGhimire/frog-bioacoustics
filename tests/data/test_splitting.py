from __future__ import annotations

import unittest

from frog_classifier.data.manifest import LabeledExample
from frog_classifier.data.splitting import create_split_plan
from frog_classifier.data.validation import ManifestValidationError


def synthetic_examples(
    *,
    positive_groups: int = 15,
    negative_groups: int = 45,
) -> tuple[LabeledExample, ...]:
    examples = [
        LabeledExample(
            example_id=f"positive_{index}_start0s",
            image_path=f"labeled/litoria_aurea/positive_{index}_start0s.png",
            label=1,
            label_name="litoria_aurea",
            recording_id=f"positive_{index}",
            start_s=0,
        )
        for index in range(positive_groups)
    ]
    examples.extend(
        LabeledExample(
            example_id=f"negative_{index}_start0s",
            image_path=f"labeled/non_target/negative_{index}_start0s.png",
            label=0,
            label_name="non_target",
            recording_id=f"negative_{index}",
            start_s=0,
        )
        for index in range(negative_groups)
    )
    if negative_groups:
        examples.append(
            LabeledExample(
                example_id="negative_0_start5s",
                image_path="labeled/non_target/negative_0_start5s.png",
                label=0,
                label_name="non_target",
                recording_id="negative_0",
                start_s=5,
            )
        )
    return tuple(examples)


class CreateSplitPlanTests(unittest.TestCase):
    def test_is_deterministic_and_independent_of_example_order(self) -> None:
        examples = synthetic_examples()

        first = create_split_plan(examples, seed=42)
        second = create_split_plan(tuple(reversed(examples)), seed=42)

        self.assertEqual(dict(first.fold_by_recording), dict(second.fold_by_recording))
        self.assertEqual(dict(first.split_by_recording), dict(second.split_by_recording))

    def test_places_both_labels_in_every_fold_and_split(self) -> None:
        examples = synthetic_examples()
        plan = create_split_plan(examples)

        labels_by_fold = {fold: set() for fold in range(plan.folds)}
        labels_by_split = {split: set() for split in ("train", "val", "test")}
        for example in examples:
            labels_by_fold[plan.fold_by_recording[example.recording_id]].add(example.label)
            labels_by_split[plan.split_by_recording[example.recording_id]].add(example.label)

        self.assertEqual(set(labels_by_fold), set(range(5)))
        self.assertTrue(all(labels == {0, 1} for labels in labels_by_fold.values()))
        self.assertTrue(all(labels == {0, 1} for labels in labels_by_split.values()))

    def test_assigns_exactly_one_fold_and_split_per_recording(self) -> None:
        examples = synthetic_examples()
        plan = create_split_plan(examples)

        assignments: dict[str, set[tuple[int, str]]] = {}
        for example in examples:
            assignments.setdefault(example.recording_id, set()).add((
                plan.fold_by_recording[example.recording_id],
                plan.split_by_recording[example.recording_id],
            ))

        self.assertTrue(all(len(values) == 1 for values in assignments.values()))
        self.assertEqual(set(plan.fold_by_recording), set(assignments))
        self.assertEqual(set(plan.split_by_recording), set(assignments))

    def test_rejects_fewer_class_groups_than_folds(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            create_split_plan(synthetic_examples(positive_groups=4))

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["insufficient_class_groups"],
        )

    def test_rejects_fewer_than_three_folds(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            create_split_plan(synthetic_examples(), folds=2)

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["invalid_fold_count"],
        )

    def test_rejects_identical_test_and_validation_folds(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            create_split_plan(synthetic_examples(), test_fold=1, val_fold=1)

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["invalid_fold_selection"],
        )

    def test_rejects_out_of_range_fold_selections(self) -> None:
        for arguments in ({"test_fold": -1}, {"val_fold": 5}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ManifestValidationError) as raised:
                    create_split_plan(synthetic_examples(), **arguments)

                self.assertEqual(
                    [issue.code for issue in raised.exception.issues],
                    ["invalid_fold_selection"],
                )


if __name__ == "__main__":
    unittest.main()
