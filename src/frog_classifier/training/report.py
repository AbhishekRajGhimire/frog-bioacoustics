"""Human-readable run report and precision-recall figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from .metrics import evaluate


@dataclass(frozen=True)
class CurveInput:
    title: str
    labels: np.ndarray
    resnet: np.ndarray
    baseline: np.ndarray
    resnet_threshold: float
    baseline_threshold: float


def write_precision_recall_figure(path: Path, panels: Sequence[CurveInput]) -> None:
    """One panel per split, both models' curves, and a dot at each chosen threshold."""
    figure = Figure(figsize=(5 * len(panels), 4.5), layout="constrained")
    for axes, panel in zip(figure.subplots(1, len(panels), squeeze=False)[0], panels):
        for name, probabilities, threshold, colour in (
            ("ResNet-18", panel.resnet, panel.resnet_threshold, "tab:blue"),
            ("Logistic baseline", panel.baseline, panel.baseline_threshold, "tab:gray"),
        ):
            precision, recall, _ = precision_recall_curve(panel.labels, probabilities)
            axes.plot(recall, precision, color=colour, label=name)
            point = evaluate(panel.labels, probabilities, threshold)
            axes.plot(point.recall, point.precision, "o", color=colour)
        axes.set(title=panel.title, xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01))
        axes.grid(alpha=0.3)
        axes.legend(loc="lower left")
    figure.savefig(path, dpi=120)


def render_report(metadata: Mapping[str, Any]) -> str:
    """Markdown summary built only from run.json content, so it can be regenerated."""
    metrics = metadata["metrics"]
    resnet, baseline = metrics["resnet"], metrics["baseline"]
    threshold = metadata["threshold"]["value"]
    test = resnet["test"]
    training = metadata["training"]
    data = metadata["data"]
    lines = [
        f"# Training run {metadata['run_name']}",
        "",
        "## Test results",
        "",
        "The test fold holds recordings that played no part in training or in choosing the threshold.",
        "",
        *_model_table([
            ("ResNet-18", resnet["test"], threshold),
            ("Logistic baseline", baseline["test"], baseline["threshold"]),
        ]),
        "",
        f"Threshold rule: {metadata['threshold']['rule']}.",
        f"At this threshold the ResNet flags {test['false_positive_rate']:.1%} of test background clips, "
        f"about {metrics['false_detections_per_hour']:.1f} false detections per hour of recording without a frog call.",
        "",
        "## Test confusion matrix",
        "",
        "| Actual | Predicted background | Predicted frog |",
        "| --- | ---: | ---: |",
        f"| Background | {test['true_negatives']} | {test['false_positives']} |",
        f"| Frog | {test['false_negatives']} | {test['true_positives']} |",
        "",
        "## Validation results",
        "",
        *_model_table([
            ("ResNet-18", resnet["validation"], threshold),
            ("Logistic baseline", baseline["validation"], baseline["threshold"]),
        ]),
        "",
        "## Test errors by recording hour",
        "",
        "| Hour | Background clips | False positives | Frog clips | Missed calls |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {_hour(row['hour'])} | {row['background']} | {row['false_positives']} | {row['frog']} | {row['missed']} |"
            for row in metrics["test_by_hour"]
        ),
        "",
        "## Faint calls",
        "",
        "Frog clips labeled with the Frog faint button are counted separately from the other frog clips.",
        "",
        "| Split | Faint frog clips | Detected | Other frog clips | Detected |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {split.capitalize()} | {counts['faint']['clips']} | {counts['faint']['detected']} "
            f"| {counts['other']['clips']} | {counts['other']['detected']} |"
            for split, counts in metrics["faint_recall"].items()
        ),
        "",
        *_cross_validation_section(metadata["cross_validation"]),
        "## Training",
        "",
        f"- Best validation epoch: {training['best_epoch']} of {training['epochs_run']} run.",
        f"- Device: {training['device_name']} ({training['device']}).",
        f"- Training time for all rotations: {training['seconds']:.0f} seconds.",
        f"- Seed: {metadata['seed']}.",
        f"- Starting weights: {metadata['model']['pretrained_weights'] or 'random initialization'}.",
        "",
        "## Data",
        "",
        "| Split | Clips | Frog | Background | Recordings |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {name.capitalize()} | {split['examples']} | {split['frog']} | {split['background']} | {split['recordings']} |"
            for name, split in data["splits"].items()
        ),
        "",
        f"Test fold {data['test_fold']}, validation fold {data['validation_fold']}.",
        f"Faint-tagged frog clips in the manifest: {data['faint_tagged_frog_clips']}.",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "| --- | --- |",
        *(f"| {name} SHA-256 | `{value}` |" for name, value in metadata["checksums"].items()),
        f"| Git commit | `{metadata['source']['git_commit']}` |",
        f"| Uncommitted changes | {metadata['source']['git_worktree_dirty']} |",
        *(f"| {name} | {value} |" for name, value in metadata["environment"].items()),
        "",
    ]
    return "\n".join(lines)


def _model_table(rows: Sequence[tuple[str, Mapping[str, Any], float]]) -> list[str]:
    return [
        "| Model | Average precision | Precision | Recall | F1 | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {name} | {split['average_precision']:.3f} | {split['precision']:.3f} "
            f"| {split['recall']:.3f} | {split['f1']:.3f} | {threshold:.3f} |"
            for name, split, threshold in rows
        ),
    ]


def _cross_validation_section(cross_validation: Mapping[str, Any] | None) -> list[str]:
    if cross_validation is None:
        return ["Cross-validation was skipped for this run.", ""]
    lines = [
        "## Cross-validation",
        "",
        "Each rotation trains a new model with a different test fold and chooses its own threshold on the next fold.",
        "",
        "| Rotation | Test fold | Validation fold | ResNet AP | ResNet F1 | Baseline AP | Baseline F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {row['rotation']} | {row['test_fold']} | {row['validation_fold']} "
            f"| {row['resnet']['average_precision']:.3f} | {row['resnet']['f1']:.3f} "
            f"| {row['baseline']['average_precision']:.3f} | {row['baseline']['f1']:.3f} |"
            for row in cross_validation["rotations"]
        ),
        "",
        "Mean and standard deviation over the rotations:",
        "",
        "| Model | Average precision | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, name in (("resnet", "ResNet-18"), ("baseline", "Logistic baseline")):
        summary = cross_validation["summary"][key]
        cells = " | ".join(
            f"{summary[metric]['mean']:.3f} ({summary[metric]['std']:.3f})"
            for metric in ("average_precision", "precision", "recall", "f1")
        )
        lines.append(f"| {name} | {cells} |")
    return [*lines, ""]


def _hour(hour: int | None) -> str:
    return "unknown" if hour is None else f"{hour:02d}:00"
