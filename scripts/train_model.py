"""Train the ResNet-18 frog detector from the validated manifest and save a run folder under models/."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from frog_classifier.training.artifact import read_run
from frog_classifier.training.fit import select_device
from frog_classifier.training.run import RunPaths, train_and_save


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    defaults = RunPaths.defaults(repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument("--training-config", type=Path, default=defaults.training_config)
    parser.add_argument("--models-root", type=Path, default=defaults.models_root)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    parser.add_argument(
        "--no-cross-validation",
        action="store_true",
        help="Train only the saved model and skip the four extra fold rotations.",
    )
    args = parser.parse_args()

    paths = replace(
        defaults,
        manifest=_under(repo_root, args.manifest),
        training_config=_under(repo_root, args.training_config),
        models_root=_under(repo_root, args.models_root),
    )
    run_dir = train_and_save(paths, cross_validate=not args.no_cross_validation, device=select_device(args.device))

    metadata = read_run(run_dir)
    test = metadata["metrics"]["resnet"]["test"]
    print(f"Saved run: {run_dir}")
    print(
        f"Test: average precision {test['average_precision']:.3f}, precision {test['precision']:.3f}, "
        f"recall {test['recall']:.3f}, F1 {test['f1']:.3f} at threshold {metadata['threshold']['value']:.3f}"
    )
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def _under(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
