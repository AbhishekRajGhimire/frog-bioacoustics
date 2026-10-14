"""Check that a saved training run reproduces its own predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from frog_classifier.training.fit import select_device
from frog_classifier.training.verify import verify_run


LISTED_PROBLEMS = 20


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder, for example models/resnet18-20261014T113000Z.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else repo_root / args.run
    result = verify_run(run_dir, repo_root, device=select_device(args.device))

    print(f"Checked {result.checked} predictions; largest difference {result.largest_difference:.2e}")
    for problem in result.problems[:LISTED_PROBLEMS]:
        print(f"FAIL {problem}")
    if len(result.problems) > LISTED_PROBLEMS:
        print(f"... and {len(result.problems) - LISTED_PROBLEMS} more")
    print("PASSED" if result.passed else f"FAILED with {len(result.problems)} problems")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
