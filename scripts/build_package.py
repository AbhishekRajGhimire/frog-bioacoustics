"""Build the site staff package: a portable Windows folder and zip that start the app with a double-click.

The package carries its own Python, the locked dependencies with PyTorch for the
processor, one saved model, and the app. It goes to dist/ and is kept only when
the package's own Python passes the self-check. Building needs a committed tree,
so every package maps to one commit.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from frog_classifier.app.models import ModelError, newest_model
from frog_classifier.packaging.build import BuildError, build_package, find_uv


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", type=Path, default=None, help="Saved run folder. Default: the newest run under models/.")
    parser.add_argument("--output", type=Path, default=Path("dist"), help="Output folder. Default: dist/.")
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="Build from uncommitted sources, for testing; the package is marked uncommitted.",
    )
    args = parser.parse_args()

    try:
        model_dir = _resolve(repo_root, args.model) if args.model is not None else newest_model(repo_root / "models").run_dir
        built = build_package(
            repo_root,
            model_dir,
            _resolve(repo_root, args.output),
            uv=find_uv(os.environ),
            allow_uncommitted=args.allow_uncommitted,
            now=datetime.now(timezone.utc),
        )
    except (BuildError, ModelError) as error:
        print(f"Cannot build the package: {error}", file=sys.stderr)
        return 1
    print(f"Package: {built.zip_path} ({built.zip_path.stat().st_size / 1e6:,.0f} MB)")
    print(
        f"Unzipped: {built.folder}, {built.files:,} files, {built.size_bytes / 1e9:.2f} GB, "
        f"longest path {built.longest_path} characters"
    )
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
