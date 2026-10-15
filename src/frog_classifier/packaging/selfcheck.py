"""Check that the site staff package runs on its own.

The build runs python -I -m frog_classifier.packaging.selfcheck PACKAGE with the
package's own Python and an environment holding only what Windows programs need,
so nothing from the build computer can help a check pass. PyTorch is imported
only inside the checks that need it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from frog_classifier.app import launch
from frog_classifier.app.models import newest_model
from frog_classifier.app.paths import MODELS_VARIABLE, ROOT_VARIABLE

from .contents import APP_FILE, MODELS_FOLDER


class SelfCheckError(RuntimeError):
    pass


def check_isolated(package: Path, search_path: Sequence[str]) -> None:
    """Every import location is inside the package."""
    inside = package.resolve()
    for entry in search_path:
        location = Path(entry).resolve()
        if not location.is_relative_to(inside):
            raise SelfCheckError(f"Python imports from {location}, outside the package")


def check_app_without_pytorch() -> None:
    """The app's own process never loads PyTorch; only scans do."""
    program = (
        "import importlib, pkgutil, sys, frog_classifier.app as app\n"
        "for module in pkgutil.iter_modules(app.__path__):\n"
        "    importlib.import_module(f'frog_classifier.app.{module.name}')\n"
        "print('torch' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    if result.returncode != 0:
        raise SelfCheckError(f"the app modules do not import: {result.stderr.strip()}")
    if result.stdout.strip() != "False":
        raise SelfCheckError("importing the app modules loads PyTorch")


def check_cpu_pytorch() -> None:
    """PyTorch is the processor build the package is meant to carry."""
    import torch

    if torch.version.cuda is not None:
        raise SelfCheckError(f"PyTorch {torch.__version__} is a graphics-card build, not the processor build")


def check_model(models: Path) -> Path:
    """The newest packaged model loads, which checks every file its run.json records."""
    from frog_classifier.training.saved import SavedRunError, load_saved_run

    run_dir = newest_model(models).run_dir
    try:
        load_saved_run(run_dir)
    except SavedRunError as error:
        raise SelfCheckError(str(error)) from error
    return run_dir


def check_scan(run_dir: Path) -> None:
    """A ten-second recording is scanned, with rendering worker processes, into two windows."""
    import torch

    from frog_classifier.inference.detect import detect_calls

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        _write_noise(folder / "recordings" / "20220123_010000.WAV", seconds=10)
        summary = detect_calls(
            run_dir, folder / "recordings", folder / "record", device=torch.device("cpu"), workers=2,
        )
    if (summary.processed, summary.failed, summary.windows) != (1, 0, 2):
        raise SelfCheckError(
            f"scanning a ten-second recording gave {summary.processed} processed, {summary.failed} failed, "
            f"and {summary.windows} windows, not 1, 0, and 2"
        )


def check_app_renders(app_file: Path, models: Path) -> None:
    """The app page runs in this Python and finds the packaged model."""
    from streamlit.testing.v1 import AppTest

    expected = f"Detector {newest_model(models).name}"
    with tempfile.TemporaryDirectory() as data, _environment({ROOT_VARIABLE: data, MODELS_VARIABLE: str(models)}):
        app = AppTest.from_file(str(app_file), default_timeout=60)
        app.run()
    if app.exception:
        raise SelfCheckError(f"the app page failed: {app.exception[0].value}")
    if not any(expected in caption.value for caption in app.caption):
        raise SelfCheckError("the app page does not name the packaged model")


def check_app_starts(app_file: Path, models: Path) -> None:
    """The launcher's Streamlit start answers on this computer."""
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        log_path = folder / "streamlit.log"
        port = launch.free_port()
        with log_path.open("wb") as log:
            process = launch.start_app(
                app_file, port, launch.child_environment(os.environ, folder / "data", models), log=log,
            )
            try:
                launch.wait_until_ready(port, process)
            except launch.LaunchError as error:
                raise SelfCheckError(f"{error}: {log_path.read_text(errors='replace')[-2000:]}") from error
            finally:
                process.terminate()
                process.wait(timeout=30)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the site staff package runs on its own.")
    parser.add_argument("package", type=Path, help="The package folder, Frog Call Scanner.")
    args = parser.parse_args(argv)
    package = args.package.resolve()
    app_file = package / APP_FILE
    models = package / MODELS_FOLDER
    checks: tuple[tuple[str, Callable[[], object]], ...] = (
        ("imports come only from the package", lambda: check_isolated(package, sys.path)),
        ("the app modules leave PyTorch unloaded", check_app_without_pytorch),
        ("PyTorch is the processor build", check_cpu_pytorch),
        ("the model loads", lambda: check_model(models)),
        ("a recording is scanned", lambda: check_scan(newest_model(models).run_dir)),
        ("the app page runs", lambda: check_app_renders(app_file, models)),
        ("the app answers", lambda: check_app_starts(app_file, models)),
    )
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as error:  # Every check reports, so one failure cannot hide another.
            failures += 1
            print(f"FAILED  {name}: {error}", flush=True)
        else:
            print(f"ok      {name}", flush=True)
    return 1 if failures else 0


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_noise(path: Path, *, seconds: int, sample_rate_hz: int = 48000) -> None:
    samples = np.random.default_rng(0).normal(0.0, 0.1, seconds * sample_rate_hz)
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(sample_rate_hz)
        sound.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
