"""Start the frog call scanner for a person: the browser opens, and closing the window stops it.

The site staff package's Start Frog Call Scanner.bat runs
python -I -m frog_classifier.app.launch --app APP --models MODELS --data DATA.
Streamlit, and every scan it starts, shares this console window, so closing the
window ends them all, and the app continues an interrupted scan on request.
A lock in the data folder makes a second start open the running app instead of
starting another, even while the first is still starting.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from .paths import MODELS_VARIABLE, ROOT_VARIABLE


PREFERRED_PORT = 8571
READY_TIMEOUT_S = 120.0
LOCK_FILE = "launcher.lock"
# The port has its own file: Windows refuses any read that overlaps a locked byte.
PORT_FILE = "launcher.port"
RUNTIME_ADDRESS = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
# Headless mode skips Streamlit's first-run email question, which would wait for
# typing in the window; the launcher opens the browser itself once the app answers.
STREAMLIT_SETTINGS = (
    "--server.address", "localhost",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
    "--client.toolbarMode", "minimal",
    "--server.fileWatcherType", "none",
)
_POLL_SECONDS = 0.25
# Requests to the app on this computer must never go through a proxy set up for the internet.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LaunchError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the frog call scanner and open it in the browser.")
    parser.add_argument("--app", type=Path, required=True, help="The app page, classifier_app/app.py.")
    parser.add_argument("--models", type=Path, required=True, help="Folder of saved model runs.")
    parser.add_argument("--data", type=Path, required=True, help="Folder for results and uploaded recordings.")
    parser.add_argument("--port", type=int, default=PREFERRED_PORT, help="Preferred port. Default: %(default)s.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    args = parser.parse_args(argv)
    try:
        return _launch(args.app, args.models, args.data, args.port, open_browser=not args.no_browser)
    except LaunchError as error:
        print(f"\nThe frog call scanner cannot start: {error}.\n")
        return 1
    except KeyboardInterrupt:
        return 0


def _launch(app_file: Path, models: Path, data: Path, preferred_port: int, *, open_browser: bool) -> int:
    if not app_file.is_file() or not models.is_dir():
        missing = app_file if not app_file.is_file() else models
        raise LaunchError(f"{missing} is missing; unzip the whole package again")
    prepare_data_folder(data)
    problem = runtime_problem()
    if problem is not None:
        raise LaunchError(problem)
    claim = claim_launch(data)
    while claim is None:
        print("The frog call scanner is already open in another window.")
        running_port = wait_for_other_launch(data)
        if running_port is not None:
            _show(f"http://localhost:{running_port}", open_browser)
            return 0
        claim = claim_launch(data)
    with claim:
        port = choose_port(preferred_port)
        record_port(data, port)
        url = f"http://localhost:{port}"
        print("Starting the frog call scanner...")
        process = start_app(app_file, port, child_environment(os.environ, data, models))
        try:
            wait_until_ready(port, process)
            print(f"\nThe frog call scanner is running at {url}")
            print("Keep this window open while you use it. Closing it stops the scanner and any scan.\n")
            _show(url, open_browser)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()


def prepare_data_folder(data: Path) -> Path:
    if not data.is_absolute():
        raise LaunchError(f"the data folder {data} is not a full path")
    try:
        data.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LaunchError(f"the data folder {data} cannot be created: {error}") from error
    return data


def runtime_problem() -> str | None:
    """Why the detector cannot load on this computer, or None.

    PyTorch needs msvcp140.dll, which neither Python nor PyTorch ships.
    """
    if sys.platform != "win32":
        return None
    try:
        ctypes.WinDLL("msvcp140.dll")
    except OSError:
        return (
            "this computer lacks Microsoft's Visual C++ Redistributable (x64), which the detector needs; "
            f"ask whoever looks after this computer to install it from {RUNTIME_ADDRESS}, then start the scanner again"
        )
    return None


def app_answers(port: int) -> bool:
    try:
        with _DIRECT.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def choose_port(preferred: int) -> int:
    """The preferred port, or a free one when another program holds it.

    An app answering there is not reused: this computer's ports are shared by every
    Windows user, so it may be someone else's scanner with someone else's results.
    """
    return preferred if port_is_free(preferred) else free_port()


def claim_launch(data: Path) -> IO[bytes] | None:
    """Hold the data folder's launcher lock until the returned file is closed, or None
    when another launcher holds it.

    The operating system frees the lock when its launcher ends, however it ends, so a
    closed window never leaves it behind.
    """
    path = data / LOCK_FILE
    try:
        handle = open(os.open(path, os.O_RDWR | os.O_CREAT), "r+b", buffering=0)
    except OSError as error:
        raise LaunchError(f"{path} cannot be opened: {error}") from error
    try:
        _lock(handle)
    except OSError:
        handle.close()
        return None
    return handle


def record_port(data: Path, port: int) -> None:
    """Tell a second launcher which port the app of the launcher holding data answers on."""
    try:
        (data / PORT_FILE).write_bytes(str(port).encode("ascii"))
    except OSError as error:
        raise LaunchError(f"{data / PORT_FILE} cannot be written: {error}") from error


def wait_for_other_launch(data: Path, *, timeout_s: float = READY_TIMEOUT_S) -> int | None:
    """The port of the app the launcher holding data started, once it answers, or
    None when that launcher has ended."""
    deadline = time.monotonic() + timeout_s
    while True:
        port = _recorded_port(data)
        if port is not None and app_answers(port):
            return port
        claim = claim_launch(data)
        if claim is not None:
            claim.close()
            return None
        if time.monotonic() > deadline:
            raise LaunchError(f"the scanner in the other window did not answer within {timeout_s:.0f} seconds")
        time.sleep(_POLL_SECONDS)


def child_environment(environ: Mapping[str, str], data: Path, models: Path) -> dict[str, str]:
    """The person's environment without their Python settings, pointing the app at its folders.

    GIS and other tools sometimes set PYTHONHOME or PYTHONPATH for their own Python;
    scans inherit this environment, so those settings must not reach them.
    """
    environment = {name: value for name, value in environ.items() if not name.upper().startswith("PYTHON")}
    environment["PYTHONNOUSERSITE"] = "1"
    environment[ROOT_VARIABLE] = str(data)
    environment[MODELS_VARIABLE] = str(models)
    return environment


def streamlit_command(app_file: Path, port: int) -> list[str]:
    return [sys.executable, "-I", "-m", "streamlit", "run", str(app_file), "--server.port", str(port), *STREAMLIT_SETTINGS]


def start_app(
    app_file: Path,
    port: int,
    environment: Mapping[str, str],
    *,
    log: IO[bytes] | None = None,
) -> subprocess.Popen:
    """Start Streamlit; its messages go to this window, or to log when given."""
    return subprocess.Popen(
        streamlit_command(app_file, port),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
    )


def wait_until_ready(port: int, process: subprocess.Popen, *, timeout_s: float = READY_TIMEOUT_S) -> None:
    """Wait for the app to answer; a first start can be slow while Windows checks the new files."""
    deadline = time.monotonic() + timeout_s
    while not app_answers(port):
        if process.poll() is not None:
            raise LaunchError(
                f"the app stopped while starting, with exit code {process.returncode}; the messages above say why"
            )
        if time.monotonic() > deadline:
            raise LaunchError(f"the app did not answer within {timeout_s:.0f} seconds")
        time.sleep(_POLL_SECONDS)


def _lock(handle: IO[bytes]) -> None:
    """Lock handle's file, or raise OSError when another launcher holds it."""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _recorded_port(data: Path) -> int | None:
    try:
        return int((data / PORT_FILE).read_bytes().decode("ascii"))
    except (OSError, ValueError):
        return None


def _show(url: str, open_browser: bool) -> None:
    if open_browser and not webbrowser.open(url):
        print(f"Open {url} in your web browser.")


if __name__ == "__main__":
    raise SystemExit(main())
