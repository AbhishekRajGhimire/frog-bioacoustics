"""What the site staff package holds: its layout, names, texts, and limits."""

from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath


PACKAGE_FOLDER = "Frog Call Scanner"
LAUNCHER_FILE = "Start Frog Call Scanner.bat"
INSTRUCTIONS_FILE = "How to use.txt"
BUILD_INFO_FILE = "build.json"
PYTHON_FOLDER = Path("app", "python")
APP_FILE = Path("app", "classifier_app", "app.py")
MODELS_FOLDER = Path("app", "models")
DATA_FOLDER = r"%LOCALAPPDATA%\Frog Call Scanner"
# Windows Explorer and Python fail on paths over 260 characters unless long paths
# are enabled, so 160 inside the package leaves 100 for wherever it is unzipped.
PATH_BUDGET = 160
# Only compiling needs these; Scripts also holds launchers that record the build
# computer's paths and would not work anywhere else.
PRUNED_FOLDERS = (
    Path("include"),
    Path("libs"),
    Path("Scripts"),
    Path("Lib", "site-packages", "torch", "include"),
    Path("Lib", "site-packages", "torch", "share"),
)
PRUNED_SUFFIX = ".lib"
_WIDTH = 76
_VERBATIM = "   "


@dataclass(frozen=True)
class BuildInfo:
    version: str
    commit: str
    uncommitted: bool
    built_at: str
    python: str
    torch: str
    model_run: str


def zip_name(info: BuildInfo) -> str:
    uncommitted = "-uncommitted" if info.uncommitted else ""
    return f"FrogCallScanner-{info.version}-{info.commit[:7]}{uncommitted}.zip"


def build_info_json(info: BuildInfo) -> bytes:
    return (json.dumps(asdict(info), indent=2) + "\n").encode("utf-8")


def launcher_text() -> str:
    """The double-click launcher, with the CRLF line endings Windows batch files expect."""
    python = rf"%~dp0{PureWindowsPath(PYTHON_FOLDER)}\python.exe"
    lines = (
        "@echo off",
        "setlocal",
        "title Frog Call Scanner",
        f'if not exist "{python}" (',
        "  echo This folder is incomplete. Unzip the whole download first: right-click the zip file,",
        "  echo choose Extract All, then start the scanner from the unzipped folder.",
        "  echo.",
        "  pause",
        "  exit /b 1",
        ")",
        f'"{python}" -I -m frog_classifier.app.launch'
        f' --app "%~dp0{PureWindowsPath(APP_FILE)}"'
        f' --models "%~dp0{PureWindowsPath(MODELS_FOLDER)}"'
        f' --data "{DATA_FOLDER}"',
        "if errorlevel 1 pause",
    )
    return "\r\n".join(lines) + "\r\n"


def instructions_text(info: BuildInfo) -> str:
    """Plain instructions for the person who unzips the package."""
    uncommitted = ", uncommitted" if info.uncommitted else ""
    launcher = Path(LAUNCHER_FILE).stem
    paragraphs = (
        f"Frog Call Scanner {info.version} (build {info.commit[:7]}{uncommitted}, {info.built_at[:10]})",
        "Finds the calls of the green and golden bell frog, Litoria aurea, in field recordings and shows "
        "when the frogs called. It needs Windows 10 or 11, 64-bit, and nothing else.",
        "STARTING",
        "1. Unzip the whole download first: right-click the zip file and choose Extract All. Starting from "
        "inside the zip does not work.",
        f'2. Double-click "{launcher}". The first start can take a minute while Windows checks the new files. '
        "If Windows says it protected your PC, choose More info and then Run anyway; if it asks whether to "
        "run the file, choose Run.",
        "3. The scanner opens in your web browser. Keep the black window open while you use it. Closing that "
        "window stops the scanner and any scan; a stopped scan carries on where it stopped when you press "
        "Continue scan.",
        "RESULTS",
        "Results and uploaded recordings are kept in this folder, which the About page also shows:",
        f"{_VERBATIM}{DATA_FOLDER}",
        "Paste it into the address bar of File Explorer to open it. Results stay there when this folder is "
        "replaced by a newer version or deleted, and Download results saves a copy wherever you choose.",
        "UPDATING",
        "Delete this folder and unzip the new version in its place. Results are kept.",
        "REMOVING",
        "Delete this folder, and the results folder above too if the results are no longer needed.",
        "BEFORE RELYING ON THE NUMBERS",
        f"The detector, {info.model_run}, was trained and checked on recordings from one site, IWS076, made "
        "between midnight and noon from January to April 2022. At other sites, seasons, or hours, listen to "
        "a sample of detections before relying on the numbers.",
        "IF IT DOES NOT START",
        "Read the message in the black window; it says what to do.",
    )
    lines = []
    for paragraph in paragraphs:
        if paragraph.startswith(_VERBATIM):
            # A folder must stay on one line so it can be copied whole.
            lines.append(paragraph)
        else:
            indent = _VERBATIM if paragraph[:2] in {"1.", "2.", "3."} else ""
            lines.extend(textwrap.wrap(paragraph, _WIDTH, subsequent_indent=indent))
        lines.append("")
    return "\r\n".join(lines)


def prune(python_dir: Path) -> None:
    """Remove the files only compiling needs from an installed Python."""
    for folder in PRUNED_FOLDERS:
        if (python_dir / folder).is_dir():
            shutil.rmtree(python_dir / folder)
    for path in python_dir.rglob(f"*{PRUNED_SUFFIX}"):
        if path.is_file():
            path.unlink()


def paths_over_budget(package_dir: Path, budget: int = PATH_BUDGET) -> list[str]:
    """Paths inside package_dir longer than budget characters, as Windows writes them, longest first."""
    over = (path for path in _relative_paths(package_dir) if len(path) > budget)
    return sorted(over, key=len, reverse=True)


def longest_path(package_dir: Path) -> int:
    return max((len(path) for path in _relative_paths(package_dir)), default=0)


def _relative_paths(package_dir: Path) -> list[str]:
    return [str(PureWindowsPath(path.relative_to(package_dir))) for path in package_dir.rglob("*")]
