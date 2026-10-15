"""Build the site staff package: a portable folder and a zip, kept only when its self-check passes."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from .contents import (
    APP_FILE,
    BUILD_INFO_FILE,
    INSTRUCTIONS_FILE,
    LAUNCHER_FILE,
    MODELS_FOLDER,
    PACKAGE_FOLDER,
    PATH_BUDGET,
    PYTHON_FOLDER,
    BuildInfo,
    build_info_json,
    instructions_text,
    launcher_text,
    longest_path,
    paths_over_budget,
    prune,
    zip_name,
)


PYTHON_VERSION = "3.13.14"
DEPENDENCY_GROUP = "model-cpu"
# What ends up in the package; changes anywhere else do not make a build uncommitted.
PACKAGED_SOURCES = ("src", "classifier_app", "pyproject.toml", "uv.lock")
_ESSENTIAL_VARIABLES = (
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE", "TEMP", "TMP",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH", "USERNAME",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuiltPackage:
    folder: Path
    zip_path: Path
    info: BuildInfo
    files: int
    size_bytes: int
    longest_path: int


def find_uv(environ: Mapping[str, str]) -> str:
    """The uv that runs this build; uv run names itself in UV."""
    uv = environ.get("UV") or shutil.which("uv")
    if not uv:
        raise BuildError("uv was not found; run this command with uv run")
    return uv


def source_state(repo: Path) -> tuple[str, bool]:
    """The commit at HEAD, and whether the packaged sources differ from it."""
    commit = _git(repo, "rev-parse", "HEAD")
    changes = _git(repo, "status", "--porcelain", "--", *PACKAGED_SOURCES)
    return commit, bool(changes)


def python_install_command(uv: str, install_dir: Path) -> list[str]:
    # Without --no-bin and --no-registry, uv would add this Python to the build
    # computer's own tools folder and Windows registry.
    return [uv, "python", "install", PYTHON_VERSION, "--install-dir", str(install_dir), "--no-bin", "--no-registry"]


def sync_command(uv: str) -> list[str]:
    return [uv, "sync", "--frozen", "--no-default-groups", "--group", DEPENDENCY_GROUP, "--no-editable"]


def essential_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Only what Windows programs need, so the self-check cannot lean on the build computer's Python or uv."""
    environment = {name: environ[name] for name in _ESSENTIAL_VARIABLES if name in environ}
    system_root = environ.get("SYSTEMROOT", r"C:\Windows")
    environment["PATH"] = os.pathsep.join((rf"{system_root}\System32", system_root, rf"{system_root}\System32\Wbem"))
    return environment


def build_package(
    repo: Path,
    model_dir: Path,
    output: Path,
    *,
    uv: str,
    allow_uncommitted: bool,
    now: datetime,
    log: Callable[[str], None] = print,
) -> BuiltPackage:
    """Build, check, and zip the package; dist keeps it only when every check passes."""
    commit, uncommitted = source_state(repo)
    if uncommitted and not allow_uncommitted:
        raise BuildError(
            "the packaged sources have uncommitted changes; commit them, or pass --allow-uncommitted for a test build"
        )
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=output))
    try:
        package = staging / PACKAGE_FOLDER
        python_dir = package / PYTHON_FOLDER
        log(f"Installing Python {PYTHON_VERSION}...")
        _install_python(uv, staging / "python-download", python_dir)
        log("Installing the locked dependencies, with PyTorch for the processor...")
        _run(sync_command(uv), cwd=repo, env=_sync_environment(os.environ, python_dir))
        prune(python_dir)
        log("Compiling the Python modules...")
        _compile(python_dir)
        _copy_app(repo, model_dir, package)
        info = BuildInfo(
            version=_project_version(repo),
            commit=commit,
            uncommitted=uncommitted,
            built_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            python=PYTHON_VERSION,
            torch=_installed_version(python_dir, "torch"),
            model_run=model_dir.name,
        )
        (package / LAUNCHER_FILE).write_bytes(launcher_text().encode("ascii"))
        (package / INSTRUCTIONS_FILE).write_bytes(instructions_text(info).encode("utf-8"))
        (package / BUILD_INFO_FILE).write_bytes(build_info_json(info))
        too_long = paths_over_budget(package)
        if too_long:
            raise BuildError(
                f"{len(too_long)} paths inside the package are longer than {PATH_BUDGET} characters, "
                "the longest: " + "; ".join(too_long[:3])
            )
        log("Checking the package with its own Python...")
        _self_check(python_dir, package)
        files = [path for path in package.rglob("*") if path.is_file()]
        size_bytes = sum(path.stat().st_size for path in files)
        longest = longest_path(package)
        log("Zipping...")
        built_zip = staging / zip_name(info)
        _zip(package, built_zip)
        folder, zip_path = _keep(package, built_zip, output)
        return BuiltPackage(folder, zip_path, info, len(files), size_bytes, longest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _install_python(uv: str, install_dir: Path, python_dir: Path) -> None:
    _run(python_install_command(uv, install_dir))
    installed = [path for path in install_dir.glob(f"cpython-{PYTHON_VERSION}-*") if path.is_dir()]
    if len(installed) != 1:
        raise BuildError(f"expected one Python {PYTHON_VERSION} in {install_dir}, found {len(installed)}")
    python_dir.parent.mkdir(parents=True)
    shutil.move(installed[0], python_dir)


def _sync_environment(environ: Mapping[str, str], python_dir: Path) -> dict[str, str]:
    # uv run sets VIRTUAL_ENV to the development environment; the sync targets the package's Python.
    environment = {name: value for name, value in environ.items() if name != "VIRTUAL_ENV"}
    environment["UV_PROJECT_ENVIRONMENT"] = str(python_dir)
    return environment


def _compile(python_dir: Path) -> None:
    # Unchecked-hash bytecode is used whatever timestamps unzipping sets, so the first
    # start never recompiles. A few packages ship files that are not meant to compile,
    # such as templates, so compileall's failures are expected and harmless.
    subprocess.run(
        [
            str(python_dir / "python.exe"), "-I", "-m", "compileall", "-qq", "-f", "-j", "0",
            "--invalidation-mode", "unchecked-hash", str(python_dir / "Lib"),
        ],
        check=False,
    )


def _copy_app(repo: Path, model_dir: Path, package: Path) -> None:
    (package / APP_FILE).parent.mkdir(parents=True)
    shutil.copy2(repo / "classifier_app" / "app.py", package / APP_FILE)
    # The whole run, because loading it checks every file run.json records.
    shutil.copytree(model_dir, package / MODELS_FOLDER / model_dir.name)


def _project_version(repo: Path) -> str:
    return tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _installed_version(python_dir: Path, distribution: str) -> str:
    return _run([
        str(python_dir / "python.exe"), "-I", "-c",
        f"import importlib.metadata as metadata; print(metadata.version({distribution!r}))",
    ])


def _self_check(python_dir: Path, package: Path) -> None:
    result = subprocess.run(
        [str(python_dir / "python.exe"), "-I", "-m", "frog_classifier.packaging.selfcheck", str(package)],
        cwd=package,
        env=essential_environment(os.environ),
    )
    if result.returncode != 0:
        raise BuildError("the package failed its self-check; the lines above name each failed check")


def _zip(package: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, PurePosixPath(PACKAGE_FOLDER, *path.relative_to(package).parts).as_posix())


def _keep(package: Path, built_zip: Path, output: Path) -> tuple[Path, Path]:
    folder = output / PACKAGE_FOLDER
    zip_path = output / built_zip.name
    try:
        if folder.exists():
            shutil.rmtree(folder)
        package.rename(folder)
        built_zip.replace(zip_path)
    except OSError as error:
        raise BuildError(
            f"the new package could not replace {folder}; close any scanner started from it and build again ({error})"
        ) from error
    return folder, zip_path


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _run(command: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=None if env is None else dict(env), capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"{' '.join(command[:3])} failed:\n{(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()
