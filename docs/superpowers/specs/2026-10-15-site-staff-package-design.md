# Site Staff Package Design

## Summary

This is the third and last part of Phase 6.
It turns the site staff app into a folder that a person unzips and starts with a double-click, on a Windows computer that has no Python, no uv, and no command line skills.
One build command makes the package from a committed version of the repository, checks it with the package's own Python, and zips it.
The package carries its own Python, every dependency from the lock file, the newest saved model, and the app.
Results live outside the package, so replacing the package with a newer one keeps them.

## Context

Part 2 built `classifier_app/app.py`, which runs with `uv run streamlit run classifier_app/app.py --server.address localhost` and starts each scan in its own Python process.
`Launch_Labeler.bat` already starts the labeler with a double-click, but it needs Python 3.13 installed and downloads dependencies on every start.
uv 0.12.1 can install a standalone, relocatable CPython for Windows; its 3.13.14 build includes Tk, which the Browse button needs, and its own `vcruntime140.dll` and `vcruntime140_1.dll`.
PyTorch also loads `msvcp140.dll`, which neither Python nor PyTorch ships and which a bare Windows computer may lack.

Measured on this computer, a scan of 83 recordings took 17 seconds with the graphics card and 33 seconds with the processor alone, with identical detections, 214 of 4,980 windows.
The processor-only PyTorch build installs in 540 MB; the graphics-card build installs in 2.9 GB.
A trial install of the processor-only dependency set into the standalone Python, without files that only compiling needs, took 1.4 GB in 31,400 files, and its longest path below the Python folder was 145 characters.

## Goals

- Start the app with a double-click on Windows 10 or 11, 64-bit, with nothing else installed.
- Work offline once unzipped.
- Keep results and uploads when the package is replaced by a newer one.
- Open the browser at the app, and open it again rather than start a second copy when the package is started twice.
- Stop the app, and any scan, when its window is closed; the scan continues later as in part 2.
- Say plainly what to do when the computer lacks Microsoft's C++ runtime.
- Make every package traceable to a commit, a model run, and the exact dependency versions.
- Refuse to produce a package that fails its own checks.

## Non-goals

- No installer, Start menu entry, or code signing; Windows will show its warning for files from the internet, and the instructions say how to continue.
- No automatic updates.
- No macOS or Linux package.
- No graphics-card package; the processor build is fast enough and a quarter of the size.
- No copy of Microsoft's C++ runtime inside the package.
- No labeler, training, or command-line tools in the package.

## Decisions

### A portable folder in a zip

The package is a folder, `Frog Call Scanner`, inside a zip file.
An installer would need tools outside the project's uv toolchain, a signing certificate to avoid Windows' strongest warning, and splitting for payloads near 2 GB, and it would wrap this same folder anyway.
A signed installer can be added later as its own decision.

The zip is named `FrogCallScanner-<version>-<commit>.zip`, with `-uncommitted` before `.zip` when built from changed files.
Windows' Extract All puts it in a folder of that name, so a file inside the package ends up at about 90 characters plus its path inside the package.

### Path budget

Windows Explorer and Python both fail on paths longer than 260 characters unless long paths are enabled, which is off by default.
Every path inside the package, measured from inside `Frog Call Scanner`, must be at most 160 characters, which leaves 100 characters for wherever the zip is extracted.
The build fails and names the longest paths when any exceeds that.

### Layout

```text
Frog Call Scanner/
  Start Frog Call Scanner.bat   the double-click launcher
  How to use.txt                plain instructions, with the version and build
  build.json                    version, commit, build time, Python, PyTorch, and model run
  app/
    python/                     standalone CPython 3.13.14 with every dependency and frog_classifier installed
    classifier_app/app.py       the Streamlit page
    models/<run>/               the whole saved run, as run.json lists it
```

The whole run folder is copied, because loading a run checks every file `run.json` records.

### Python and dependencies

The build installs CPython 3.13.14 with `uv python install` into a temporary folder and copies it into the package.
It installs the dependencies and the project with `uv sync --frozen --no-default-groups --group model-cpu --no-editable`, pointing `UV_PROJECT_ENVIRONMENT` at that Python, so every version and hash comes from `uv.lock`.

The lock gains a `model-cpu` dependency group with the same PyTorch and torchvision versions from the PyTorch CPU index, declared as conflicting with the `model` group.
The `model` group stays the default, so development environments keep the graphics-card build and `requirements_labeler.txt` does not change.

After installing, the build removes files that only compiling needs: Python's `include`, `libs`, and `Scripts` folders, PyTorch's `include` and `share` folders, and every `.lib` import library.
`Scripts` also holds launchers that record the build computer's paths and would not work elsewhere.

The build then compiles every module with unchecked-hash bytecode, so the timestamps a zip tool sets never make Python recompile, and the first start is not slowed by compiling.

### Data outside the package

Results and uploads go to `%LOCALAPPDATA%\Frog Call Scanner`, with the same `results/detections` and `raw/uploads` layout as the repository.
That folder survives replacing or deleting the package, needs no administrator rights, and is never synchronized by OneDrive, which can lock or duplicate an SQLite database it syncs.
The About page shows the data folder.

The model stays inside the package, so a newer package brings its newer model and an older one never picks it up.
`frog_classifier.app.paths` therefore reads the model folder from `FROG_CLASSIFIER_MODELS` when it is set, and from `<root>/models` otherwise.
Records made with an older model stay listed; their results still show, and listening explains that their model is no longer present.

### Launcher

`Start Frog Call Scanner.bat` runs `app\python\python.exe -I -m frog_classifier.app.launch` with the app file, the model folder, and the data folder as arguments, every path built from the batch file's own location, so it works from any folder, including a network one.
When the packaged Python is missing, which is what starting it from inside the zip looks like, it says to extract the whole download first.
`-I` keeps the person's own Python settings and user packages out of the launcher.
The batch file pauses only when the launcher reports a problem, so the message stays readable.

`frog_classifier.app.launch` then:

1. Refuses a data folder that is not an absolute path, and creates it.
2. Checks that `msvcp140.dll` loads; if not, says that Microsoft's Visual C++ Redistributable (x64) must be installed, gives its address, `https://aka.ms/vs/17/release/vc_redist.x64.exe`, and stops.
3. Uses port 8571; if an app already answers there, opens the browser at it and stops; if another program holds the port, takes a free one.
4. Starts Streamlit with the same Python, in a copy of the environment without any `PYTHON` variables, with `PYTHONNOUSERSITE=1`, `FROG_CLASSIFIER_ROOT` set to the data folder, and `FROG_CLASSIFIER_MODELS` set to the model folder; scan processes inherit that environment.
5. Waits until the app answers its health address, opens the browser at `http://localhost:<port>`, prints that the window must stay open while the app is used, and waits for Streamlit to end.

Streamlit runs with `--server.address localhost`, `--server.headless true`, `--browser.gatherUsageStats false`, `--client.toolbarMode minimal`, and `--server.fileWatcherType none`.
Headless mode avoids Streamlit's first-run email question, which would wait for typing in the window; the launcher opens the browser itself once the app answers.
The minimal toolbar hides the Deploy button and the developer menu.

Closing the window ends the launcher, Streamlit, and any scan, because all of them share the window's console; part 2 already shows such a scan as interrupted and continues it on request.
`--no-browser` skips opening the browser, for tests.

### App changes

- The processor message becomes "Using the processor.", because in this package a computer with a graphics card still scans on the processor, and the old hint suggested buying hardware.
- The About page shows where results and uploads are kept.

### Build command

`uv run python scripts/build_package.py` builds the package from the newest saved model, or from `--model <run folder>`.
It refuses a working tree with uncommitted changes to tracked files unless `--allow-uncommitted` is given, which marks the package as uncommitted in its name and `build.json`.
It builds in a new folder under `dist/`, replaces `dist/Frog Call Scanner` and the zip only after every check passes, and prints the zip's path and size.
`dist/` is ignored by Git.

### Self-check

Before zipping, the build runs `frog_classifier.packaging.selfcheck` with the package's own Python and an environment holding only the Windows essentials, so nothing from the build computer's Python, uv, or repository can help it pass.
It fails the build unless:

- every `sys.path` entry is inside the package;
- PyTorch is a CPU build;
- the app modules import without loading PyTorch;
- the packaged model loads, which verifies its checksums;
- a synthetic ten-second recording is scanned by `detect_calls` with rendering worker processes, giving two stored windows;
- the launcher's Streamlit start serves the app page on a free port, with a temporary data folder.

## Architecture

```text
scripts/build_package.py              command wrapper
src/frog_classifier/packaging/
  __init__.py                         package docstring only
  contents.py                         names, build information, launcher and instruction text, pruning, path budget
  build.py                            one build: check the tree, install, copy, prune, compile, self-check, zip
  selfcheck.py                        the checks run by the package's own Python
src/frog_classifier/app/
  launch.py                           the launcher: runtime check, port, environment, Streamlit start, browser
  paths.py                            FROG_CLASSIFIER_MODELS
```

`launch.py` imports no PyTorch, like the rest of `frog_classifier.app`.
`selfcheck.py` is installed with the package so it can run there; it imports PyTorch only inside the checks that need it.

## Test strategy

- `launch`: the Streamlit command and its flags, the child environment, port choice with a held port and with an app already answering, the runtime check's message, the already-running path opening the browser without starting anything, and a real Streamlit start of `classifier_app/app.py` on a synthetic data root that answers its health address and serves its page.
- `paths`: `FROG_CLASSIFIER_MODELS` overrides the model folder only.
- `contents`: the zip name with and without uncommitted changes, `build.json`, the launcher's arguments, the instructions naming the data folder and the limits of the detector, pruning on a synthetic tree, and the path budget naming the paths over it.
- `build`: the uncommitted-tree refusal on a temporary Git repository.
- `selfcheck`: each check's failure message on synthetic inputs where it can be made to fail without a package.
- The repository tests: the new modules and command in the navigation and canonical layout, `dist/` ignored, and the build command documented.
- The full build runs at acceptance, not in the unit suite, because it downloads and installs about a gigabyte.

## Acceptance criteria

- `uv lock --check` passes, development environments still install PyTorch `+cu130`, and regenerating `requirements_labeler.txt` changes nothing.
- The build makes the zip from a committed tree, its self-check passes, and its size, file count, and longest path are recorded.
- The zip extracted with spaces in its location, and started with an environment that has no Python, uv, or repository on its path, opens the app.
- Through that app, a scan of real recordings finishes on the processor, with detections matching a command-line scan of the same recordings, and results, listening, and the download work.
- Starting the package a second time opens the running app instead of a second one.
- Closing the launcher stops the app and leaves an interrupted scan that continues.
- Results land in the data folder, not in the package.
