# Data Integrity and Trustworthy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clear, reusable, deterministic, and leakage-safe data pipeline that produces a validated manifest and matching quality reports.

**Architecture:** Install a focused `src/frog_classifier/data` package and retain thin command-line wrappers under `scripts/`.
The manifest remains the source of truth, while configuration, naming, discovery, splitting, validation, reporting, and baseline loading become separately testable units.

**Tech Stack:** Python 3.13, standard-library dataclasses, TOML, CSV, JSON, SHA-256, and unittest, plus scikit-learn `StratifiedGroupKFold` and Hatchling packaging.

## Global Constraints

- Preserve current preprocessing and labeling behavior.
- Do not modify raw recordings, generated spectrograms, or existing human labels.
- Keep `litoria_aurea = 1` and `non_target = 0` as the trainable classes.
- Use five folds and seed `1337` by default.
- Use fold `0` for test, fold `1` for validation, and remaining folds for training by default.
- Reject invalid or ambiguous data without replacing existing outputs.
- Keep generated manifests and reports ignored by Git.
- Run every test without private recordings or spectrograms.
- Treat uncertain clips as review items, never as implicit negatives.

---

## File structure

Create these focused modules:

```text
src/frog_classifier/
  __init__.py
  data/
    __init__.py
    config.py
    naming.py
    manifest.py
    splitting.py
    validation.py
    reporting.py
config/
  preprocessing.toml
tests/data/
  __init__.py
  helpers.py
  test_config.py
  test_naming.py
  test_manifest.py
  test_splitting.py
  test_validation.py
  test_reporting.py
  test_manifest_cli.py
```

Modify `pyproject.toml`, `scripts/slice_audio.py`, `scripts/build_manifest.py`, and `scripts/train_baseline.py` to consume the package.
Create `README.md`, move `PROCESS.md` to `docs/workflow.md`, and move `docs/system design.md` to `docs/architecture.md`.
Remove the unused tracked `dataset/**/.gitkeep` placeholders only after verifying that `dataset/` contains no user files.

---

### Task 1: Installable package and preprocessing contract

**Files:**

- Create: `src/frog_classifier/__init__.py`
- Create: `src/frog_classifier/data/__init__.py`
- Create: `src/frog_classifier/data/config.py`
- Create: `config/preprocessing.toml`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_config.py`
- Modify: `pyproject.toml`
- Modify: `scripts/slice_audio.py`
- Modify: `uv.lock`
- Regenerate: `requirements_labeler.txt`

**Interfaces:**

- Produces: `load_preprocessing_config(path: Path) -> PreprocessingConfig`.
- Produces: immutable audio, spectrogram, rendering, class, source-path, and SHA-256 values.
- Consumes: the exact tracked TOML bytes.

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/data/__init__.py` as an empty file.
Create `tests/data/test_config.py` using this complete valid fixture:

```python
VALID_CONFIG = b'''schema_version = 1

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 0
fmax_hz = 8000
power = 2.0
n_fft = 2048
hop_length = 512

[rendering]
format = "png"
figure_width_inches = 3.2
figure_height_inches = 3.2
dpi = 150
axis_visible = false
interpolation = "nearest"

[classes]
litoria_aurea = 1
non_target = 0
'''
```

Test every loaded value and compare `config.sha256` with `hashlib.sha256(VALID_CONFIG).hexdigest()`.
Add independent tests that replace the schema version, remove the rendering section, make overlap equal chunk duration, make `fmax_hz` exceed Nyquist, and replace a canonical class.
Each invalid test must assert `ConfigError` with the affected field in its message.

- [ ] **Step 2: Run the test and confirm RED**

Run:

```powershell
uv run python -m unittest tests.data.test_config -v
```

Expected: import failure for `frog_classifier` because the package does not exist.

- [ ] **Step 3: Enable package installation**

Add this exact packaging contract to `pyproject.toml` and change `tool.uv.package` to `true`:

```toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/frog_classifier"]
```

Create `src/frog_classifier/__init__.py` with `__version__ = "0.1.0"`.

- [ ] **Step 4: Implement the configuration loader**

Implement these exact public types in `config.py`:

```python
@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    mono: bool
    chunk_seconds: int
    overlap_seconds: int
    drop_incomplete_final_chunk: bool


@dataclass(frozen=True)
class SpectrogramConfig:
    kind: str
    n_mels: int
    fmin_hz: int
    fmax_hz: int
    power: float
    n_fft: int
    hop_length: int


@dataclass(frozen=True)
class RenderingConfig:
    format: str
    figure_width_inches: float
    figure_height_inches: float
    dpi: int
    axis_visible: bool
    interpolation: str


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_version: int
    audio: AudioConfig
    spectrogram: SpectrogramConfig
    rendering: RenderingConfig
    classes: Mapping[str, int]
    source_path: Path
    sha256: str


class ConfigError(ValueError):
    pass


def _section(data: dict[str, object], name: str, keys: set[str]) -> dict[str, object]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid {name} section")
    if set(value) != keys:
        raise ConfigError(f"invalid {name} keys: {sorted(value)}")
    return value


def _value(data: dict[str, object], key: str, expected: type) -> object:
    value = data[key]
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be an integer")
    if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be numeric")
    if expected is bool and not isinstance(value, bool):
        raise ConfigError(f"{key} must be a Boolean")
    if expected is str and not isinstance(value, str):
        raise ConfigError(f"{key} must be text")
    return value


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    source = path.read_bytes()
    try:
        data = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid preprocessing TOML: {error}") from error

    expected_top_level = {"schema_version", "audio", "spectrogram", "rendering", "classes"}
    if set(data) != expected_top_level:
        raise ConfigError(f"invalid top-level keys: {sorted(data)}")
    if _value(data, "schema_version", int) != 1:
        raise ConfigError("schema_version must be 1")

    audio_data = _section(data, "audio", {
        "sample_rate_hz", "mono", "chunk_seconds", "overlap_seconds",
        "drop_incomplete_final_chunk",
    })
    spectrogram_data = _section(data, "spectrogram", {
        "kind", "n_mels", "fmin_hz", "fmax_hz", "power", "n_fft", "hop_length",
    })
    rendering_data = _section(data, "rendering", {
        "format", "figure_width_inches", "figure_height_inches", "dpi",
        "axis_visible", "interpolation",
    })
    classes_data = _section(data, "classes", {"litoria_aurea", "non_target"})

    audio = AudioConfig(
        sample_rate_hz=int(_value(audio_data, "sample_rate_hz", int)),
        mono=bool(_value(audio_data, "mono", bool)),
        chunk_seconds=int(_value(audio_data, "chunk_seconds", int)),
        overlap_seconds=int(_value(audio_data, "overlap_seconds", int)),
        drop_incomplete_final_chunk=bool(_value(audio_data, "drop_incomplete_final_chunk", bool)),
    )
    spectrogram = SpectrogramConfig(
        kind=str(_value(spectrogram_data, "kind", str)),
        n_mels=int(_value(spectrogram_data, "n_mels", int)),
        fmin_hz=int(_value(spectrogram_data, "fmin_hz", int)),
        fmax_hz=int(_value(spectrogram_data, "fmax_hz", int)),
        power=float(_value(spectrogram_data, "power", float)),
        n_fft=int(_value(spectrogram_data, "n_fft", int)),
        hop_length=int(_value(spectrogram_data, "hop_length", int)),
    )
    rendering = RenderingConfig(
        format=str(_value(rendering_data, "format", str)),
        figure_width_inches=float(_value(rendering_data, "figure_width_inches", float)),
        figure_height_inches=float(_value(rendering_data, "figure_height_inches", float)),
        dpi=int(_value(rendering_data, "dpi", int)),
        axis_visible=bool(_value(rendering_data, "axis_visible", bool)),
        interpolation=str(_value(rendering_data, "interpolation", str)),
    )
    classes = {
        "litoria_aurea": int(_value(classes_data, "litoria_aurea", int)),
        "non_target": int(_value(classes_data, "non_target", int)),
    }

    if audio.sample_rate_hz <= 0 or audio.chunk_seconds <= 0:
        raise ConfigError("audio dimensions must be positive")
    if not 0 <= audio.overlap_seconds < audio.chunk_seconds:
        raise ConfigError("overlap must be nonnegative and shorter than the chunk")
    if not audio.mono or audio.overlap_seconds != 0 or not audio.drop_incomplete_final_chunk:
        raise ConfigError("audio settings must match the supported fixed-window contract")
    if not 0 <= spectrogram.fmin_hz < spectrogram.fmax_hz <= audio.sample_rate_hz / 2:
        raise ConfigError("frequency range must not exceed Nyquist")
    if min(
        spectrogram.n_mels,
        spectrogram.n_fft,
        spectrogram.hop_length,
        spectrogram.power,
    ) <= 0:
        raise ConfigError("spectrogram dimensions and power must be positive")
    if min(
        rendering.figure_width_inches,
        rendering.figure_height_inches,
        rendering.dpi,
    ) <= 0:
        raise ConfigError("rendering dimensions and DPI must be positive")
    if spectrogram.kind != "mel" or rendering.format != "png":
        raise ConfigError("only mel spectrograms rendered as png are supported")
    if rendering.axis_visible or rendering.interpolation != "nearest":
        raise ConfigError("rendering settings must match the supported image contract")
    if classes != {"litoria_aurea": 1, "non_target": 0}:
        raise ConfigError("classes must match the canonical mapping")

    return PreprocessingConfig(
        schema_version=1,
        audio=audio,
        spectrogram=spectrogram,
        rendering=rendering,
        classes=MappingProxyType(classes),
        source_path=path.resolve(),
        sha256=hashlib.sha256(source).hexdigest(),
    )
```

Use `tomllib.loads(source.decode("utf-8"))` and SHA-256 over `source`.
Require the exact schema version and class mapping.
Validate positive dimensions, nonnegative frequencies, `fmin_hz < fmax_hz <= sample_rate_hz / 2`, `0 <= overlap_seconds < chunk_seconds`, Mel kind, PNG format, and supported rendering options.
Create `config/preprocessing.toml` from `VALID_CONFIG` and re-export the public types from `data/__init__.py`.

- [ ] **Step 5: Make preprocessing use tracked defaults**

Load the tracked configuration before creating arguments in `scripts/slice_audio.py`.
Use it for the current sample-rate, chunk, Mel-bin, and frequency defaults.
Add power, FFT, hop length, figure dimensions, DPI, and interpolation to its local processing configuration and pass them explicitly to librosa and Matplotlib.
Retain every current argument and override.

- [ ] **Step 6: Refresh and verify dependencies**

Run:

```powershell
uv lock
uv sync --frozen
uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file requirements_labeler.txt
uv run python -m unittest tests.data.test_config -v
uv pip check
```

Expected: all configuration tests pass and pip reports no broken requirements.

- [ ] **Step 7: Commit Task 1**

```powershell
git add pyproject.toml uv.lock requirements_labeler.txt config src tests/data scripts/slice_audio.py
git commit -m "feat: add preprocessing configuration contract"
```

---

### Task 2: Canonical naming and labeled-example discovery

**Files:**

- Create: `src/frog_classifier/data/naming.py`
- Create: `src/frog_classifier/data/validation.py`
- Create: `src/frog_classifier/data/manifest.py`
- Create: `tests/data/helpers.py`
- Create: `tests/data/test_naming.py`
- Create: `tests/data/test_manifest.py`
- Modify: `src/frog_classifier/data/__init__.py`

**Interfaces:**

- Produces: `parse_example_filename(path: Path, chunk_seconds: int) -> ExampleKey`.
- Produces: `discover_labeled_examples(label_root, repo_root, config) -> tuple[LabeledExample, ...]`.
- Produces: stable `ManifestIssue` values and one aggregate `ManifestValidationError`.

- [ ] **Step 1: Write the failing naming tests**

Test these successful calls:

```python
simple = parse_example_filename(Path("20220308_050000_start30s.png"), chunk_seconds=5)
prefixed = parse_example_filename(Path("2MM03935_20251130_180000_start230s.png"), chunk_seconds=5)
self.assertEqual((simple.recording_id, simple.start_s), ("20220308_050000", 30))
self.assertEqual(prefixed.example_id, "2MM03935_20251130_180000_start230s")
```

Test `recording.png`, `_start5s.png`, `recording_start-5s.png`, `recording_start2.5s.png`, `recording_start6s.png`, and `recording_start5s.jpg` as failures.

- [ ] **Step 2: Run naming tests and confirm RED**

Run `uv run python -m unittest tests.data.test_naming -v`.
Expected: import failure for the missing naming module.

- [ ] **Step 3: Implement canonical naming**

Implement:

```python
@dataclass(frozen=True)
class ExampleKey:
    recording_id: str
    start_s: int

    @property
    def example_id(self) -> str:
        return f"{self.recording_id}_start{self.start_s}s"


class ExampleNameError(ValueError):
    pass


def parse_example_filename(path: Path, *, chunk_seconds: int) -> ExampleKey:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    match = re.fullmatch(
        r"(?P<recording_id>.+)_start(?P<start_s>\d+)s\.png",
        path.name,
    )
    if match is None:
        raise ExampleNameError(f"invalid example filename: {path.name}")
    start_s = int(match.group("start_s"))
    if start_s % chunk_seconds:
        raise ExampleNameError(
            f"start time {start_s} is not aligned to {chunk_seconds}-second chunks"
        )
    return ExampleKey(recording_id=match.group("recording_id"), start_s=start_s)
```

Use a full-match expression over `path.name`, require lowercase `.png`, parse a nonnegative integer, reject an empty recording ID, and require alignment to `chunk_seconds`.

- [ ] **Step 4: Run naming tests and confirm GREEN**

Run `uv run python -m unittest tests.data.test_naming -v`.
Expected: all naming tests pass.

- [ ] **Step 5: Write failing discovery tests**

Use temporary roots with empty synthetic PNG files.
Test canonical class mapping, nested relative paths, deterministic path ordering, a missing class directory, a PNG under `labeled/review/`, an unparseable name, and the same example identity in both classes.
Assert stable issue codes for every invalid case.

- [ ] **Step 6: Run discovery tests and confirm RED**

Run `uv run python -m unittest tests.data.test_manifest -v`.
Expected: failures because discovery and aggregate validation do not exist.

- [ ] **Step 7: Implement issues and discovery**

Implement these public types:

```python
@dataclass(frozen=True, order=True)
class ManifestIssue:
    code: str
    message: str
    subject: str = ""


class ManifestValidationError(ValueError):
    def __init__(self, issues: Iterable[ManifestIssue]):
        self.issues = tuple(sorted(issues))
        super().__init__("\n".join(
            f"[{issue.code}] {issue.subject}: {issue.message}"
            for issue in self.issues
        ))


@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    image_path: str
    label: int
    label_name: str
    recording_id: str
    start_s: int


@dataclass(frozen=True)
class ManifestRow:
    manifest_version: int
    example_id: str
    image_path: str
    label: int
    label_name: str
    recording_id: str
    start_s: int
    fold: int
    split: str
    preprocessing_config_sha256: str
```

Scan all case-insensitive PNG suffixes below the root.
Use the first relative component as the exact label name.
Normalize repository-relative paths to `/`.
Aggregate missing directories, unknown directories, naming errors, duplicate case-folded paths, and duplicate example identities.
Return examples sorted by image path only when no issue exists.

- [ ] **Step 8: Verify and commit Task 2**

Run:

```powershell
uv run python -m unittest tests.data.test_naming tests.data.test_manifest -v
git add src/frog_classifier/data tests/data
git commit -m "feat: validate labeled example discovery"
```

Expected: all naming and discovery tests pass before the commit.

---

### Task 3: Deterministic class-aware folds and completed-row validation

**Files:**

- Create: `src/frog_classifier/data/splitting.py`
- Create: `tests/data/test_splitting.py`
- Create: `tests/data/test_validation.py`
- Modify: `src/frog_classifier/data/validation.py`
- Modify: `src/frog_classifier/data/manifest.py`
- Modify: `src/frog_classifier/data/__init__.py`

**Interfaces:**

- Produces: `create_split_plan(examples, folds, seed, test_fold, val_fold) -> SplitPlan`.
- Produces: `build_manifest_rows(examples, plan, config_sha256) -> tuple[ManifestRow, ...]`.
- Produces: `validate_manifest_rows(rows, repo_root, classes, config_sha256, require_files=True) -> None`.

- [ ] **Step 1: Write failing split tests**

Build a synthetic fixture with 15 positive recording groups, 45 negative recording groups, and an additional row in one negative group.
Assert identical plans under the same seed, both labels in every fold and split, and exactly one fold and split per recording ID.
Add independent failures for fewer than five positive groups, fewer than three requested folds, identical test and validation folds, and out-of-range fold selections.

- [ ] **Step 2: Run split tests and confirm RED**

Run `uv run python -m unittest tests.data.test_splitting -v`.
Expected: import failure for the missing splitting module.

- [ ] **Step 3: Implement the split plan**

Implement:

```python
@dataclass(frozen=True)
class SplitPlan:
    folds: int
    seed: int
    test_fold: int
    val_fold: int
    fold_by_recording: Mapping[str, int]
    split_by_recording: Mapping[str, str]


def create_split_plan(
    examples: Sequence[LabeledExample],
    *,
    folds: int = 5,
    seed: int = 1337,
    test_fold: int = 0,
    val_fold: int = 1,
) -> SplitPlan:
    if folds < 3:
        raise ManifestValidationError((ManifestIssue(
            "invalid_fold_count", "at least three folds are required", str(folds)
        ),))
    if test_fold == val_fold or not 0 <= test_fold < folds or not 0 <= val_fold < folds:
        raise ManifestValidationError((ManifestIssue(
            "invalid_fold_selection", "test and validation folds must be distinct and in range"
        ),))

    ordered = sorted(examples, key=lambda row: (row.recording_id, row.start_s, row.image_path))
    groups_by_label: dict[int, set[str]] = defaultdict(set)
    for example in ordered:
        groups_by_label[example.label].add(example.recording_id)
    issues = tuple(
        ManifestIssue(
            "insufficient_class_groups",
            f"label {label} has {len(groups)} groups but {folds} are required",
            str(label),
        )
        for label, groups in sorted(groups_by_label.items())
        if len(groups) < folds
    )
    if issues:
        raise ManifestValidationError(issues)

    labels = [example.label for example in ordered]
    groups = [example.recording_id for example in ordered]
    assignments = [-1] * len(ordered)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split([0] * len(ordered), labels, groups)):
        for index in held_out:
            assignments[int(index)] = fold

    expected_labels = set(groups_by_label)
    fold_issues = []
    for fold in range(folds):
        observed = {labels[index] for index, assigned in enumerate(assignments) if assigned == fold}
        if observed != expected_labels:
            fold_issues.append(ManifestIssue(
                "fold_missing_class", f"fold has labels {sorted(observed)}", str(fold)
            ))
    if fold_issues:
        raise ManifestValidationError(fold_issues)

    fold_by_recording: dict[str, int] = {}
    for example, fold in zip(ordered, assignments, strict=True):
        previous = fold_by_recording.setdefault(example.recording_id, fold)
        if previous != fold:
            raise AssertionError("StratifiedGroupKFold split one recording group")
    split_by_recording = {
        recording_id: "test" if fold == test_fold else "val" if fold == val_fold else "train"
        for recording_id, fold in fold_by_recording.items()
    }
    return SplitPlan(
        folds=folds,
        seed=seed,
        test_fold=test_fold,
        val_fold=val_fold,
        fold_by_recording=MappingProxyType(fold_by_recording),
        split_by_recording=MappingProxyType(split_by_recording),
    )
```

Sort examples by `(recording_id, start_s, image_path)`.
Require at least `folds` distinct recording groups for each label.
Use `StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)`.
Assign held-out indexes to folds, validate both labels in each fold, and freeze result mappings with `MappingProxyType`.

- [ ] **Step 4: Run split tests and confirm GREEN**

Run `uv run python -m unittest tests.data.test_splitting -v`.
Expected: every split test passes without a scikit-learn warning.

- [ ] **Step 5: Write failing completed-row validation tests**

Create a valid synthetic row set and corrupt one property per test.
Cover a missing file, wrong manifest version, wrong numeric label, invalid split, invalid fold, invalid SHA-256, duplicate identity, duplicate path including case-only variants, recording fold leakage, recording split leakage, missing fold number, a fold lacking one class, and a split lacking one class.
Assert one stable issue code for each corruption.

- [ ] **Step 6: Run validation tests and confirm RED**

Run `uv run python -m unittest tests.data.test_validation -v`.
Expected: failure because completed-row validation is missing.

- [ ] **Step 7: Implement row creation and validation**

`build_manifest_rows` must map each example through the split plan, set manifest version `1`, and record the lowercase configuration digest.
`validate_manifest_rows` must aggregate all issues and raise once.
It must require existing files when requested, exact class mapping, valid hashes, unique identities and paths, contiguous fold numbers, one fold and split per recording ID, exactly one test fold, exactly one validation fold, one or more training folds, and both classes in every fold and split.

- [ ] **Step 8: Verify and commit Task 3**

Run:

```powershell
uv run python -m unittest discover -s tests/data -v
git add src/frog_classifier/data tests/data
git commit -m "feat: add leakage-safe grouped folds"
```

Expected: every data unit test passes before the commit.

---

### Task 4: Versioned CSV, quality reports, and manifest command

**Files:**

- Create: `src/frog_classifier/data/reporting.py`
- Create: `tests/data/test_reporting.py`
- Create: `tests/data/test_manifest_cli.py`
- Modify: `src/frog_classifier/data/manifest.py`
- Modify: `src/frog_classifier/data/__init__.py`
- Replace: `scripts/build_manifest.py`
- Create: `results/data_quality/.gitkeep`

**Interfaces:**

- Produces: `serialize_manifest(rows) -> bytes`.
- Produces: `load_manifest(path, repo_root, config) -> tuple[ManifestRow, ...]`.
- Produces: stable JSON and Markdown from one `DataQualityReport`.
- Produces: `write_output_bundle(outputs: Mapping[Path, bytes]) -> None`.

- [ ] **Step 1: Write failing CSV tests**

Assert this exact header, UTF-8 bytes, deterministic row order, and successful round-trip loading:

```python
MANIFEST_COLUMNS = (
    "manifest_version",
    "example_id",
    "image_path",
    "label",
    "label_name",
    "recording_id",
    "start_s",
    "fold",
    "split",
    "preprocessing_config_sha256",
)
```

Add failure cases for missing columns, extra columns, invalid integers, unsupported versions, invalid hashes, and missing image files.

- [ ] **Step 2: Run CSV tests and confirm RED**

Run `uv run python -m unittest tests.data.test_manifest -v`.
Expected: failures for missing serialization and loading functions.

- [ ] **Step 3: Implement strict CSV serialization and loading**

Use `io.StringIO(newline="")`, `csv.DictWriter`, the exact header constant, and UTF-8 encoding.
Require `reader.fieldnames == list(MANIFEST_COLUMNS)`.
Parse every typed value, aggregate row parse issues, and call `validate_manifest_rows` before returning rows.

- [ ] **Step 4: Write failing report tests**

Use a known synthetic row set and assert exact example and group totals by class, fold, and split.
Assert seed, fold selections, configuration checksum, manifest checksum, validation check names, and provenance note.
Serialize twice and assert byte equality for JSON and Markdown.
Assert neither serialization contains a wall-clock timestamp.

- [ ] **Step 5: Run report tests and confirm RED**

Run `uv run python -m unittest tests.data.test_reporting -v`.
Expected: import failure for the report implementation.

- [ ] **Step 6: Implement stable reports and safe output replacement**

Define a frozen `DataQualityReport` whose fields contain only dataclasses, tuples, strings, integers, Booleans, and sorted dictionaries.
Use this JSON contract:

```python
payload = json.dumps(
    asdict(report),
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
) + "\n"
```

Render Markdown headings and tables in a fixed order with no timestamp.
`write_output_bundle` must write one flushed temporary file per destination in the same directory, close every temporary file, and replace destinations only after all temporary writes succeed.
Clean up only temporary files created by the function when preparation fails.

- [ ] **Step 7: Write the failing end-to-end command tests**

Run the command through `subprocess.run` against a temporary synthetic tree with enough groups for five folds.
Pass absolute paths for training root, CSV, report directory, and configuration.
Assert exit status `0`, all three outputs, both classes in every fold, and byte-identical second-run outputs.
Add one invalid tree with both an unknown directory and malformed filename.
Assert status `2`, both issue codes in stderr, and no output replacement.

- [ ] **Step 8: Replace the command with a thin wrapper**

Support these defaults and options:

```text
--training-root labeled
--out-csv labeled/manifest.csv
--report-dir results/data_quality
--config config/preprocessing.toml
--seed 1337
--folds 5
--test-fold 0
--val-fold 1
```

Resolve relative paths against the repository root.
Call package interfaces for every algorithm.
Print output paths and concise class and group counts.
Catch `ConfigError` and `ManifestValidationError`, print every issue to stderr, and return status `2`.
Do not retain the obsolete fractional split arguments.

- [ ] **Step 9: Verify and commit Task 4**

Run:

```powershell
uv run python -m unittest tests.data.test_manifest tests.data.test_reporting tests.data.test_manifest_cli -v
uv run python scripts/build_manifest.py --help
git add src/frog_classifier/data scripts/build_manifest.py tests/data results/data_quality/.gitkeep
git commit -m "feat: generate validated manifests and reports"
```

Expected: all tests pass and help shows fold, report, and configuration options.

---

### Task 5: Strict baseline input

**Files:**

- Create: `tests/test_train_baseline.py`
- Modify: `scripts/train_baseline.py`

**Interfaces:**

- Consumes: `load_manifest(path, repo_root, config) -> tuple[ManifestRow, ...]`.
- Preserves: current image featurization, logistic regression, and metric output after validation.

- [ ] **Step 1: Write failing baseline tests**

Import the baseline script with `importlib.util.spec_from_file_location`.
Build one valid synthetic manifest with tiny real PNG images.
Add invalid cases for a missing image, empty validation partition, single-class test partition, and one recording ID assigned to train and test.
Patch `LogisticRegression.fit` only in the valid-input test so input behavior is isolated from convergence.
Assert every invalid case raises `ManifestValidationError` and never calls `fit`.

- [ ] **Step 2: Run baseline tests and confirm RED**

Run `uv run python -m unittest tests.test_train_baseline -v`.
Expected: current missing-file filtering and all-data fallback make the new tests fail for their intended reasons.

- [ ] **Step 3: Make baseline input strict**

Remove its local CSV loader.
Load `config/preprocessing.toml` and call the package manifest loader.
Convert validated rows into local `Example` values.
Delete the missing-path filter, the all-data fallback, and optional validation or test behavior.
Require and evaluate train, validation, and test after package validation.
Do not alter the logistic-regression settings in this phase.

- [ ] **Step 4: Verify and commit Task 5**

Run:

```powershell
uv run python -m unittest tests.test_train_baseline -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/train_baseline.py
git add scripts/train_baseline.py tests/test_train_baseline.py
git commit -m "fix: require valid baseline evaluation data"
```

Expected: baseline tests pass and every script compiles.

---

### Task 6: Project navigation and storage cleanup

**Files:**

- Create: `README.md`
- Move: `PROCESS.md` to `docs/workflow.md`
- Move: `docs/system design.md` to `docs/architecture.md`
- Modify: `.gitignore`
- Modify: `tests/test_repository_reproducibility.py`
- Delete: six tracked `dataset/**/.gitkeep` files
- Create: `results/data_quality/.gitkeep`

**Interfaces:**

- Produces: one root entry point and two focused supporting documents.
- Preserves: all current labeling and preprocessing commands.

- [ ] **Step 1: Write failing organization tests**

Require `README.md`, `docs/architecture.md`, `docs/workflow.md`, `config/preprocessing.toml`, all data-package modules, and `results/data_quality/.gitkeep`.
Remove dataset placeholders and dataset generated paths from expected layout.
Remove `src/` and `src\\` from stale-token assertions because `src` is now canonical.
Assert `PROCESS.md`, `docs/system design.md`, and tracked dataset placeholders do not exist.
Check that relative Markdown link targets exist in README, roadmap, architecture, and workflow.

- [ ] **Step 2: Run organization tests and confirm RED**

Run `uv run python -m unittest tests.test_repository_reproducibility -v`.
Expected: failures for the missing README, old document locations, and dataset placeholders.

- [ ] **Step 3: Confirm dataset contains no user files**

Run:

```powershell
Get-ChildItem -LiteralPath dataset -Recurse -File | Where-Object Name -ne '.gitkeep'
```

Expected: no output.
If any file appears, stop and ask the user before deleting or moving anything under `dataset/`.

- [ ] **Step 4: Reorganize documentation and storage**

Use Git-aware moves for the existing documents.
Create README with project purpose, current checkpoint, pipeline, setup, common commands, and links to roadmap, architecture, and workflow.
Update architecture and workflow for the package, versioned manifest, folds, reports, configuration, and strict baseline.

Include this exact policy in README and workflow:

```text
Use Frog only when the target call is confidently present, even if it is faint.
Use Background only when the clip is confidently non-target.
Use Skip when identification is uncertain, and do not convert uncertainty into a negative label.
Phase 3 will add an explicit review-later state and signal-quality metadata.
```

Remove dataset rules from `.gitignore` only after the empty-tree check passes.
Keep reports ignored and keep the general `.gitkeep` exception.

- [ ] **Step 5: Verify and commit Task 6**

Run:

```powershell
uv run python -m unittest tests.test_repository_reproducibility -v
rg -n "PROCESS\.md|system design\.md|dataset/|dataset\\\\" README.md roadmap.md docs scripts tests src
git add README.md .gitignore docs tests/test_repository_reproducibility.py results/data_quality/.gitkeep dataset
git commit -m "docs: clarify project data workflow"
```

Expected: tests pass and no current operational reference uses the old locations.
Historical superpowers specifications and plans may retain old-path context.

---

### Task 7: Real-data acceptance and milestone completion

**Files:**

- Modify: `roadmap.md`
- Modify: `docs/workflow.md`
- Generate and ignore: `labeled/manifest.csv`
- Generate and ignore: `results/data_quality/manifest-report.json`
- Generate and ignore: `results/data_quality/manifest-report.md`

**Interfaces:**

- Validates: the complete workflow against the current 61 labels without changing them.
- Records: Phase 2 completion only after every acceptance check passes.

- [ ] **Step 1: Capture local data counts**

Count raw WAV files, queued spectrogram PNGs, positive labels, negative labels, model files, and result files excluding `.gitkeep` and the expected generated reports.
Record counts in the command transcript, not in a new tracked file.
Expected label counts are 13 positive and 48 negative.

- [ ] **Step 2: Generate the real manifest and reports**

Run:

```powershell
uv run python scripts/build_manifest.py
Get-Content -LiteralPath results/data_quality/manifest-report.md
```

Expected: 61 examples, 60 recording groups, five folds, both classes in every fold and split, and no validation issue.

- [ ] **Step 3: Prove deterministic output**

Calculate SHA-256 for the CSV and both reports.
Run the command again and calculate the three hashes again.
Expected: all hashes are unchanged.

- [ ] **Step 4: Run the baseline end to end**

Run `uv run python scripts/train_baseline.py`.
Expected: train, validation, and test metrics appear and no fallback or missing-partition message appears.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
uv lock --check
uv sync --frozen
uv pip check
uv run python -m unittest discover -s tests -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/train_baseline.py
git diff --check
```

Expected: dependencies are valid, all tests pass, all scripts compile, and Git reports no whitespace error.

- [ ] **Step 6: Confirm data counts remain unchanged**

Repeat Step 1 counts.
Expected: raw recordings, queued spectrograms, positive labels, negative labels, and models are unchanged.
Only the ignored manifest and two ignored reports may be new or replaced.

- [ ] **Step 7: Mark the verified milestone**

Change Phase 2 in `roadmap.md` from `Status: Next` to `Status: Complete`.
Change Phase 3 from `Status: Planned` to `Status: Next`.
Record the verified test count, manifest counts, fold class counts, and deterministic-output result.
Update workflow with the exact successful commands and output paths.

- [ ] **Step 8: Run final text and repository checks**

Run the full tests, `git diff --check`, a prohibited em-dash scan over changed text files, and `git status --short`.
Expected: all checks pass and only intended Phase 2 changes are tracked.

- [ ] **Step 9: Commit the milestone**

```powershell
git add roadmap.md docs/workflow.md
git commit -m "docs: complete trustworthy evaluation phase"
```

- [ ] **Step 10: Request code review**

Use the `superpowers:requesting-code-review` workflow over the complete Phase 2 commit range.
Resolve only findings within the approved scope and rerun every verification command.

---

## Completion evidence

Report all of the following before claiming success:

- Final branch and commit range.
- Automated test count with zero failures.
- Real manifest example and recording-group counts.
- Per-fold and per-split class counts.
- Stable SHA-256 values across two generations.
- Unchanged raw, queued, positive-label, negative-label, and model counts.
- Exact generated output paths.
- The remaining statistical limitation caused by the small positive label set.
