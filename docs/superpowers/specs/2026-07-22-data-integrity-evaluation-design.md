# Data Integrity and Trustworthy Evaluation Design

## Summary

Phase 2 will reorganize the data and evaluation pipeline into a small reusable Python package.
The package will make labeled-example discovery, filename parsing, manifest construction, grouped splitting, validation, configuration, and reporting independently understandable and testable.
The working preprocessing and labeling interfaces will retain their current behavior.

The manifest will remain the source of truth for training and evaluation membership.
The unused `dataset/` split tree will be removed because it currently suggests duplicated data that does not exist.

## Goals

- Give each data-pipeline responsibility one clear module and public interface.
- Reject malformed, ambiguous, incomplete, or leakage-prone data before writing outputs.
- Produce deterministic recording-group folds that contain both canonical classes.
- Preserve a convenient default train, validation, and test split without limiting later cross-validation.
- Record the preprocessing contract used to interpret labeled examples.
- Generate reports that explain exactly which examples and recording groups enter each partition.
- Give new contributors a clear root entry point and a predictable documentation structure.
- Keep all tests runnable without the private audio or spectrogram collection.

## Non-goals

- This phase will not train or save a production model.
- This phase will not introduce transfer learning, batch inference, analytics, or the classifier application.
- This phase will not redesign the Streamlit or Matplotlib labeling interfaces.
- This phase will not relabel examples or alter raw recordings, generated spectrograms, or existing human label decisions.
- This phase will not claim statistically stable model performance from the current small positive label set.

## Current-state findings

The repository currently places manifest discovery, filename parsing, split assignment, CSV serialization, and command-line behavior in `scripts/build_manifest.py`.
The current split protects recording groups from direct leakage, but it shuffles groups without considering their labels.
With the current small label set, validation or test can therefore omit `litoria_aurea` entirely.

The current local label set contains 13 `litoria_aurea` examples and 48 `non_target` examples across 60 recording groups.
Five stratified group folds are feasible with this distribution, although the resulting metrics will still have high uncertainty.

The baseline trainer silently falls back to treating all examples as training data when its training partition is too small.
That behavior can conceal invalid evaluation data and will be removed.

The executable preprocessing command currently defaults to 22,050 Hz mono audio, five-second non-overlapping chunks, 128 Mel bins, and a 0 Hz to 8,000 Hz frequency range.
These values are described in documentation but are not yet represented by a shared machine-readable contract.

The empty `dataset/train`, `dataset/val`, and `dataset/test` directories duplicate a partition concept that is already represented by the manifest.
No current command materializes those directories.

## Architecture

The new package will use a `src` layout and will be installed by the project environment.
Existing scripts will remain as thin, familiar command-line entry points.

```text
src/
  frog_classifier/
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
scripts/
  build_manifest.py
  train_baseline.py
tests/
  data/
    __init__.py
    test_config.py
    test_naming.py
    test_manifest.py
    test_splitting.py
    test_validation.py
    test_reporting.py
    test_manifest_cli.py
```

`pyproject.toml` will define the build backend and install the package from `src/frog_classifier`.
Hatchling will provide the build backend, and its resolved build requirement will be recorded in `uv.lock` without becoming a runtime dependency.
The project will continue to support Python 3.13 and the existing locked runtime dependencies.

### `config.py`

This module will load and validate the tracked preprocessing TOML file.
It will expose an immutable `PreprocessingConfig` value containing every declared setting and the SHA-256 checksum of the source file.
It will reject missing sections, unknown schema versions, invalid numeric ranges, and internally inconsistent frequency settings.

### `naming.py`

This module will own the canonical spectrogram filename contract.
It will expose an immutable `ExampleKey` containing `recording_id` and `start_s`.
The parser will accept names in the form `<recording_id>_start<non-negative-integer>s.png` and reject all other forms.
It will also require `start_s` to be aligned to the configured chunk duration.

### `manifest.py`

This module will discover labeled PNG files and convert them into immutable `LabeledExample` and `ManifestRow` values.
It will normalize repository-relative paths to POSIX separators and produce rows in deterministic sorted order.
It will be responsible for CSV serialization and validated CSV loading.

### `splitting.py`

This module will assign complete recording groups to deterministic stratified folds.
It will use scikit-learn's `StratifiedGroupKFold` with shuffling and an explicit seed.
It will expose a `SplitPlan` that maps every recording ID to one fold and one default split.

### `validation.py`

This module will hold reusable validation rules and an aggregate `ManifestValidationError`.
Errors will contain stable issue codes, affected paths or keys, and actionable messages.
The command-line interface will print all discovered issues in deterministic order instead of forcing users through one failure per run.

### `reporting.py`

This module will build one structured `DataQualityReport` and serialize it to both JSON and Markdown.
The JSON report will be suitable for later automation.
The Markdown report will be optimized for human review.

### Command-line wrappers

`scripts/build_manifest.py` will parse command-line arguments, call package interfaces, print a concise summary, and translate validation failures into a nonzero exit status.
It will contain no parsing, splitting, or reporting algorithms.

`scripts/train_baseline.py` will load the manifest through the package validator.
It will retain its current sanity-baseline model but will refuse missing files, invalid schemas, incomplete splits, single-class partitions, or group leakage.

## Preprocessing contract

`config/preprocessing.toml` will have schema version `1` and declare the following current behavior:

```toml
schema_version = 1

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
```

The explicit FFT and hop-length values match the defaults currently supplied by the locked librosa version.
Making them explicit prevents a future dependency change from silently altering the feature representation.

`scripts/slice_audio.py` will load these values as its defaults while preserving its existing command-line overrides and output behavior.
The manifest command will record the selected preprocessing configuration checksum.
This checksum records the declared contract but does not retroactively prove how every existing PNG was generated.
The quality report will state that limitation explicitly for pre-existing artifacts.

## Labeled-example discovery

The label root is expected to contain the canonical `litoria_aurea` and `non_target` directories.
A PNG may be located directly in a class directory or in a nested directory beneath it.
The first path component beneath the label root determines its class.

Discovery will scan all PNG files beneath the label root, not only the two expected directories.
A PNG outside a canonical class directory will produce an `unknown_label_directory` issue instead of being ignored.
Missing canonical class directories will produce `missing_label_directory` issues.

Every discovered path must exist as a regular file and must be expressible relative to the repository root.
The numeric label must match the class mapping in the preprocessing configuration.

## Filename and identity contract

The canonical example identity is the pair `(recording_id, start_s)`.
The canonical string `example_id` is `<recording_id>_start<start_s>s`.

The parser will reject an empty recording ID, a missing start suffix, a negative or non-integer start, a non-PNG extension, and a start time that is not divisible by the configured chunk duration.

The manifest will reject duplicate normalized paths.
It will also reject duplicate example identities, including duplicates placed in opposite class directories.
This prevents one recording window from receiving contradictory labels or silently appearing more than once.

The current labeler flattens labeled files into class directories, so the filename cannot preserve a source-root namespace.
Phase 2 will protect this limitation by rejecting collisions.
A future labeling enhancement may preserve source-relative directories without changing the manifest identity rules.

## Fold and split strategy

The default fold count will be five and the default seed will remain `1337`.
The command will accept explicit `--folds` and `--seed` values.
At least three folds are required because the default view needs separate training, validation, and test membership.

Rows will be sorted before fold assignment so filesystem traversal order cannot affect results.
All rows sharing a recording ID will be passed as one group to `StratifiedGroupKFold`.

Before assignment, each class must occur in at least as many distinct recording groups as the requested fold count.
After assignment, every fold must contain both canonical classes.
Every recording group must belong to exactly one fold.

The default split view will use fold `0` as `test`, fold `1` as `val`, and all remaining folds as `train`.
The command will accept distinct `--test-fold` and `--val-fold` values within range.

The resulting `fold` column supports later group-aware cross-validation.
The `split` column preserves the simple interface expected by the current baseline trainer.

## Manifest schema

The CSV will use this exact column order:

```text
manifest_version,example_id,image_path,label,label_name,recording_id,start_s,fold,split,preprocessing_config_sha256
```

`manifest_version` will be `1`.
`image_path` will be repository-relative and will use `/` separators.
`label` and `label_name` must agree with the tracked class mapping.
`fold` will be an integer in the configured range.
`split` will be one of `train`, `val`, or `test`.
`preprocessing_config_sha256` will contain the lowercase SHA-256 digest of the selected TOML file.

Manifest loading will reject missing columns, additional columns, unsupported versions, invalid values, nonexistent image paths, duplicate examples, split-to-fold inconsistencies, and recording-group leakage.

## Output transaction and failure behavior

The complete input will be discovered, parsed, split, and validated in memory before any destination is replaced.
The CSV, JSON report, and Markdown report will first be written to temporary files in their respective destination directories.
Each completed temporary file will then replace its destination.

The default outputs will be:

```text
labeled/manifest.csv
results/data_quality/manifest-report.json
results/data_quality/manifest-report.md
```

Generated outputs will remain ignored by Git.
`results/data_quality/.gitkeep` will preserve the intended report directory in a fresh clone.

Validation failures will not replace any existing output.
The command will write a concise heading and all issues to standard error and exit with status `2`.
Unexpected I/O failures will retain normal exception context and exit unsuccessfully.

## Data-quality report

The JSON and Markdown reports will describe the same `DataQualityReport` value.
They will include:

- Report schema version
- Generation seed and fold configuration
- Manifest version and SHA-256 checksum
- Preprocessing configuration path and SHA-256 checksum
- Total example count
- Total distinct recording-group count
- Example and recording-group counts by class
- Example and recording-group counts by fold
- Example and recording-group counts by split
- Class counts inside every fold and split
- Validation checks and their pass status
- A provenance note for pre-existing artifacts

Dictionary keys and table rows will be emitted in stable order.
Repeated runs against unchanged inputs and configuration will produce byte-identical CSV and JSON outputs.
The Markdown report may include its output-generation context but will not include a volatile wall-clock timestamp.

## Baseline behavior

The baseline trainer will use the package manifest loader before reading images.
It will require nonempty `train`, `val`, and `test` partitions.
Each partition must contain both canonical classes.
No recording ID may appear in more than one split.

The fallback that replaces an undersized training split with the complete dataset will be deleted.
Missing image files will be validation errors rather than silently filtered rows.

This phase does not change the logistic-regression algorithm or claim production-quality metrics.
It only makes the inputs and reported partitions structurally trustworthy.

## Documentation organization

The repository will use the following human-facing documentation structure:

```text
README.md
roadmap.md
docs/
  architecture.md
  workflow.md
```

`README.md` will be the entry point for project purpose, current status, the pipeline at a glance, setup, common commands, and links to deeper documents.
`roadmap.md` will continue to describe sequencing and acceptance criteria.
`docs/architecture.md` will replace `docs/system design.md` and describe verified component contracts and data flow.
`docs/workflow.md` will replace `PROCESS.md` and contain operational procedures and reproducibility commands.

Every command shown in documentation will be exercised by a test or a verification step.
Documentation will distinguish generated private data from tracked source code and configuration.

## Test strategy

Tests will use Python's standard `unittest` framework and temporary directories.
They will create empty synthetic PNG files where image decoding is not part of the behavior under test.
No test will require the private recordings, generated spectrogram collection, or existing local labels.

### Configuration tests

- Load the tracked schema version and all expected values.
- Reject missing sections, unsupported versions, invalid chunk settings, and invalid frequency ranges.
- Produce a stable checksum for unchanged bytes.

### Naming tests

- Parse representative short and prefixed recording IDs.
- Reject malformed, negative, non-integer, misaligned, and non-PNG names.
- Produce stable example identities.

### Discovery and manifest tests

- Map canonical directories to the correct numeric and string labels.
- Normalize nested relative paths.
- Reject unknown class directories and missing class directories.
- Reject duplicate example identities within one class and across classes.
- Emit deterministic row ordering and the exact versioned schema.

### Splitting and validation tests

- Produce identical folds for identical rows and seeds.
- Keep every recording group in exactly one fold and split.
- Put both classes in every fold and default split.
- Reject insufficient per-class recording groups.
- Reject invalid fold selections and split-to-fold inconsistencies.
- Detect leakage in a deliberately corrupted manifest.

### Reporting tests

- Verify all example, group, class, fold, and split counts.
- Verify stable JSON serialization and the expected Markdown tables.
- Verify manifest and configuration checksums.
- Verify that reports contain no volatile timestamp.

### End-to-end command tests

- Run the manifest command against a synthetic labeled tree.
- Verify the CSV and both reports are created together.
- Run the command twice and verify deterministic outputs.
- Inject multiple simultaneous validation problems and verify all are reported with exit status `2`.
- Verify failed generation leaves existing outputs unchanged.

### Baseline tests

- Accept a complete validated synthetic manifest.
- Reject a missing image, a missing partition, a single-class partition, and recording leakage.
- Verify that no all-data training fallback remains.

## Migration and compatibility

The existing manifest command will remain:

```powershell
uv run python scripts/build_manifest.py
```

The baseline command will remain:

```powershell
uv run python scripts/train_baseline.py
```

The old `--val-frac` and `--test-frac` manifest arguments will be removed because fold selection replaces fractional unstratified splitting.
The new command will explain the replacement when an obsolete argument is supplied by normal argument parsing.

The `dataset/` placeholder tree will be removed from Git and from documentation.
No user data will be deleted because those directories currently contain only tracked placeholders and ignored content is not expected or used by any command.
Before removal, implementation verification will confirm that no non-placeholder local files exist beneath `dataset/`.

## Acceptance criteria

- The reusable data package is importable from the locked environment.
- A synthetic end-to-end run produces a versioned manifest and matching JSON and Markdown reports.
- Repeated runs with identical inputs, seed, folds, and configuration are deterministic.
- All discovered labeled PNGs are either represented exactly once or cause a clear failure.
- No recording group crosses folds or default splits.
- Every fold and default split contains both canonical classes.
- Invalid or ambiguous input leaves existing outputs unchanged and exits with status `2`.
- The baseline refuses invalid evaluation data and has no fallback that trains on all examples.
- The tracked preprocessing configuration matches current command defaults.
- The root README and focused documentation make the current workflow and component boundaries discoverable.
- All repository tests pass without access to private data.
- Local raw, processed, and labeled example counts remain unchanged.

## Known limitations after Phase 2

Structural validation cannot make the current small label set statistically representative.
Phase 3 must expand positive and hard-negative labels across independent recording groups and field conditions.

The preprocessing checksum records a declared configuration rather than cryptographic provenance for PNGs generated before this feature exists.
New generation workflows can preserve stronger provenance in a later focused enhancement.

The flattened labeled filename layout cannot distinguish identical recording stems originating from different source roots.
Phase 2 detects and rejects resulting identity collisions rather than silently merging them.
