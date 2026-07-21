# Frog Classifier Roadmap

This roadmap turns the current preprocessing and labeling prototype into a trustworthy frog-call detection system.
Work is ordered so data integrity and evaluation quality are established before model complexity or application development increases.

## Current checkpoint

Verified on July 22, 2026:

- Raw recordings: 618
- Duration per recording: five minutes
- Total five-second examples: 37,080
- Spectrograms awaiting review: 37,019
- Positive `litoria_aurea` labels: 13
- Negative `non_target` labels: 48
- Unique labeled recording groups: 60
- Saved models: 0
- Inference or analytics result artifacts: 0
- Active Git branch: `modern`

Preprocessing accounts for every expected five-second example.
The project is currently limited by label coverage and an evaluation split that can omit the positive class from validation or test data.

## Working principles

- Protect raw recordings as immutable source material.
- Prevent recording-level leakage in every evaluation method.
- Prefer trustworthy data and metrics over additional model complexity.
- Preserve the preprocessing configuration with every trained model.
- Treat recall, false-positive rate, and threshold behavior as first-class results.
- Keep generated data, trained models, and results reproducible but outside normal Git history.
- Mark roadmap work complete only after its acceptance checks pass.

## Phase 1: Repository reproducibility

Status: Complete

### Purpose

Make a fresh clone reproduce the intended project structure, Python environment, commands, and class vocabulary.

### Work

- Track the empty `raw`, `processed`, `labeled`, `dataset`, `models`, and `results` skeleton with `.gitkeep` files.
- Keep all generated content beneath those roots ignored by Git.
- Standardize stored class names on `litoria_aurea` and `non_target`.
- Declare direct dependencies and Python 3.13 support in `pyproject.toml`.
- Lock the exact environment in `uv.lock`.
- Generate the pip-compatible `requirements_labeler.txt` used by the Windows launcher.
- Align operational documentation and script help text with the current layout.
- Add automated repository reproducibility checks.

### Acceptance criteria

- Every intended placeholder exists and is trackable.
- Representative audio, image, model, and result files remain ignored.
- `uv lock --check` succeeds.
- The compatibility requirements export regenerates deterministically.
- All operational paths and commands refer to files that exist.
- Repository tests, Python compilation, and dependency checks pass.
- Local data and artifact counts remain unchanged.

## Phase 2: Data integrity and trustworthy evaluation

Status: Complete

### Purpose

Ensure every reported metric is based on valid, leakage-safe data containing both classes.

### Work

- Replace the unstratified recording-group split with a class-aware grouped strategy.
- Keep every recording ID in exactly one partition.
- Reject any validation or test split that lacks `litoria_aurea` or `non_target` examples.
- Use group-aware cross-validation while the number of positive recording groups remains small.
- Add tests for filename parsing, recording ID extraction, relative path mapping, and class labels.
- Protect manifest generation from duplicate filenames and unparseable labeled files.
- Report row counts, recording-group counts, and class counts for every partition.
- Decide whether `dataset/` remains a reserved layout or becomes a materialized view generated from the manifest.
- Record preprocessing parameters in a machine-readable configuration shared by training and inference.

### Acceptance criteria

- Automated tests prove that no recording group crosses partitions.
- Every evaluation fold contains both classes.
- Manifest creation fails clearly on invalid or ambiguous input instead of silently skipping it.
- A generated data-quality report explains exactly what enters training and evaluation.
- Repeated runs with the same seed produce the same validated split.

### Verified acceptance

Verified on July 22, 2026:

- The complete repository suite passed 66 tests with zero failures.
- The real manifest contains 61 examples from 60 recording groups: 13 `litoria_aurea` examples from 13 groups and 48 `non_target` examples from 47 groups.
- Fold class counts for `litoria_aurea` and `non_target` examples are fold 0: 3 and 10, fold 1: 3 and 9, fold 2: 2 and 10, fold 3: 2 and 10, and fold 4: 3 and 9.
- Split class counts for `litoria_aurea` and `non_target` examples are train: 7 and 29, validation: 3 and 9, and test: 3 and 10.
- Two consecutive manifest generations produced identical SHA-256 values: `6fb0b3bacf9ab6371cbaeeff9d74fa238e9523cadd0921bc4f2aa298b765e831` for the CSV, `d272255ba1315108792bce3c4a1d769b776075eae9b9f173cda7b09562753944` for the JSON report, and `5fa127a72fdce136cce97b2530086bf652f7e8835a4b23c58a96185ce7ef72ce` for the Markdown report.
- The strict baseline produced train, validation, and test metrics without a fallback or missing-partition path.
- The 13 positive examples leave only two or three positives in each fold, so grouped validation metrics remain statistically unstable until Phase 3 expands positive label coverage.

## Phase 3: Purposeful label expansion

Status: Next

### Purpose

Build a diverse label set that represents the real conditions in which the detector will operate.

### Work

- Prioritize positive candidates across distinct recordings, dates, sites, weather, and noise conditions.
- Use the known calling-time notes to create review queues instead of sampling all hours uniformly.
- Label hard negatives such as insects, birds, wind, water, equipment noise, and other frog species.
- Add an explicit uncertain or review-later workflow without treating uncertain examples as negatives.
- Track labeling progress by class and recording group, not only by image count.
- Review a sample of labels twice to estimate consistency and correct systematic mistakes.
- Prevent near-duplicate chunks from dominating any class.
- Preserve enough held-out recording groups for later evaluation.

### Acceptance criteria

- Positive examples span enough independent recording groups for stable grouped evaluation.
- Each major field condition has representative positive and negative examples.
- Label audits identify and resolve uncertain or contradictory decisions.
- Class and group distributions are documented before training begins.

## Phase 4: Reproducible model training and persistence

Status: Planned

### Purpose

Train a model whose performance, preprocessing, and artifacts can be reproduced and compared.

### Work

- Retain the logistic-regression baseline as a pipeline and leakage sanity check.
- Establish a stronger image baseline with transfer learning, initially using a compact pretrained architecture such as ResNet-18.
- Apply training-only augmentation appropriate to spectrograms.
- Evaluate precision, recall, F1, confusion matrices, precision-recall curves, and false positives per recording hour.
- Select detection thresholds using validation data rather than a fixed arbitrary confidence.
- Measure variation across grouped folds or repeated seeds.
- Save model weights, class mapping, preprocessing parameters, training configuration, dependency lock reference, and metrics together.
- Compare candidate models against the simple baseline and reject complexity that does not improve held-out performance.

### Acceptance criteria

- A saved artifact can reproduce evaluation predictions from a clean environment.
- The model has documented performance on unseen recording groups containing both classes.
- Threshold selection and expected false-positive behavior are explicit.
- Training runs record configuration, seed, data manifest, and output checksums.

## Phase 5: Batch inference and structured detections

Status: Planned

### Purpose

Apply the validated model to new long recordings and preserve useful detection provenance.

### Work

- Reuse the exact training preprocessing configuration during inference.
- Scan nested recording directories in deterministic five-second windows.
- Batch model calls efficiently while keeping memory bounded.
- Record source path, recording timestamp, window start, window end, predicted class, confidence, model version, and preprocessing version.
- Write detections to CSV initially and add SQLite when query volume justifies it.
- Support resumable processing without duplicating completed detections.
- Log unreadable audio, malformed timestamps, and failed files without losing the rest of a batch.
- Measure throughput and GPU or CPU utilization on representative recordings.

### Acceptance criteria

- A known recording produces repeatable detections with traceable source windows.
- Interrupted batches can resume safely.
- Output rows identify the exact model and preprocessing configuration used.
- Performance is sufficient for the intended collection size and hardware.

## Phase 6: Activity analytics and user-facing application

Status: Planned

### Purpose

Turn detection records into understandable activity patterns and an accessible review workflow.

### Work

- Aggregate detections by hour, date, site, and recording group.
- Create time-of-day histograms and circular clock plots.
- Separate raw detection counts from effort-normalized activity rates.
- Allow confidence-threshold filtering and show how results change with the threshold.
- Provide direct links from analytics back to source recordings and detected windows.
- Build the application in `classifier_app/` with a small, testable boundary around inference and analytics services.
- Support uploading or selecting recordings, viewing detections, listening to windows, and exporting results.
- Clearly display model version, threshold, and data coverage in the interface.

### Acceptance criteria

- Analytics distinguish recording effort from detected activity.
- Users can trace every chart value back to underlying detection records.
- The application can process a sample recording and display its results without manual command-line work.
- Interface tests cover core upload, processing, review, and export paths.

## Phase 7: Operational quality and maintenance

Status: Planned

### Purpose

Keep the system reliable as data, dependencies, and models evolve.

### Work

- Add continuous integration for tests, formatting, dependency-lock validation, and small synthetic pipeline fixtures.
- Define versioning for data manifests, preprocessing configurations, models, and result schemas.
- Add privacy and storage rules for recordings and derived artifacts.
- Back up labels, manifests, model metadata, and result databases independently of regenerable spectrograms.
- Monitor label distribution, inference failures, confidence drift, and false-positive rates on new deployments.
- Establish a documented model refresh and rollback process.
- Keep the process log, system design, and roadmap synchronized with verified behavior.

### Acceptance criteria

- A clean checkout passes automated checks without access to the private audio collection.
- Critical non-regenerable metadata has a tested backup and restore path.
- Model and schema changes are versioned and reversible.
- Operational failures and data drift are visible before they silently affect analysis.

## Recommended execution order

Complete Phase 2 before investing in a more complex model.
Run Phase 3 iteratively with Phase 2 reports so labeling targets the largest evidence gaps.
Begin Phase 4 only when grouped evaluation is valid and label diversity is documented.
Start Phases 5 and 6 only after a saved model has credible held-out performance and an explicit threshold policy.
Treat Phase 7 as a continuous quality layer, with its first continuous-integration checks added as soon as the Phase 2 test suite exists.
