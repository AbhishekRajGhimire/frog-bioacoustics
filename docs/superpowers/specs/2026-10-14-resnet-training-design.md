# ResNet Training Design

## Summary

Phase 4 trains the first real *Litoria aurea* detector from the validated manifest.
The design fine-tunes an ImageNet-pretrained ResNet-18 on the stored spectrogram images, selects a detection threshold on the validation fold, reports held-out performance on the test fold, and measures variation by rotating the five grouped folds.
Every run is saved as one self-contained folder under `models/` holding the weights, the exact configurations, a copy of the manifest, per-example predictions, metrics, a report, and checksums.
A separate command reloads a saved run in a fresh process and proves it reproduces its own predictions.

The labeler, the manifest builder, the preprocessing contract, and the stored spectrograms do not change.

## Context

On October 14, 2026 the validated manifest holds 4,224 examples from 570 recording groups.
There are 1,204 frog clips from 176 recordings and 3,020 background clips from 502 recordings.
Every fold holds about 240 frog clips from at least 32 recordings, so grouped evaluation is stable enough to train and measure a model.
All recordings come from one site between January and March 2022, recorded between midnight and noon.
Only 47 frog clips carry the faint tag.

The machine has an NVIDIA RTX 4080 laptop GPU with 12 GB of memory.
PyTorch is not installed yet.
uv is installed at `%USERPROFILE%\.local\bin\uv.exe`, so the lock can be regenerated properly.

## Goals

- Train a ResNet-18 detector whose held-out performance on unseen recordings is measured honestly.
- Choose the detection threshold with validation data by a stated rule.
- State expected false-positive behavior explicitly, including by hour of recording.
- Show how much performance varies across recording groups.
- Compare against the logistic-regression baseline on identical data.
- Save everything needed to reproduce the evaluation predictions, and prove it with a command.
- Produce an out-of-fold probability for every labeled clip, so confident disagreements with a human label can be re-checked later.

## Non-goals

- No recording-level batch inference; that is Phase 5.
- No analytics or application work.
- No labeler changes, including a mode for reviewing model disagreements; that is a follow-up.
- No hyperparameter search, alternative architectures, mixed precision, or model export formats.
- No Python upgrade and no change to the existing runtime dependencies.

## Decisions

### Python stays at 3.13

Python 3.15.0 is released and PyTorch publishes wheels for it, but the heavy work runs in compiled PyTorch code, so a newer interpreter brings no training benefit.
Upgrading would force newer numpy and librosa releases, which can change spectrogram bytes, and that would make any shift in Phase 4 results hard to attribute.
Python 3.13 receives security fixes until October 2029.
An upgrade is a separate change with its own full verification, including `scripts/verify_spectrograms.py --all`.

### PyTorch is a default dependency group, not a runtime dependency

`pyproject.toml` gains a `model` dependency group with `torch>=2.14,<2.15` and `torchvision>=0.29,<0.30`.
`[tool.uv] default-groups = ["model"]` makes `uv sync --frozen` install it.
On Windows and Linux both packages come from the PyTorch CUDA 13.0 index, declared as an explicit uv index so no other package can resolve from it; other platforms use PyPI.
The labeler export is regenerated with `--no-default-groups`, so `requirements_labeler.txt` and the labeler launcher keep their current contents and never download PyTorch.
The group is named `model` rather than `train` because Phase 5 inference needs the same packages.

### Model

The model is torchvision's ResNet-18 with the `IMAGENET1K_V1` weights and its final layer replaced by one output, the frog logit.
It takes the stored image at its native size of 128 by 216 pixels; no resizing distorts the time or frequency axes.
The one grayscale channel is scaled to 0 to 1, repeated into three channels, and normalized with the ImageNet mean and standard deviation inside the model, so every caller feeds the same input.
The loss is binary cross-entropy on the logit, without class weighting; the class ratio is about one to two and a half, and the threshold rule absorbs it.

### Training

The training split is folds 2 to 4, validation is fold 1, and test is fold 0, exactly as the manifest's `split` column says.
AdamW with learning rate 0.0003 and weight decay 0.01 trains all layers with batch size 64 under cosine annealing over at most 30 epochs.
After each epoch the model predicts the validation fold without augmentation, and the epoch with the highest validation average precision is kept.
Training stops after 8 epochs without improvement.
All images are loaded into memory once; the full set is about 117 MB.
The seed fixes the shuffle order, augmentation draws, and weight initialization of the new layer, and cuDNN runs in deterministic mode with TF32 disabled.
GPU training is not guaranteed to be bit-identical between runs, but a saved model's predictions are reproducible, and that is what the verify command proves.
Training uses the GPU when one is available and the CPU otherwise, and records which.

### Augmentation

Augmentation applies to training batches only, and each draw comes from a seeded generator.

- **Time shift.** Every image is rolled circularly along time by a random offset, because a call can start anywhere in a window.
- **Attenuation.** Every image is made quieter by a random 0 to 6 dB, applied to the energy above the noise floor in the power domain, so the floor stays where it is and the clip sounds as if its sources were farther away. This targets distant calls, which are the reason for the noise-floor rendering.
- **Time and frequency masks.** One block of up to 20 time frames and one block of up to 12 Mel bins are set to the noise floor.

Time reversal and frequency flips are excluded because they change what a call looks like.

### Threshold rule

The threshold is the probability that maximizes F1 on the validation fold, with ties going to the higher threshold.
It is chosen once per trained model and applied unchanged to the test fold.
The report also states the expected number of false detections per hour of recording without a frog call, which is the test false-positive rate times the 720 windows in an hour.

### Evaluation

For the validation and test folds the run reports the number of examples, average precision, ROC AUC, and, at the threshold, precision, recall, F1, false-positive rate, and the confusion matrix.
The test fold is also broken down by recording hour into background clips, false positives, frog clips, and missed calls, because hour-dependent errors would distort time-of-day activity results.
Recall is reported separately for faint-tagged frog clips, read from the decision log at training time and stored with the predictions so the report can be reproduced.

### Baseline comparison

The logistic-regression baseline is fitted on the same training split with the same features as `scripts/train_baseline.py`: 64 by 64 grayscale images, balanced class weights, and regularization strength 1.0.
Its threshold is chosen by the same rule on the same validation fold, and its test metrics sit beside the ResNet's in the report.
The featurization moves into the package so the script and the comparison share one implementation.

### Cross-validation

By default the run also trains four more models, rotating the folds so that each fold is the test fold once and the next fold is its validation fold.
Rotation zero is the saved model itself, so the five rotations cover every fold.
Each rotation chooses its own threshold on its own validation fold, and the report gives the mean, standard deviation, minimum, and maximum of test average precision, precision, recall, and F1 for both the ResNet and the baseline.
The test-fold predictions of the five rotations form an out-of-fold probability for every labeled clip, each from a model that never trained on that clip's recording.
Only the rotation-zero weights are saved.
`--no-cross-validation` skips the four extra rotations for quick iterations.

### Saved run

Each run is written to a new folder, `models/resnet18-<UTC timestamp>/`, first under a temporary name and renamed when complete, so an interrupted run never looks finished.
The command refuses to reuse an existing folder.

| File | Contents |
| --- | --- |
| `model.pt` | Rotation-zero weights as a PyTorch state dictionary |
| `run.json` | Metadata, metrics, threshold, provenance, and checksums |
| `preprocessing.toml` | Exact copy of the preprocessing configuration |
| `training.toml` | Exact copy of the training configuration |
| `manifest.csv` | Exact copy of the manifest used |
| `predictions.csv` | One row per manifest example with its image checksum, faint tag, ResNet probability, and baseline probability |
| `history.csv` | Training and validation loss, validation average precision, learning rate, and time per epoch |
| `cross_validation.csv` | Test metrics per rotation for both models |
| `out_of_fold_predictions.csv` | Out-of-fold probability for every manifest example |
| `precision_recall.png` | Validation and test precision-recall curves with the threshold marked |
| `report.md` | Human-readable summary |

`run.json` records the artifact schema version, creation time, architecture, pretrained weights, input shape, class mapping, threshold and rule, seed, device and GPU name, Python, PyTorch, torchvision, numpy, and CUDA versions, the git commit and whether the worktree had changes, SHA-256 checksums of the preprocessing configuration, training configuration, manifest, and `uv.lock`, all metrics, and the SHA-256 of every other file in the folder.
The two cross-validation files are absent when cross-validation is skipped.

### Verification

`scripts/verify_model.py <run folder>` checks every file checksum listed in `run.json`, loads the weights without downloading anything, re-reads every image listed in `predictions.csv`, and compares each recomputed probability with the saved one.
A probability that differs by more than 0.0001 fails, a missing image fails, and an image whose checksum changed is reported as changed rather than as a prediction mismatch.
`--device cpu` proves the predictions reproduce without the GPU.
The command exits non-zero on any failure.

### Training configuration

`config/training.toml` is the machine-readable training contract, loaded and validated like the preprocessing configuration: unknown or missing keys and out-of-range values are rejected, and its SHA-256 is recorded.

```toml
schema_version = 1
seed = 1337

[model]
pretrained = true

[optimization]
batch_size = 64
max_epochs = 30
learning_rate = 0.0003
weight_decay = 0.01
early_stopping_patience = 8

[augmentation]
time_mask_max_frames = 20
frequency_mask_max_bins = 12
max_attenuation_db = 6.0

[baseline]
image_size = 64
inverse_regularization = 1.0
```

## Architecture

```text
config/training.toml               training contract
src/frog_classifier/training/
  __init__.py                      package docstring only; modules are imported directly
  config.py                        load and validate the training configuration
  images.py                        load stored spectrograms into one array and checksum them
  augment.py                       seeded training-batch augmentation
  model.py                         ResNet-18 wrapper for one-channel spectrograms
  fit.py                           training loop, early stopping, and batch prediction
  metrics.py                       threshold rule, split metrics, and hour breakdown
  baseline.py                      logistic-regression features and fitting
  artifact.py                      run folder writing, checksums, and reading
  report.py                        Markdown report and precision-recall figure
  run.py                           one complete training run from manifest to saved folder
  verify.py                        reproduce a saved run's predictions
scripts/
  train_model.py                   command wrapper for run.py
  verify_model.py                  command wrapper for verify.py
  train_baseline.py                uses the shared baseline featurization
tests/training/                    synthetic tests for every module and both commands
```

The package `__init__` re-exports nothing, so importing the baseline module never loads PyTorch.
The training package reads the data package for the manifest and configuration, and the labeling package only to read faint tags and recording hours; nothing in the existing packages imports the training package.
Image shape is checked against the preprocessing contract: the height is the number of Mel bins and the width is one frame more than the samples per window divided by the hop length, which is librosa's centred framing and gives 216 under the current contract.

## Error handling

- An invalid manifest or configuration stops the run before any image is loaded, through the existing validators.
- An image with the wrong size or mode stops the run and names the file.
- An existing output folder stops the run before training.
- A failure during training leaves only the temporary folder, which the next run ignores and the operator can delete.
- The verify command reports every failing row rather than stopping at the first.

## Test strategy

All tests use synthetic images in temporary directories, run on the CPU, and never download weights, because the test configuration sets `pretrained = false`.

- Configuration: a valid file loads with its checksum; unknown keys, missing keys, wrong types, out-of-range values, and another schema version are rejected.
- Images: PNGs load into one array in order; a wrong size or a colour image is rejected with its path.
- Augmentation: shape and range are preserved; the same seed gives the same batch; attenuation never brightens a pixel and leaves the floor at zero; masks zero whole columns and rows; a time shift permutes columns.
- Model: the output is one logit per image; saving and reloading the state dictionary reproduces outputs exactly.
- Metrics: the threshold rule picks the F1 maximum and breaks ties upward; counts, precision, recall, and the hour table match hand-computed values.
- Baseline: features have the expected shape and range.
- Run: a tiny synthetic manifest trains for two epochs and writes every listed file; checksums in `run.json` match; out-of-fold predictions cover every example exactly once; an existing output folder is refused.
- Verify: a fresh run verifies; a changed image is reported as changed; a tampered file fails its checksum.
- Commands: both scripts wire their arguments to the package, and the baseline script's existing tests keep every assertion, now patching the classifier class directly because the package constructs it.
- Repository: the training modules and configuration are required files, the workflow documents both new commands, the labeler export contains no PyTorch, and the model group pins PyTorch and torchvision.

## Acceptance criteria

- `uv lock --check` passes, the labeler export differs only in its generating command, and `uv sync --frozen` installs PyTorch with CUDA available.
- The full test suite, script compilation, `git diff --check`, and the em dash scan pass.
- A real training run on the October 14 manifest completes, writes a run folder, and reports test metrics, the threshold, false positives by hour, faint recall, the baseline comparison, and cross-validation spread.
- `scripts/verify_model.py` passes on that run on the GPU and with `--device cpu`.
- The README, outline, architecture, workflow, and roadmap describe the training commands and the real results.

## Known limitations after this change

- One site, three months, and recordings between midnight and noon only; performance at other sites is unknown until their recordings are labeled.
- Few faint-tagged frog clips, so faint recall is measured on a small sample.
- Test precision reflects the class mix of the labeled set, which follows the labeling queue rather than a census of every window.
- GPU training runs are close but not bit-identical; saved predictions are exactly reproducible within the stated tolerance.
