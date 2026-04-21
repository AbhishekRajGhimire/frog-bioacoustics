# Frog Classifier — System Design

This document describes how the project components connect end-to-end: where data comes from, how it is transformed, how labels are created, and how we prepare a leakage-safe dataset for training.

## Goals

- Convert long field recordings into **fixed-size** examples (5s chunks) suitable for ML.
- Keep a **reproducible** mapping between:
  - a spectrogram PNG,
  - its source raw audio recording,
  - and the exact time window within that recording.
- Enable fast human labeling with **audio playback**.
- Prevent overly optimistic metrics by ensuring a **leakage-safe split** (split by recording/day, not by chunk).

## High-level pipeline

```mermaid
flowchart LR
  A[Raw audio\nData/raw or Data/good data] -->|slice + mel| B[Spectrogram PNGs\nData/spectrograms*]
  B -->|manual review + keypress| C[Labeled PNGs\nData/training_data/{litoria_aurea,background}]
  C -->|grouped split| D[manifest.csv\n(split by recording/day)]
  D --> E[Training (future)\nimage classifier baseline]
  E --> F[Inference (future)\n"frog present?" on new audio]
  B -->|on-demand export window| G[5s chunk WAVs\nData/processed]
  G --> C
```

## Components (scripts)

### 1) Spectrogram generation: `src/slice_audio.py`

**Responsibility**
- Walk a raw audio tree, slice each recording into **non-overlapping 5-second chunks**, and write a Mel-spectrogram PNG for each chunk.

**Inputs**
- `--raw-root` (default `Data/raw`)
  - Can also point at `Data/good data` (or any other nested folder of recordings).

**Outputs**
- `--out-root` (default `Data/spectrograms`)
- Output structure mirrors `--raw-root` structure:
  - `out_root/<same subfolders>/<recording_stem>_start{N}s.png`

**Key invariants**
- **Chunk length** is fixed (`--chunk-seconds`, default 5).
- **No overlap**; remainder shorter than a chunk is dropped.
- **Filename encodes the start time**:
  - `<recording_stem>_start0s.png`, `<recording_stem>_start5s.png`, ...

### 2) Manual labeling (with audio): `src/label_spectrograms.py`

**Responsibility**
- Iterate over spectrogram PNGs, show them, play their corresponding 5-second audio window, and move each PNG into the appropriate label folder.

**Inputs**
- `--spectrogram-root` (default `Data/spectrograms`)
- `--raw-root` (default `Data/raw`)
- `--processed-root` (default `Data/processed`)
- `--training-root` (default `Data/training_data`)

**Outputs**
- Moves labeled PNGs into:
  - `Data/training_data/litoria_aurea/`
  - `Data/training_data/background/`

**Audio playback / export behavior**
- Looks for the matching chunk audio under `Data/processed/` (same subfolders + same basename as the PNG).
- If missing, it will **export the exact 5s window** from the corresponding raw recording into `Data/processed/` and then play it (WAV).

**How PNG → raw audio mapping works**
- From PNG name `recording01_start30s.png`, parse:
  - `recording_id = recording01`
  - `start_s = 30`
- Then look for raw audio in the mirrored folder:
  - `raw_root/<same subfolders>/recording01.(wav|mp3)`

### 3) Leakage-safe manifest + split: `src/build_manifest.py`

**Responsibility**
- Create `manifest.csv` from labeled PNGs and assign `train/val/test` split **by group** to avoid leakage.

**Inputs**
- `--training-root` (default `Data/training_data`)

**Outputs**
- `--out-csv` (default `Data/training_data/manifest.csv`)
- CSV columns:
  - `image_path,label,recording_id,start_s,split`

**Leakage avoidance**
- Default grouping is `--group-by recording` so all chunks from the same recording stay together.
- Alternative grouping:
  - `--group-by day` if your recording IDs start with `YYYYMMDD_...`
  - `--group-by folder` to group by label folder structure (useful if you preserve site/batch folders there)

## Storage layout (canonical)

```
frog-classifier/
  Data/
    raw/                       # raw recordings (original dataset)
    good data/                 # optional: curated raw source (same idea, different root)
    spectrograms/              # outputs from slice_audio.py on Data/raw
    spectrograms_good/         # outputs from slice_audio.py on Data/good data (recommended)
    processed/                 # 5s WAV chunks (created on-demand by labeler)
    training_data/
      litoria_aurea/           # labeled positives (PNGs)
      background/             # labeled negatives (PNGs)
      manifest.csv             # leakage-safe split manifest
  src/
    slice_audio.py
    label_spectrograms.py
    build_manifest.py
  docs/
    system design.md
  PROCESS.md
```

## Recommended operating modes

### A) Generate “good data” spectrograms without mixing with older outputs

- Use a dedicated spectrogram output root (mirrors the “good data” raw tree exactly):

```bash
python src/slice_audio.py --raw-root "Data/good data" --out-root "Data/spectrograms_good" --sample-rate 48000
```

### B) Label with reliable mapping back to raw audio

```bash
python src/label_spectrograms.py --spectrogram-root "Data/spectrograms_good" --raw-root "Data/good data" --limit 12 --shuffle
```

### C) Build manifest + split by recording/day (avoid leakage)

```bash
python src/build_manifest.py --training-root "Data/training_data" --out-csv "Data/training_data/manifest.csv" --group-by recording
```

## Design decisions (why)

- **PNG spectrograms as the first-class artifact**: easy to inspect/label and works with standard image classification pipelines.
- **Path mirroring** (`raw_root` → `out_root`): ensures deterministic mapping and reduces bookkeeping.
- **Filename encodes start time**: allows reconstructing the exact audio window without extra metadata files.
- **Split by recording/day**: prevents chunk-level leakage that would inflate validation/test scores.

## Future additions (planned)

- Training script that reads `manifest.csv`, applies augmentations, and trains a baseline model.
- Inference script that runs the slicer + model over new recordings and outputs “frog present” timestamps.
- Optional: store features as `.npy` (faster training I/O) while keeping PNGs for human review.

