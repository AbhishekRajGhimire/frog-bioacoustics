# Frog Classifier — Process Log

This file is a running log of **what we’ve built so far**, **how to reproduce it**, and **what decisions we’ve made**. Append to this over time as the project evolves.

## Current repository layout (relevant parts)

- `Data/raw/`: source audio dataset (nested folders; “pond” is typically the first folder under `raw/`, e.g. `1A`, `1B`).
- `Data/spectrograms/`: generated outputs (mirrors the directory structure from `Data/raw/`).
- `src/slice_audio.py`: audio traversal + slicing + Mel-spectrogram PNG generation script.

## What we implemented

### 1) Dataset traversal (nested)

- Uses `os.walk` to discover audio files under `Data/raw/`.
- Supported extensions: `.wav`, `.mp3`.
- Preserves the directory structure under `raw/` when writing outputs under `Data/spectrograms/`.

### 2) Audio loading and slicing

- Loads audio with `librosa.load(..., sr=22050, mono=True)` by default (configurable via CLI).
- Slices each file into **5-second non-overlapping chunks**.
- Drops any leftover tail shorter than 5 seconds (no padding).

### 3) Spectrogram generation

- Creates a **Mel-spectrogram** per chunk:
  - `n_mels = 128`
  - `fmin = 0` (default)
  - `fmax = 8000` (default)
- Converts power spectrogram to dB (log scale) and saves as an **axis-free PNG** (headless-safe via Matplotlib `Agg` backend).

### 4) Naming convention

Each PNG includes:

- the original audio **base name**, and
- the **chunk start time** in seconds.

Example:

- If the file is `recording01.wav`, the chunk starting at 30 seconds becomes:
  - `recording01_start30s.png`

The folder path under `Data/spectrograms/` mirrors `Data/raw/`, e.g.:

- `Data/raw/1A/.../recording01.wav`
- `Data/spectrograms/1A/.../recording01_start30s.png`

## How to run

From the repo root:

```bash
python src/slice_audio.py
```

Run on your "good data" tree (example):

```bash
python src/slice_audio.py --raw-root "Data/good data" --out-root "Data/spectrograms" --out-subdir "good_data"
```

Quick demo (process only the first 2 files per top-level folder):

```bash
python src/slice_audio.py --raw-root "Data/good data" --out-root "Data/spectrograms" --out-subdir "good_data" --limit-files 2
```

Note: AudioMoth recordings in `Data/good data/` are often 48 kHz (see `CONFIG.TXT`), but we **resample to 22050 Hz by default** for consistency. You can override with `--sample-rate`.

What you should see:

- A progress bar per pond folder (top-level directory under `Data/raw/`), e.g. “Processing pond 1A”.
- A final summary: how many PNGs were written and where.

## Dependencies

The script expects these Python packages to be installed:

- `librosa`
- `numpy`
- `matplotlib`
- `tqdm`

## Manual labeling (spectrogram review → training folders)

We added an interactive labeling script to quickly sort spectrogram PNGs into two classes:

- **Frog present**: `Data/training_data/litoria_aurea/`
- **Background/other**: `Data/training_data/background/`

### Script

- `src/label_spectrograms.py`

What it does:

- Reads PNGs from `Data/spectrograms/` (recursive).
- Displays each image (loaded via PIL) and waits for a keypress:
  - `y` → move image to `litoria_aurea`
  - `n` → move image to `background`
  - `s` → skip
  - `q` / `esc` → quit
- Prints progress like: `Image 12 of 500`.
- Tries to locate and play the matching **5-second audio chunk** from `Data/processed/` using the PNG filename (expects `*_start{N}s`).

### Demo run 

Run on only a small number of images first:

```bash
python src/label_spectrograms.py --limit 12 --shuffle
```

Demo run on the **good data** spectrograms:

```bash
python src/label_spectrograms.py --spectrogram-root "Data/spectrograms/good_data" --raw-root "Data/good data" --processed-root "Data/processed/good_data" --limit 12 --shuffle
```

### Full run later

When ready to label everything:

```bash
python src/label_spectrograms.py --limit 0 --shuffle
```

### Audio playback note

Audio playback currently supports **WAV** chunks only (uses Windows `winsound` by default; can use `simpleaudio` if installed).

Expected chunk location pattern:

- `Data/spectrograms/<subfolders>/<name>_start30s.png`
- `Data/processed/<subfolders>/<name>_start30s.wav`

If the matching processed chunk does not exist, the script will (by default) export that 5-second window from `Data/raw/` into `Data/processed/` on-demand and play it.

## Labeling UI (web frontend — recommended)

If the Matplotlib window closes unexpectedly or terminal key capture is flaky, use the Streamlit frontend.

### Script

- `src/label_frontend.py`

### Install + run

```bash
pip install streamlit
streamlit run src/label_frontend.py
```

### One-click launcher (Windows, non-technical)

Double-click:
- `Launch_Labeler.bat`

First run will:
- create a local `.venv/`
- install dependencies from `requirements_labeler.txt`
- open the UI in your browser

In the sidebar, confirm these defaults (or adjust):
- `spectrogram_root`: `Data/spectrograms/good_data`
- `raw_root`: `Data/good data`
- `processed_root`: `Data/processed/good_data`
- `training_root`: `Data/training_data`

This UI provides:
- persistent image viewer
- reliable **Frog / Background / Skip** buttons
- **Replay audio** button

## Notes / decisions (so far)

- **Chunking**: fixed 5-second chunks; remainder is dropped (no padding) to keep consistent shapes.
- **Outputs**: PNGs are stored without axes/ticks to make them easier to use as ML inputs.
- **Robustness**: per-file errors are logged and the run continues with remaining files.

## Leakage-safe splitting (manifest.csv)

Chunks from the same 5-minute recording are highly correlated. To avoid train/val/test leakage, we split by **recording** (or **day**), not by chunk.

### Script

- `src/build_manifest.py`

### Example

```bash
python src/build_manifest.py --training-root "Data/training_data" --out-csv "Data/training_data/manifest.csv"
```

## Baseline training (sanity check)

### Script

- `src/train_baseline.py`

### Example

```bash
python src/train_baseline.py --manifest "Data/training_data/manifest.csv"
```

## Next steps (suggested)

- Add a `requirements.txt` (or `pyproject.toml`) to pin versions for reproducibility.
- Consider adding optional:
  - padding for last chunk vs dropping remainder
  - amplitude normalization
  - resampling quality controls
  - parallel processing (CPU-bound parts) if runtime becomes large
- Decide training input format:
  - use PNG images directly, or
  - save spectrogram tensors (`.npy`) for faster training I/O.

