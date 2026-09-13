# Spectrogram Noise-Floor Rendering Design

## Summary

The preprocessing contract will move to schema version 2 and render every five-second Mel spectrogram as decibels above the noise floor of its own recording.
Pixel brightness will then mean the same thing in every image, faint distant calls will stay visible, and steady background such as insect drone will fall away.
The rendering will be implemented once in a reusable package so that slicing, verification, and later batch inference share one code path.
All existing spectrograms will be regenerated from raw audio, and the 61 labeled images will be replaced in place by name so that no human label is lost.

The choice of this rendering over three alternatives is recorded in [the rendering decision](../../decisions/2026-09-13-spectrogram-rendering.md) with the comparison sheets.

## Goals

- Make pixel intensity comparable across chunks, recordings, nights, and sites.
- Keep faint calls visible when a louder unrelated sound shares the chunk.
- Remove stationary background per Mel band without removing non-stationary sounds.
- Store images losslessly at native resolution with no figure, axis, or resampling artifacts.
- Give the rendering one tested implementation that slicing and future inference both call.
- Provide a command that proves stored images match the declared contract by re-rendering a sample from raw audio.
- Regenerate every spectrogram and every labeled image without deleting raw audio or human decisions.

## Non-goals

- This change will not alter chunking, the 400 Hz to 4,000 Hz band, Mel resolution, class names, filenames, or the manifest schema.
- This change will not train a model or add inference.
- This change will not redesign the labeling interfaces beyond displaying the new images through a colour map.
- This change will not adopt per-channel energy normalization.
- This change will not delete raw recordings, cached playback chunks, or any labeled file.
- This change will not claim a measured detection improvement, because only 13 positives exist.

## Current-state findings

The slicer converts each chunk with `power_to_db(ref=np.max)`, so every chunk is referenced to its own loudest moment.
It then draws the array into a Matplotlib figure with automatic colour limits and saves a cropped 372 by 369 pixel RGBA image.
A distant call in a chunk that also contains wind, a bird, or loud insects is compressed into a few grey levels, while the same call in a quiet chunk is clearly visible.

Rendering all 13 labeled positives and six labeled negatives under four treatments showed that subtracting each Mel band's median level over the whole recording makes the harmonic stacks stand out in ten of thirteen positives and removes the insect band from every negative.
A fixed dynamic range changed almost nothing, and per-channel energy normalization gave lower contrast plus a warm-up stripe on the left edge of every image.

Measured on the 61 labeled recordings, the median floor and the 25th percentile floor differ by 2.6 dB on average and 4.3 dB at most.
The brightest frog pixels reach 31.7 dB above the median floor, and loud non-target transients reach 46 dB.

Every preprocessing value already comes from `config/preprocessing.toml`, whose SHA-256 is carried by every manifest row.
The data-quality report notes that this checksum cannot prove how pre-existing images were generated, and the September 13 band correction showed that gap is real.

Raw recordings are 48 kHz mono 16-bit WAV files of 300 seconds, and all 618 file stems are unique.
A 300-second file produces 60 complete chunks, so the collection holds 37,080 images of which 61 are labeled.

## Architecture

```text
src/
  frog_classifier/
    data/
      config.py                 schema 2 loader with the normalization section
    preprocessing/
      __init__.py
      spectrogram.py            pure array functions: Mel dB, noise floor, chunk image, PNG bytes
      recording.py              load one recording and yield rendered chunks
scripts/
  slice_audio.py                thin command over the package, overwrites images by name
  sync_labeled_images.py        replace labeled images with their freshly generated counterparts
  verify_spectrograms.py        re-render a sample from raw audio and compare bytes
tests/
  preprocessing/
    __init__.py
    test_spectrogram.py
    test_recording.py
  test_slice_audio.py
  test_sync_labeled_images.py
  test_verify_spectrograms.py
docs/
  decisions/
    2026-09-13-spectrogram-rendering.md
    images/
```

The `data` package keeps ownership of configuration loading because the manifest, reports, and baseline already import it from there.
The new `preprocessing` package depends on `data.config` and on NumPy, librosa, and Pillow, all of which are already locked runtime dependencies.
Matplotlib leaves the slicer entirely and remains a dependency only for the labeling interfaces.

### `spectrogram.py`

This module will expose small pure functions over NumPy arrays.

- `mel_decibels(waveform, config)` returns the Mel power spectrogram in absolute decibels using the configured Mel, FFT, and hop settings with `ref=1.0` and no `top_db` clipping.
- `noise_floor(mel_decibels, percentile)` returns one value per Mel band, the requested percentile of that band across all frames.
- `chunk_image(mel_decibels, floor, config)` subtracts the floor from each band, maps `db_floor` to 0 and `db_ceiling` to 255 linearly, clips outside that range, rounds to unsigned 8-bit, and flips rows so the lowest Mel band is the bottom row.
- `encode_png(image)` returns the PNG bytes of an 8-bit grayscale image with no metadata chunks, so identical arrays always produce identical bytes.

### `recording.py`

This module will expose `render_recording(path, config)`.
It loads the whole file as mono audio at the configured sample rate, computes the Mel decibels of the entire file, and derives the noise floor from every frame of the file including any incomplete tail.
It then iterates complete non-overlapping chunks in order, computes each chunk's Mel decibels from the chunk's own samples, and yields a `RenderedChunk` with `start_s` and `png_bytes`.
Chunk Mel spectrograms are computed from the chunk samples rather than sliced from the full-file spectrogram so that a future streaming or batch inference path produces identical pixels.
A file shorter than one chunk yields nothing.

### `slice_audio.py`

The command will keep `--raw-root`, `--out-root`, `--out-subdir`, and `--limit-files`.
It will drop `--sample-rate`, `--chunk-seconds`, `--n-mels`, `--fmin`, and `--fmax`, because overriding the contract from the command line is how the stored images and the declared configuration diverged.
It will discover `.wav` and `.mp3` files recursively, preserve their relative folders, write `<stem>_start<N>s.png` for every rendered chunk, and overwrite any existing file of the same name.
It will not delete anything.
A failure in one file is reported and the run continues, as it does today.

### `sync_labeled_images.py`

The command will take `--labeled-root`, `--spectrogram-root`, and `--dry-run`.
For every PNG beneath the labeled root it will parse the example name with the existing naming contract, and then locate exactly one file with the same name beneath the spectrogram root.
All checks run before any change.
A labeled name that does not parse, has no counterpart, or has more than one counterpart fails the command with every issue listed and no file touched.
When every check passes the command moves each fresh file over its labeled counterpart with an atomic per-file replace, so the labeled image is updated and the queue no longer contains a labeled example.
It prints how many images were replaced.

### `verify_spectrograms.py`

The command will take `--raw-root`, `--spectrogram-root`, `--labeled-root`, `--sample`, `--seed`, and `--all`.
It draws a seeded random sample from the labeled tree and from the spectrogram tree, locates each example's recording by stem beneath the raw root, re-renders that recording with `render_recording`, and compares the stored bytes with the freshly rendered bytes.
It prints one line per example and exits non-zero if any example differs or cannot be located.
`--all` renders every recording once and checks every stored image, which is the full provenance check.
This command replaces the reliance on a checksum note and would have caught the band mismatch found on September 13.

## Preprocessing contract

`config/preprocessing.toml` will become:

```toml
schema_version = 2

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 400
fmax_hz = 4000
power = 2.0
n_fft = 2048
hop_length = 512

[normalization]
reference = "recording"
noise_floor_percentile = 50
db_floor = 0
db_ceiling = 30

[rendering]
format = "png"
bit_depth = 8
low_frequency_at_bottom = true

[classes]
litoria_aurea = 1
non_target = 0
```

The loader will accept only schema version 2 and will reject a version 1 file with a message that says the spectrograms must be regenerated.
The `normalization` section is new.
`reference` must be `"recording"`, `noise_floor_percentile` must be an integer from 0 to 100, and `db_floor` must be a finite number below the finite `db_ceiling`.
The `rendering` section loses `figure_width_inches`, `figure_height_inches`, `dpi`, `axis_visible`, and `interpolation`, and gains `bit_depth`, which must be 8, and `low_frequency_at_bottom`, which must be true.
Every other rule from schema version 1 remains, including exact key sets per section, the Nyquist check, and the canonical class mapping.

The median is the default floor because it is the classic robust estimate of a band's typical level and it is what the comparison sheets used.
The percentile stays configurable so that a site with dense choruses can lower it if calls occupy most of a recording and lift the floor.
The ceiling is 30 dB because the loudest labeled frog pixels reach 31.7 dB above the floor and a 25 dB ceiling would clip them.
Louder non-target transients saturate, which is acceptable.

The resulting image for the current contract is 216 pixels wide and 128 pixels tall, because a 110,250-sample chunk at hop 512 yields 216 centred frames.
Width and height are consequences of the contract and are not configured separately.

## Labeler display

Both labelers will keep reading PNG files by name and moving them between directories exactly as they do now.
The Streamlit labeler will load the grayscale image, map it through the viridis colour map, and display the result scaled to the column width.
The Matplotlib labeler will display the grayscale array with the viridis colour map.
Storage stays grayscale, and the colour map is purely a display choice.

## Manifest, reports, and baseline

The manifest schema, fold strategy, validation rules, and report layout do not change.
The `preprocessing_config_sha256` value changes because the configuration bytes change, so the manifest and reports must be rebuilt after regeneration.
The report's provenance note will point to `verify_spectrograms.py` as the way to prove stored images match the contract.
The baseline continues to load grayscale images and resize them, so it needs no change beyond the rebuilt manifest.

## Test strategy

All tests use synthetic audio written to temporary directories with soundfile and never touch private recordings.

`tests/data/test_config.py` and `tests/data/helpers.py` will move their inline configuration to schema version 2.
New cases will reject a schema version 1 file with the regeneration message, reject a percentile outside 0 to 100, reject a ceiling at or below the floor, reject a bit depth other than 8, and reject `low_frequency_at_bottom = false`.

`tests/preprocessing/test_spectrogram.py` will prove:

- a five-second chunk renders to a 128 by 216 unsigned 8-bit array;
- a tone present for the whole synthetic recording renders near black in its band after floor subtraction;
- a tone present in only one chunk renders bright in that chunk and dark in the others;
- a low-frequency tone lights the bottom rows;
- a value at the floor maps to 0, a value at the ceiling maps to 255, and louder values clip at 255;
- rendering the same array twice gives identical PNG bytes.

`tests/preprocessing/test_recording.py` will prove that a 17-second file yields chunks starting at 0, 5, and 10 seconds and drops the tail, and that a file shorter than one chunk yields nothing.

`tests/test_slice_audio.py` will prove the command writes the expected names under the preserved folder structure, exposes no preprocessing overrides, overwrites an existing image, and continues after one unreadable file.

`tests/test_sync_labeled_images.py` will prove the happy path replaces labeled bytes and removes the queue copy, that a missing counterpart, a duplicate counterpart, or an unparseable name leaves every file untouched, and that `--dry-run` changes nothing.

`tests/test_verify_spectrograms.py` will prove a consistent tree exits zero and a tampered image exits non-zero naming that file.

The repository reproducibility test will add the new scripts and package modules to its compile and documentation checks.

## Migration and compatibility

1. Implement the package, the schema 2 loader, the three commands, the labeler display change, and the tests, and run the full suite.
2. Run `scripts/slice_audio.py --limit-files 1 --out-root <scratch>` and open a few images in the Streamlit labeler to confirm they display correctly.
3. Run `scripts/slice_audio.py` against `raw/external`, which overwrites the 37,019 queued images and recreates the 61 that were moved out during labeling.
4. Run `scripts/sync_labeled_images.py`, which replaces the 61 labeled images and removes their fresh copies from the queue.
5. Run `scripts/verify_spectrograms.py --sample 50` and require every line to report a match.
6. Run `scripts/build_manifest.py` twice and confirm identical checksums, then run `scripts/train_baseline.py`.
7. Update the architecture, workflow, outline, and README text that describes images or commands, and record the new checkpoint in the roadmap.

No raw file, cached playback chunk, or labeled file is deleted at any step.
The only removals are the fresh queue copies of labeled examples, which the sync command moves rather than deletes.
Old-format images are overwritten by name, so a partially completed regeneration leaves a mixed tree that `verify_spectrograms.py --all` will expose.
The dependency lock does not change.

## Acceptance criteria

- The loader accepts the tracked schema 2 file and rejects a schema 1 file with the regeneration message.
- `verify_spectrograms.py --sample 50` reports a byte-identical match for every sampled labeled and queued image.
- The queue holds 37,019 images and the labeled tree holds 61, for 37,080 in total.
- Two consecutive manifest builds produce identical checksums, and every row carries the schema 2 configuration checksum.
- The baseline runs to completion on the rebuilt manifest.
- The full test suite, script compilation, `git diff --check`, and the em dash scan all pass.
- Both labelers display the new images legibly on manual inspection.
- The decision record and its two comparison sheets are tracked under `docs/decisions/`.

## Known limitations after this change

- The floor is estimated per recording file, so a site that records files of very different lengths will normalize over different amounts of context.
- Dense choruses that occupy most of a file lift the median floor and reduce call contrast, which the percentile setting can counter but only by configuration.
- Recording identity is still the file stem, so two sites that name files by timestamp alone could collide once pond data is added.
- The visual gain is judged on 13 positives and has not been measured as a detection improvement.
