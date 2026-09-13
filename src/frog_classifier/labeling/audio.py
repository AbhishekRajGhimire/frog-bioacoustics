from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from frog_classifier.data.config import PreprocessingConfig
from frog_classifier.data.naming import ExampleNameError, parse_example_filename


SUPPORTED_EXTS = {".wav", ".mp3"}


def find_recording(
    raw_root: Path,
    recording_id: str,
    *,
    preferred_folder: Path | None = None,
) -> Path | None:
    """The recording with this stem: the preferred folder first, then a
    recursive search that must find exactly one file."""
    if preferred_folder is not None:
        folder = raw_root / preferred_folder
        if folder.is_dir():
            direct = sorted(path for path in folder.iterdir() if _is_recording(path, recording_id))
            if len(direct) == 1:
                return direct[0]
    matches = sorted(path for path in raw_root.rglob("*") if _is_recording(path, recording_id))
    return matches[0] if len(matches) == 1 else None


def find_or_export_chunk(
    image_path: Path,
    *,
    raw_root: Path,
    chunk_root: Path,
    config: PreprocessingConfig,
    spectrogram_root: Path | None = None,
) -> Path | None:
    """The playback WAV for a clip, exported from the recording on first use.

    The chunk lives under chunk_root in the recording's folder relative to
    raw_root, so a clip found in the queue and the same clip after labeling
    share one cached file.
    """
    try:
        key = parse_example_filename(image_path, chunk_seconds=config.audio.chunk_seconds)
    except ExampleNameError:
        return None
    preferred = None
    if spectrogram_root is not None:
        try:
            preferred = image_path.resolve().relative_to(spectrogram_root.resolve()).parent
        except ValueError:
            preferred = None
    recording = find_recording(raw_root, key.recording_id, preferred_folder=preferred)
    if recording is None:
        return None
    relative_folder = recording.resolve().parent.relative_to(raw_root.resolve())
    chunk_path = chunk_root / relative_folder / f"{key.example_id}.wav"
    if chunk_path.is_file():
        return chunk_path

    import librosa  # imported lazily so the labeler starts quickly

    waveform, sample_rate = librosa.load(
        recording,
        sr=config.audio.sample_rate_hz,
        mono=True,
        offset=float(key.start_s),
        duration=float(config.audio.chunk_seconds),
    )
    if len(waveform) == 0:
        return None
    write_wav_mono_16bit(chunk_path, waveform, int(sample_rate))
    return chunk_path


def write_wav_mono_16bit(out_path: Path, waveform: np.ndarray, sample_rate_hz: int) -> None:
    """Mono 16-bit PCM WAV through the standard library."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())


def _is_recording(path: Path, recording_id: str) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS and path.stem == recording_id
