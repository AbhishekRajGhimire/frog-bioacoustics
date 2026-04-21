from __future__ import annotations

"""
Manual spectrogram labeling tool (demo-friendly).

Workflow:
  - Source images:      Data/spectrograms/**/*.png
  - Target folders:     Data/training_data/litoria_aurea
                        Data/training_data/background
  - For each image:
      - display spectrogram
      - play the matching 5s audio chunk
        - first try Data/processed/ (if you already have chunked audio files)
        - otherwise auto-export the chunk from Data/raw/ into Data/processed/ and play it
      - wait for keypress:
          y = label as Litoria aurea (move image)
          n = label as background (move image)
          s = skip
          q/esc = quit

Filename convention expected (from our slicer):
  <original_stem>_start{N}s.png

Audio lookup convention expected:
  Data/processed/<same subfolders>/<original_stem>_start{N}s.(wav|mp3)
"""

import argparse
import os
import random
import re
import shutil
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

# Matplotlib is used to display the image.
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


SUPPORTED_AUDIO_EXTS = (".wav", ".mp3")
PNG_EXT = ".png"


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    spectrogram_root: Path
    raw_root: Path
    processed_root: Path
    training_root: Path
    frog_dir: Path
    bg_dir: Path


def _default_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    spectrogram_root = repo_root / "Data" / "spectrograms"
    raw_root = repo_root / "Data" / "raw"
    processed_root = repo_root / "Data" / "processed"
    training_root = repo_root / "Data" / "training_data"
    frog_dir = training_root / "litoria_aurea"
    bg_dir = training_root / "background"
    return Paths(
        repo_root=repo_root,
        spectrogram_root=spectrogram_root,
        raw_root=raw_root,
        processed_root=processed_root,
        training_root=training_root,
        frog_dir=frog_dir,
        bg_dir=bg_dir,
    )


def iter_pngs_os_walk(root: Path) -> Iterable[Path]:
    """Discover PNGs using os.walk (keeps behavior consistent with our dataset traversal style)."""
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for name in filenames:
            if name.lower().endswith(PNG_EXT):
                yield d / name


_START_RE = re.compile(r"^(?P<stem>.+)_start(?P<sec>\d+)s$")


def parse_chunk_from_png_name(png_path: Path) -> tuple[str, int] | None:
    """
    Extract (<original_stem>, start_seconds) from a filename like:
      recording01_start30s.png
    """
    m = _START_RE.match(png_path.stem)
    if not m:
        return None
    return m.group("stem"), int(m.group("sec"))


def find_processed_chunk_audio(paths: Paths, png_path: Path) -> Optional[Path]:
    """
    Find the corresponding 5s audio chunk in Data/processed/ based on:
      - relative folder path under Data/spectrograms/
      - png filename stem (expects *_start{N}s)
    """
    try:
        rel = png_path.relative_to(paths.spectrogram_root)
    except ValueError:
        # If a user passes a PNG outside the root, fall back to name-only lookup in processed root.
        rel = Path(png_path.name)

    base = rel.with_suffix("")  # keep same subfolders + same basename, but no suffix yet
    parent = base.parent
    name = base.name

    for ext in SUPPORTED_AUDIO_EXTS:
        candidate = paths.processed_root / parent / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_raw_audio_for_png(paths: Paths, png_path: Path) -> Optional[Path]:
    """
    Locate the original raw audio file that produced this spectrogram.

    We mirror the spectrogram folder structure, and match by filename stem.
    Example:
      spectrograms/.../recording01_start30s.png
      raw/.../recording01.(wav|mp3)
    """
    parsed = parse_chunk_from_png_name(png_path)
    if parsed is None:
        return None
    orig_stem, _start_s = parsed

    try:
        rel = png_path.relative_to(paths.spectrogram_root)
    except ValueError:
        rel = Path(png_path.name)

    raw_dir = paths.raw_root / rel.parent
    if not raw_dir.exists():
        return None

    # Prefer exact-match filenames first (fast path).
    for ext in SUPPORTED_AUDIO_EXTS:
        candidate = raw_dir / f"{orig_stem}{ext}"
        if candidate.exists():
            return candidate

    # Fall back to case-insensitive stem match (handles odd casing or extra dots).
    orig_lower = orig_stem.lower()
    for child in raw_dir.iterdir():
        if child.is_file() and child.suffix.lower() in SUPPORTED_AUDIO_EXTS and child.stem.lower() == orig_lower:
            return child
    return None


def write_wav_mono_16bit(out_path: Path, y, sr: int) -> None:
    """
    Write a mono 16-bit PCM WAV using only the Python stdlib.

    Accepts a 1D float array-like in [-1, 1] (e.g., numpy array from librosa).
    """
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_np = np.asarray(y, dtype=np.float32).reshape(-1)
    y_np = np.clip(y_np, -1.0, 1.0)
    pcm16 = (y_np * 32767.0).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


def export_processed_chunk_from_raw(
    paths: Paths,
    png_path: Path,
    *,
    chunk_seconds: int = 5,
    sr: int = 22050,
) -> Optional[Path]:
    """
    Create the processed 5-second audio chunk from the corresponding raw audio file.

    Output path: Data/processed/<same subfolders>/<png_stem>.wav
    """
    parsed = parse_chunk_from_png_name(png_path)
    if parsed is None:
        return None
    _orig_stem, start_s = parsed

    raw_audio = find_raw_audio_for_png(paths, png_path)
    if raw_audio is None:
        return None

    try:
        rel = png_path.relative_to(paths.spectrogram_root)
    except ValueError:
        rel = Path(png_path.name)

    out_wav = (paths.processed_root / rel).with_suffix(".wav")
    if out_wav.exists():
        return out_wav

    # Import lazily so startup stays fast.
    import librosa

    # Performance note:
    # Loading the entire raw file can be slow (these recordings can be long).
    # Load only the 5-second window we need.
    #
    # `offset`/`duration` are in seconds.
    y_chunk, _sr = librosa.load(raw_audio, sr=sr, mono=True, offset=float(start_s), duration=float(chunk_seconds))
    if y_chunk is None or len(y_chunk) == 0:
        return None

    write_wav_mono_16bit(out_wav, y_chunk, sr)
    return out_wav


@dataclass
class UiState:
    """Shared UI state for capturing keypresses from the image window."""

    key: Optional[str] = None
    # Keep strong references to buttons; otherwise matplotlib widgets may be GC'd.
    buttons: list[Button] = field(default_factory=list)


class AudioPlayer:
    """
    Lightweight audio player abstraction.

    Preference order:
      1) simpleaudio (if installed) - WAV only
      2) winsound (Windows builtin) - WAV only
      3) no playback (prints warning)
    """

    def __init__(self) -> None:
        self._backend = "none"
        self._play_obj = None

        try:
            import simpleaudio  # type: ignore

            self._simpleaudio = simpleaudio
            self._backend = "simpleaudio"
        except Exception:
            self._simpleaudio = None

        if self._backend == "none" and sys.platform.startswith("win"):
            try:
                import winsound

                self._winsound = winsound
                self._backend = "winsound"
            except Exception:
                self._winsound = None
        else:
            self._winsound = None

    def stop(self) -> None:
        if self._backend == "simpleaudio" and self._play_obj is not None:
            try:
                self._play_obj.stop()
            except Exception:
                pass
            self._play_obj = None

        if self._backend == "winsound" and self._winsound is not None:
            # Purge any async playback.
            try:
                self._winsound.PlaySound(None, self._winsound.SND_PURGE)
            except Exception:
                pass

    def play(self, audio_path: Path) -> None:
        self.stop()

        if not audio_path.exists():
            return

        # Current backends support WAV only.
        if audio_path.suffix.lower() != ".wav":
            print(f"[WARN] Audio playback only supports .wav right now. Skipping: {audio_path}")
            return

        if self._backend == "simpleaudio" and self._simpleaudio is not None:
            try:
                wave_obj = self._simpleaudio.WaveObject.from_wave_file(str(audio_path))
                self._play_obj = wave_obj.play()
            except Exception as e:
                print(f"[WARN] Failed to play via simpleaudio: {audio_path} ({e})")
            return

        if self._backend == "winsound" and self._winsound is not None:
            try:
                # Async so you can label while listening.
                self._winsound.PlaySound(str(audio_path), self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
            except Exception as e:
                print(f"[WARN] Failed to play via winsound: {audio_path} ({e})")
            return

        print(
            "[WARN] No audio backend available. Install `simpleaudio` for WAV playback "
            "or implement another backend. Proceeding without audio."
        )


def _init_image_window(img_path: Path, title: str, *, state: UiState):
    """
    Display an image in a non-blocking window.

    Returns (fig, ax, im). Caller is responsible for closing the figure.
    """
    # IMPORTANT (Windows): load the image into memory and close the file handle.
    # Otherwise the PNG can stay locked and `shutil.move()` will fail, terminating the run.
    import numpy as np

    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("RGB"))

    fig, ax = plt.subplots()
    # Leave room for buttons at the bottom.
    fig.subplots_adjust(bottom=0.18)
    im = ax.imshow(img_arr)
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    # Capture keypresses in the window (so user doesn't need terminal focus).
    def on_key(event) -> None:
        k = (event.key or "").lower()
        if k:
            state.key = k

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Add clickable buttons (more reliable than keyboard in some terminals).
    # These set state.key so the same wait loop handles both.
    # Button positions are [left, bottom, width, height] in figure coordinates.
    # Keep them non-overlapping (matplotlib widgets can behave oddly otherwise).
    btn_specs = [
        ("Frog (y)", "y", [0.04, 0.04, 0.18, 0.08]),
        ("Background (n)", "n", [0.24, 0.04, 0.28, 0.08]),
        ("Replay (r)", "r", [0.54, 0.04, 0.14, 0.08]),
        ("Skip (s)", "s", [0.70, 0.04, 0.12, 0.08]),
        ("Quit (q)", "q", [0.84, 0.04, 0.12, 0.08]),
    ]
    state.buttons.clear()
    for label, key, rect in btn_specs:
        bax = fig.add_axes(rect)
        b = Button(bax, label)

        def _make_cb(k: str):
            def _cb(_event) -> None:
                state.key = k

            return _cb

        b.on_clicked(_make_cb(key))
        state.buttons.append(b)

    plt.show(block=False)
    # Give the GUI a moment to render.
    plt.pause(0.05)
    return fig, ax, im


def _update_image_window(fig, ax, im, img_path: Path, title: str) -> None:
    """Update the existing figure to show a new image without closing the window."""
    import numpy as np

    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("RGB"))
    im.set_data(img_arr)
    ax.set_title(title, fontsize=10)
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
    plt.pause(0.01)


def _wait_for_label_key(*, fig=None, state: UiState) -> str:
    """
    Robust key capture.

    - Accepts keys from either:
        - the image window (matplotlib key_press_event), OR
        - the terminal (Windows msvcrt), OR
        - input() fallback on non-Windows.
    - On non-Windows: falls back to input().

    Returns: 'y', 'n', 's', 'q', 'r', 'escape' (mapped to quit), or '__reopen__'.
    """
    allowed = {"y", "n", "s", "q", "r"}

    # If the user closes the figure window, treat it as quit.
    def fig_closed() -> bool:
        if fig is None:
            return False
        try:
            return not plt.fignum_exists(fig.number)
        except Exception:
            return False

    if sys.platform.startswith("win"):
        try:
            import msvcrt  # type: ignore

            print(
                "Label using buttons or keys. (If using terminal: press y/n/s/q/r without Enter.)",
                flush=True,
            )
            while True:
                if fig_closed():
                    # Don't quit the whole run just because the window closed.
                    # We'll let the caller re-open and continue on the same image.
                    return "__reopen__"
                # Prefer window keypresses if present.
                if state.key is not None:
                    k = (state.key or "").lower()
                    state.key = None
                    if k in {"escape", "esc"}:
                        return "escape"
                    if k == "\x1b":
                        return "escape"
                    if k in allowed:
                        return k
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if not ch:
                        continue
                    ch = ch.lower()
                    if ch == "\x1b":  # ESC
                        return "escape"
                    if ch in allowed:
                        return ch
                # Keep the window responsive while waiting.
                plt.pause(0.05)
        except KeyboardInterrupt:
            return "q"
    else:
        try:
            s = input("Label [y]=frog [n]=bg [r]=replay [s]=skip [q]=quit: ").strip().lower()
            if not s:
                return "s"
            if s in allowed:
                return s
            if s in {"esc", "escape"}:
                return "escape"
            return "s"
        except KeyboardInterrupt:
            return "q"


def move_to_label(paths: Paths, img_path: Path, label: str) -> Path:
    """
    Move the image into the appropriate training_data folder.
    We keep only the filename (flatten) to avoid accidental deep nesting.
    """
    dest_dir = paths.frog_dir if label == "litoria_aurea" else paths.bg_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / img_path.name
    # Prevent accidental overwrite.
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = dest_dir / f"{stem}__dup{random.randint(1000, 9999)}{suffix}"

    shutil.move(str(img_path), str(dest))
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description="Manual labeling tool for frog spectrogram images.")
    defaults = _default_paths()
    p.add_argument("--spectrogram-root", type=Path, default=defaults.spectrogram_root)
    p.add_argument("--raw-root", type=Path, default=defaults.raw_root)
    p.add_argument("--processed-root", type=Path, default=defaults.processed_root)
    p.add_argument("--training-root", type=Path, default=defaults.training_root)
    p.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Demo mode: only label first N images. Use 0 to label all.",
    )
    p.add_argument("--shuffle", action="store_true", help="Shuffle image order.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for --shuffle.")
    p.add_argument(
        "--no-auto-export-missing-audio",
        action="store_true",
        help="Disable exporting missing chunks from Data/raw into Data/processed (labeling will run without audio).",
    )
    args = p.parse_args()

    paths = Paths(
        repo_root=defaults.repo_root,
        spectrogram_root=args.spectrogram_root,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        training_root=args.training_root,
        frog_dir=args.training_root / "litoria_aurea",
        bg_dir=args.training_root / "background",
    )

    if not paths.spectrogram_root.exists():
        raise FileNotFoundError(f"spectrogram root not found: {paths.spectrogram_root}")

    paths.frog_dir.mkdir(parents=True, exist_ok=True)
    paths.bg_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(iter_pngs_os_walk(paths.spectrogram_root))
    if not images:
        print(f"No PNGs found under {paths.spectrogram_root}")
        return 0

    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(images)

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    total = len(images)
    player = AudioPlayer()

    print("Controls: y = Litoria aurea, n = background, r = replay, s = skip, q/esc = quit")

    # Keep a single persistent window open and update its contents each step.
    fig = None
    ax = None
    im = None
    state = UiState()

    for idx, img_path in enumerate(images, start=1):
        print(f"\nImage {idx} of {total}: {img_path}", flush=True)

        try:
            # Attempt to find and play the corresponding audio chunk from processed_root.
            audio_path = find_processed_chunk_audio(paths, img_path)
            if audio_path is None and not args.no_auto_export_missing_audio:
                print(f"[INFO] Chunk missing; exporting into {paths.processed_root} ...", flush=True)
                audio_path = export_processed_chunk_from_raw(paths, img_path)

            if audio_path is None:
                print(
                    f"[WARN] No playable audio for: {img_path.name}\n"
                    f"       processed_root: {paths.processed_root}\n"
                    f"       raw_root:       {paths.raw_root}",
                    flush=True,
                )
            else:
                print(f"Playing: {audio_path}", flush=True)
                player.play(audio_path)

            try:
                rel = img_path.relative_to(paths.spectrogram_root)
            except ValueError:
                rel = Path(img_path.name)

            title = f"{rel}  |  (y) frog  (n) background  (s) skip  (q) quit"
            title = f"{rel}  |  (y) frog  (n) background  (r) replay  (s) skip  (q) quit"

            # Recreate the window if the user closed it.
            if fig is None or (hasattr(fig, "number") and not plt.fignum_exists(fig.number)):
                fig, ax, im = _init_image_window(img_path, title=title, state=state)
            else:
                _update_image_window(fig, ax, im, img_path, title=title)

            # Stay on the same image until the user chooses y/n/s/q (replay is allowed).
            while True:
                key = _wait_for_label_key(fig=fig, state=state)
                if key == "__reopen__":
                    fig, ax, im = _init_image_window(img_path, title=title, state=state)
                    continue
                if key == "r":
                    if audio_path is not None:
                        print(f"Replaying: {audio_path}", flush=True)
                        player.play(audio_path)
                    else:
                        print("[WARN] No audio available to replay for this image.", flush=True)
                    continue
                break
        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}", flush=True)
            key = "s"
        finally:
            # Stop audio + close image window regardless of what happened.
            try:
                player.stop()
            except Exception:
                pass
            # Keep the figure open across iterations (only close on quit / end).

        if key in {"q", "escape"}:
            print("Quitting.")
            break
        if key == "s":
            continue
        if key == "y":
            try:
                dest = move_to_label(paths, img_path, "litoria_aurea")
                print(f"Moved -> {dest}")
            except Exception as e:
                print(f"[ERROR] Failed to move file (frog label): {img_path} ({e})", flush=True)
            continue
        if key == "n":
            try:
                dest = move_to_label(paths, img_path, "background")
                print(f"Moved -> {dest}")
            except Exception as e:
                print(f"[ERROR] Failed to move file (background label): {img_path} ({e})", flush=True)
            continue

        print(f"[WARN] Unrecognized key '{key}'. Skipping.")

    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

