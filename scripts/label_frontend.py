from __future__ import annotations



import io
import os
import random
import shutil
from pathlib import Path

import streamlit as st

import label_spectrograms as core


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _inject_sidebar_css() -> None:
    # System-based light/dark sidebar styling using prefers-color-scheme,
    # with targeted selectors to keep widgets readable.
    st.markdown(
        """
<style>
section[data-testid="stSidebar"] {
  /* Light mode defaults */
  --sb-bg: #f8fafc;
  --sb-fg: #0f172a;
  --sb-muted: rgba(15, 23, 42, 0.70);
  --sb-border: rgba(148, 163, 184, 0.35);
  --sb-input-bg: #ffffff;
  --sb-input-border: rgba(148, 163, 184, 0.55);
  --sb-link: #1d4ed8;
  --sb-placeholder: rgba(15, 23, 42, 0.50);

  background: var(--sb-bg);
  border-right: 1px solid var(--sb-border);
}

@media (prefers-color-scheme: dark) {
  section[data-testid="stSidebar"] {
    --sb-bg: #0b1220;
    --sb-fg: #e2e8f0;
    --sb-muted: rgba(226, 232, 240, 0.70);
    --sb-border: rgba(148, 163, 184, 0.25);
    --sb-input-bg: #111a2e;
    --sb-input-border: rgba(226, 232, 240, 0.25);
    --sb-link: #60a5fa;
    --sb-placeholder: rgba(226, 232, 240, 0.55);
  }
}

/* Force readable TEXT in sidebar (avoid blanket * selector) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: var(--sb-fg) !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--sb-muted) !important;
}
section[data-testid="stSidebar"] a {
  color: var(--sb-link) !important;
}

/* Inputs / selects background */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background-color: var(--sb-input-bg) !important;
  color: var(--sb-fg) !important;
  border-color: var(--sb-input-border) !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
  color: var(--sb-placeholder) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: var(--sb-input-bg) !important;
  color: var(--sb-fg) !important;
  border-color: var(--sb-input-border) !important;
}
/* Dropdown menu itself */
section[data-testid="stSidebar"] [data-baseweb="menu"] {
  background-color: var(--sb-input-bg) !important;
}
section[data-testid="stSidebar"] [data-baseweb="menu"] * {
  color: var(--sb-fg) !important;
}

/* Make sidebar buttons feel a bit more “app-like” */
section[data-testid="stSidebar"] button {
  border-radius: 12px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _candidate_dirs(repo_root: Path) -> dict[str, list[str]]:
    """
    Provide “browse” dropdown options without needing OS file dialogs.
    (File dialogs are unreliable in Streamlit because the server owns the UI.)
    """
    out: dict[str, list[str]] = {
        "spectrogram_root": [],
        "raw_root": [],
        "processed_root": [],
        "training_root": [],
    }

    candidates = {
        "spectrogram_root": [
            repo_root / "processed" / "external" / "spectrograms",
            repo_root / "processed" / "ponds" / "spectrograms",
        ],
        "raw_root": [
            repo_root / "raw" / "external",
            repo_root / "raw" / "ponds",
        ],
        "processed_root": [
            repo_root / "processed" / "external" / "chunks",
            repo_root / "processed" / "ponds" / "chunks",
        ],
        "training_root": [
            repo_root / "labeled",
        ],
    }

    for k, paths in candidates.items():
        seen: set[str] = set()
        for p in paths:
            if p.exists():
                s = str(p)
                if s not in seen:
                    out[k].append(s)
                    seen.add(s)

    # Add immediate subfolders for spectrograms/processed (helps choose dataset-scoped subdirs).
    for base_key, base_path in [("spectrogram_root", repo_root / "processed"), ("processed_root", repo_root / "processed")]:
        if base_path.exists():
            try:
                for child in base_path.iterdir():
                    if child.is_dir():
                        out[base_key].append(str(child))
            except Exception:
                pass

    # De-duplicate while preserving order
    for k in out:
        deduped: list[str] = []
        seen2: set[str] = set()
        for s in out[k]:
            if s not in seen2:
                deduped.append(s)
                seen2.add(s)
        out[k] = deduped

    return out


def _iter_pngs(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for name in filenames:
            if name.lower().endswith(".png"):
                out.append(d / name)
    return sorted(out)


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()

def _count_pngs(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for _ in p.rglob("*.png"))

def _safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(f"{dst.stem}__dup{random.randint(1000,9999)}{dst.suffix}")
    shutil.move(str(src), str(dst))
    return dst

def _label_and_track(paths: core.Paths, img_path: Path, label_name: str) -> None:
    """Move the image to the label folder and store undo info in session_state."""
    dest_dir = paths.frog_dir if label_name == "litoria_aurea" else paths.bg_dir
    dest = _safe_move(img_path, dest_dir / img_path.name)
    st.session_state["last_action"] = {
        "src": str(img_path),
        "dst": str(dest),
        "label": label_name,
    }


def main() -> None:
    st.set_page_config(page_title="Frog Spectrogram Labeler", layout="wide")
    _inject_sidebar_css()
    st.title("Frog Spectrogram Labeler")
    st.caption("Listen to the 5-second audio and label the spectrogram. This tool moves images into training folders.")

    repo_root = _repo_root()

    # Defaults
    default_spectrogram_root = repo_root / "processed" / "external" / "spectrograms"
    default_raw_root = repo_root / "raw" / "external"
    default_processed_root = repo_root / "processed" / "external" / "chunks"
    default_training_root = repo_root / "labeled"
    browse = _candidate_dirs(repo_root)

    with st.sidebar:
        st.header("Quick start")
        st.write("1) Press Play and listen")
        st.write("2) Click Frog or Background")
        st.write("3) Use Replay if needed")

        with st.expander("Advanced settings", expanded=False):
            st.subheader("Paths")
            spectrogram_root_str = st.text_input(
                "spectrogram_root",
                str(default_spectrogram_root),
                help="Where spectrogram PNGs are read from.",
            )
            if browse["spectrogram_root"]:
                spectrogram_pick = st.selectbox(
                    "Browse spectrogram folders",
                    options=browse["spectrogram_root"],
                    index=browse["spectrogram_root"].index(str(default_spectrogram_root))
                    if str(default_spectrogram_root) in browse["spectrogram_root"]
                    else 0,
                    help="Pick a common folder instead of typing a path.",
                )
                if spectrogram_pick:
                    spectrogram_root_str = spectrogram_pick

            raw_root_str = st.text_input(
                "raw_root",
                str(default_raw_root),
                help="Where original audio lives. Used to export 5-second WAV chunks for playback.",
            )
            if browse["raw_root"]:
                raw_pick = st.selectbox(
                    "Browse raw audio folders",
                    options=browse["raw_root"],
                    index=browse["raw_root"].index(str(default_raw_root)) if str(default_raw_root) in browse["raw_root"] else 0,
                    help="Pick a common folder instead of typing a path.",
                )
                if raw_pick:
                    raw_root_str = raw_pick

            processed_root_str = st.text_input(
                "processed_root",
                str(default_processed_root),
                help="Cache for exported 5-second WAV chunks (auto-created).",
            )
            if browse["processed_root"]:
                processed_pick = st.selectbox(
                    "Browse processed folders",
                    options=browse["processed_root"],
                    index=browse["processed_root"].index(str(default_processed_root))
                    if str(default_processed_root) in browse["processed_root"]
                    else 0,
                    help="Pick a common folder instead of typing a path.",
                )
                if processed_pick:
                    processed_root_str = processed_pick

            training_root_str = st.text_input(
                "training_root",
                str(default_training_root),
                help="Where labeled images are moved to (litoria_aurea / non_target).",
            )
            if browse["training_root"]:
                training_pick = st.selectbox(
                    "Browse training folders",
                    options=browse["training_root"],
                    index=browse["training_root"].index(str(default_training_root))
                    if str(default_training_root) in browse["training_root"]
                    else 0,
                    help="Pick a common folder instead of typing a path.",
                )
                if training_pick:
                    training_root_str = training_pick

            spectrogram_root = Path(spectrogram_root_str)
            raw_root = Path(raw_root_str)
            processed_root = Path(processed_root_str)
            training_root = Path(training_root_str)

            st.subheader("Sampling")
            shuffle = st.checkbox("shuffle", value=True, help="If on, shows images in random order.")

            seed = st.number_input(
                "Seed (shuffle order)",
                key="seed_input",
                min_value=0,
                max_value=10_000_000,
                value=1337,
                step=1,
                help="Controls the random order when shuffle is on. Same seed = same order. Change it to reshuffle.",
            )

            limit = st.number_input(
                "Limit (0 = all)",
                key="limit_input",
                min_value=0,
                value=0,
                step=1,
                help="How many images to label in this session. 0 means label everything available.",
            )

            st.subheader("Audio")
            auto_export = st.checkbox(
                "auto-export missing audio from raw",
                value=True,
                help="If the 5-second WAV chunk isn't cached yet, create it automatically from the raw audio.",
            )

        # Use defaults if advanced expander hasn't been opened yet
        spectrogram_root = locals().get("spectrogram_root", default_spectrogram_root)
        raw_root = locals().get("raw_root", default_raw_root)
        processed_root = locals().get("processed_root", default_processed_root)
        training_root = locals().get("training_root", default_training_root)
        shuffle = locals().get("shuffle", True)
        seed = locals().get("seed", 1337)
        limit = locals().get("limit", 0)
        auto_export = locals().get("auto_export", True)

        st.divider()
        if st.button("Reload images", use_container_width=True):
            st.session_state.pop("images", None)
            st.session_state.pop("idx", None)
            st.session_state.pop("audio_replay", None)
            st.rerun()

    if not spectrogram_root.exists():
        st.error(f"spectrogram_root not found: {spectrogram_root}")
        st.stop()

    # Initialize session state
    if "images" not in st.session_state:
        images = _iter_pngs(spectrogram_root)
        if shuffle:
            rnd = random.Random(int(seed))
            rnd.shuffle(images)
        if int(limit) > 0:
            images = images[: int(limit)]
        st.session_state["images"] = images
        st.session_state["idx"] = 0
        st.session_state["audio_replay"] = 0

    images: list[Path] = st.session_state["images"]
    idx: int = st.session_state["idx"]

    # Simple labeling stats
    frog_count = _count_pngs(training_root / "litoria_aurea")
    bg_count = _count_pngs(training_root / "non_target")
    remaining = len(images) - idx

    if not images:
        st.warning("No PNGs found under spectrogram_root.")
        st.stop()

    if idx >= len(images):
        st.success("Done! No more images in this session.")
        st.stop()

    img_path = images[idx]
    progress = f"Image {idx + 1} of {len(images)}"

    paths = core.Paths(
        repo_root=repo_root,
        spectrogram_root=spectrogram_root,
        raw_root=raw_root,
        processed_root=processed_root,
        training_root=training_root,
        frog_dir=training_root / "litoria_aurea",
        bg_dir=training_root / "non_target",
    )
    paths.frog_dir.mkdir(parents=True, exist_ok=True)
    paths.bg_dir.mkdir(parents=True, exist_ok=True)

    st.subheader(progress)
    st.progress((idx + 1) / max(len(images), 1))
    m1, m2, m3 = st.columns(3)
    m1.metric("Frog labeled", frog_count)
    m2.metric("Background labeled", bg_count)
    m3.metric("Remaining (this session)", max(remaining, 0))

    with st.expander("How to label (tips)", expanded=False):
        st.write("- **Frog**: you can hear the Green & Golden Bell Frog call in the 5s audio.")
        st.write("- **Background**: no target frog call (wind, insects, other species, silence).")
        st.write("- Use **Replay** if you’re unsure, or **Skip** and come back later.")

    st.caption(str(img_path.relative_to(repo_root)) if img_path.is_absolute() and repo_root in img_path.parents else str(img_path))

    # Load/display image as bytes (avoids Windows file locking issues).
    img_bytes = _read_bytes(img_path)

    col_img, col_controls = st.columns([2, 1], gap="large")

    with col_img:
        st.image(img_bytes, use_container_width=True)

    # Prepare audio for this item
    audio_path = core.find_processed_chunk_audio(paths, img_path)
    if audio_path is None and auto_export:
        with st.spinner("Exporting 5s audio chunk from raw..."):
            audio_path = core.export_processed_chunk_from_raw(paths, img_path)

    audio_bytes: bytes | None = None
    if audio_path is not None and audio_path.exists() and audio_path.suffix.lower() == ".wav":
        audio_bytes = _read_bytes(audio_path)

    with col_controls:
        st.markdown("### Actions")

        if audio_bytes is not None:
            # Use Streamlit's native audio widget (most reliable across browsers).
            st.audio(io.BytesIO(audio_bytes), format="audio/wav")
            if audio_path is not None:
                st.download_button(
                    "Download WAV (for external playback)",
                    data=audio_bytes,
                    file_name=audio_path.name,
                    mime="audio/wav",
                    use_container_width=True,
                )
        else:
            st.info("No WAV audio available for this item (yet).")
            if audio_path is not None:
                st.caption(f"Found audio file but could not read as WAV: {audio_path}")

        c1, c2 = st.columns(2, gap="small")
        with c1:
            if st.button("Replay", use_container_width=True):
                st.session_state["audio_replay"] += 1
                st.rerun()
        with c2:
            if st.button("Skip", use_container_width=True):
                st.session_state["idx"] += 1
                st.session_state["audio_replay"] += 1
                st.rerun()

        if st.button("Undo last label", use_container_width=True, help="Move the last labeled image back to the spectrogram folder."):
            act = st.session_state.get("last_action")
            if not act:
                st.warning("Nothing to undo yet.")
            else:
                src = Path(act["src"])
                dst = Path(act["dst"])
                if dst.exists():
                    _safe_move(dst, src)
                    st.session_state["last_action"] = None
                    st.success("Undid last label.")
                    st.rerun()
                else:
                    st.warning("Cannot undo (file not found in training folder).")

        st.divider()

        if st.button("Frog (target)", type="primary", use_container_width=True):
            _label_and_track(paths, img_path, "litoria_aurea")
            st.session_state["idx"] += 1
            st.session_state["audio_replay"] += 1
            st.rerun()

        if st.button("Background (no target frog)", use_container_width=True):
            _label_and_track(paths, img_path, "non_target")
            st.session_state["idx"] += 1
            st.session_state["audio_replay"] += 1
            st.rerun()

        st.divider()
        if st.button("End session", use_container_width=True):
            st.stop()


if __name__ == "__main__":
    main()

