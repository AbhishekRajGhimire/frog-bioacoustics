from __future__ import annotations

from pathlib import Path

import streamlit as st

from frog_classifier.data import load_preprocessing_config
from frog_classifier.labeling import (
    DECISION_FOLDERS,
    LOG_NAME,
    DecisionLog,
    build_queue,
    confirm_decision,
    current_folder,
    find_or_export_chunk,
    record_decision,
    summarize_labels,
    undo_move,
)
from frog_classifier.preprocessing.display import colorize


REPO_ROOT = Path(__file__).resolve().parents[1]
MODE_LABEL = "Label new clips"
MODE_AUDIT = "Audit labeled clips"
# Two rows of two: the frequent pair (Frog, Background) sits on top.
BUTTON_ROWS = (
    (
        ("Frog (target)", "litoria_aurea", False),
        ("Background (no target frog)", "non_target", False),
    ),
    (
        ("Frog, faint", "litoria_aurea", True),
        ("Unsure (review later)", "unsure", False),
    ),
)


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


def _pngs_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".png")


def _labeled_pngs(labeled_root: Path) -> list[Path]:
    clips: list[Path] = []
    for folder in DECISION_FOLDERS:
        directory = labeled_root / folder
        if directory.is_dir():
            clips.extend(
                sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".png")
            )
    return clips


def _reset_session() -> None:
    for key in ("queue", "index", "last_move", "error"):
        st.session_state.pop(key, None)


def _decide(clip: Path, decision: str, faint: bool, *, labeled_root: Path, log: DecisionLog, chunk_seconds: int) -> None:
    try:
        if current_folder(clip, labeled_root) == decision:
            confirm_decision(clip, labeled_root=labeled_root, log=log, faint=True if faint else None, chunk_seconds=chunk_seconds)
            st.session_state["last_move"] = None
        else:
            clip_index = st.session_state["index"]
            move = record_decision(
                clip, decision, labeled_root=labeled_root, log=log, faint=faint, chunk_seconds=chunk_seconds,
            )
            st.session_state["last_move"] = (clip_index, move)
    except (ValueError, OSError) as error:
        st.session_state["error"] = str(error)
        return
    st.session_state.pop("error", None)
    st.session_state["index"] += 1


def main() -> None:
    st.set_page_config(page_title="Frog Spectrogram Labeler", layout="wide")
    _inject_sidebar_css()
    st.title("Frog Spectrogram Labeler")
    config = load_preprocessing_config(REPO_ROOT / "config" / "preprocessing.toml")
    chunk_seconds = config.audio.chunk_seconds

    with st.sidebar:
        mode = st.radio("Mode", (MODE_LABEL, MODE_AUDIT))
        st.header("Quick start")
        st.write("1) Press Play and listen")
        st.write("2) Choose Frog, Frog faint, Background, or Unsure")
        st.write("3) In audit mode, pressing the current label confirms it")
        with st.expander("Advanced settings", expanded=False):
            spectrogram_root = Path(st.text_input("spectrogram_root", str(REPO_ROOT / "processed" / "external" / "spectrograms")))
            raw_root = Path(st.text_input("raw_root", str(REPO_ROOT / "raw" / "external")))
            chunk_root = Path(st.text_input("chunk_root", str(REPO_ROOT / "processed" / "external" / "chunks")))
            labeled_root = Path(st.text_input("labeled_root", str(REPO_ROOT / "labeled")))
            seed = int(st.number_input("Seed (queue order)", min_value=0, max_value=10_000_000, value=1337, step=1))
            limit = int(st.number_input("Limit (0 = all)", min_value=0, value=0, step=1))
            cap = int(st.number_input("Clips per recording per session", min_value=1, max_value=100, value=5, step=1))
        st.divider()
        if st.button("Reload clips", use_container_width=True):
            _reset_session()
            st.rerun()

    settings = (mode, str(spectrogram_root), str(raw_root), str(chunk_root), str(labeled_root), seed, limit, cap)
    if st.session_state.get("settings") != settings:
        if "settings" in st.session_state:
            st.session_state["notice"] = "Settings changed, so the queue was rebuilt."
        _reset_session()
        st.session_state["settings"] = settings

    log = DecisionLog(labeled_root / LOG_NAME)

    if "queue" not in st.session_state:
        source = _pngs_under(spectrogram_root) if mode == MODE_LABEL else _labeled_pngs(labeled_root)
        per_recording_cap = cap if mode == MODE_LABEL else max(len(source), 1)
        queue = build_queue(source, seed=seed, chunk_seconds=chunk_seconds, per_recording_cap=per_recording_cap)
        if limit > 0:
            queue = queue[:limit]
        st.session_state.update(queue=queue, index=0, last_move=None)

    queue: list[Path] = st.session_state["queue"]
    index: int = st.session_state["index"]

    progress = summarize_labels(labeled_root, chunk_seconds=chunk_seconds)
    columns = st.columns(4)
    for column, folder, title in zip(columns, DECISION_FOLDERS, ("Frog", "Background", "Unsure")):
        column.metric(title, progress[folder].clips, f"{progress[folder].recordings} recordings")
    columns[3].metric("Remaining (this session)", max(len(queue) - index, 0))

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    if st.session_state.get("notice"):
        st.info(st.session_state.pop("notice"))

    if not queue:
        st.warning("No clips found for this mode.")
        st.stop()
    if index >= len(queue):
        st.success("Done. No more clips in this session.")
        st.stop()

    clip = queue[index]
    st.subheader(f"Clip {index + 1} of {len(queue)}")
    st.progress((index + 1) / len(queue))
    st.caption(str(clip))
    if mode == MODE_AUDIT:
        origin = current_folder(clip, labeled_root)
        if origin is None:
            st.warning("This clip is not inside the labeled root; press Reload clips.")
        else:
            st.info(f"Current label: {origin}")

    col_img, col_controls = st.columns([2, 1], gap="large")
    audio_path = None
    if not clip.is_file():
        with col_img:
            st.error("This clip no longer exists on disk; press Skip or Reload clips.")
    else:
        with col_img:
            st.image(colorize(clip.read_bytes()), use_container_width=True)
        audio_path = find_or_export_chunk(
            clip, raw_root=raw_root, chunk_root=chunk_root, config=config,
            spectrogram_root=spectrogram_root if mode == MODE_LABEL else None,
        )

    with col_controls:
        st.markdown("### Actions")
        if audio_path is not None:
            st.audio(audio_path.read_bytes(), format="audio/wav")
        else:
            st.info("No audio found for this clip.")
        for row in BUTTON_ROWS:
            for column, (label, decision, faint) in zip(st.columns(2, gap="small"), row):
                kind = "primary" if decision == "litoria_aurea" and not faint else "secondary"
                with column:
                    if st.button(label, type=kind, use_container_width=True):
                        _decide(clip, decision, faint, labeled_root=labeled_root, log=log, chunk_seconds=chunk_seconds)
                        st.rerun()
        left, right = st.columns(2, gap="small")
        with left:
            if st.button("Skip", use_container_width=True):
                st.session_state.pop("error", None)
                st.session_state["index"] += 1
                st.rerun()
        with right:
            if st.button("Undo last move", use_container_width=True):
                last_move = st.session_state.get("last_move")
                if last_move is None:
                    st.session_state["error"] = "Nothing to undo."
                else:
                    clip_index, move = last_move
                    try:
                        undo_move(move, labeled_root=labeled_root, log=log, chunk_seconds=chunk_seconds)
                        st.session_state["last_move"] = None
                        st.session_state["index"] = clip_index
                        st.session_state.pop("error", None)
                    except (ValueError, OSError) as error:
                        st.session_state["error"] = str(error)
                st.rerun()
        st.divider()
        if st.button("End session", use_container_width=True):
            st.stop()


if __name__ == "__main__":
    main()
