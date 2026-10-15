"""The system folder dialog, for choosing recordings on this computer."""

from __future__ import annotations


def choose_folder() -> str | None:
    """Ask for a folder with the system dialog; None when cancelled or when no dialog can open.

    The app runs on the person's own computer, so the dialog opens on their screen.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        return None
    root.withdraw()
    # Without this the dialog can open behind the browser window.
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(parent=root, title="Choose the folder or memory card with the recordings")
    finally:
        root.destroy()
    return chosen or None
