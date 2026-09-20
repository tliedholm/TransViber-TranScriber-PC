"""Paths that differ between running from source and the frozen (PyInstaller) build.

Running from source, everything lives in the project tree and is writable.
In the standalone build the bundled files live in the read-only _internal
folder next to the exe (which may be under Program Files), so logs and other
writable files go to the per-user app-data folder instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IS_FROZEN = bool(getattr(sys, "frozen", False))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bundle_dir() -> Path:
    """Where bundled read-only data lives (project root when running from source)."""
    if IS_FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return _PROJECT_ROOT


def asset_path(name: str) -> Path:
    """A file from app/assets (the window icon and friends).

    The spec bundles that folder at the root of the frozen build, so the
    layout differs between source and build like every other path here —
    which is why this belongs in runtime.py rather than being spelled out
    from __file__ at the call site.
    """
    if IS_FROZEN:
        return bundle_dir() / "assets" / name
    return bundle_dir() / "app" / "assets" / name


def writable_dir() -> Path:
    """Per-user folder for logs and other files the app writes."""
    if not IS_FROZEN:
        return _PROJECT_ROOT
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    path = base / "TransViber TransScriber"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_model_dir(model_name: str) -> Path | None:
    """The preloaded CTranslate2 model folder shipped inside the build, if any."""
    path = bundle_dir() / "models" / model_name
    return path if (path / "model.bin").is_file() else None
