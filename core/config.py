"""Project-wide configuration: filesystem paths and environment loading.

Every module that reads environment variables imports this module (directly or
through ``core/__init__.py``), so ``.env`` is always loaded before any
configuration value is read — regardless of import order in ``app.py``/``main.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

_env_loaded = False


def load_environment() -> None:
    """Load ``.env`` from the project root once.

    Existing process environment variables win over values in ``.env``
    (``override=False``), so shell-level configuration is never clobbered.
    """
    global _env_loaded
    if not _env_loaded:
        load_dotenv(ENV_FILE, override=False)
        _env_loaded = True


def _path_from_env(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name, "").strip()
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def chroma_dir() -> Path:
    """Persistent ChromaDB directory (``CHROMA_DIR`` overrides; default ``vector_db/``)."""
    return _path_from_env("CHROMA_DIR", PROJECT_ROOT / "vector_db")


def meetings_dir() -> Path:
    """Directory holding one JSON record per processed meeting."""
    return _path_from_env("MEETINGS_DIR", PROJECT_ROOT / "data" / "meetings")


def work_dir() -> Path:
    """Scratch directory for downloads, converted audio and audio chunks."""
    return _path_from_env("WORK_DIR", PROJECT_ROOT / "downloads")


load_environment()
