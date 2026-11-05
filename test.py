"""Manual end-to-end smoke test against the REAL services (Whisper/Sarvam + Gemini).

This is not part of the automated test suite (that lives in ``tests/`` and runs
offline with ``python -m pytest``). Running this script makes real API calls.

Usage (PowerShell):
    python test.py "https://youtu.be/<id>" english
    python test.py "C:\\path\\to\\clip.mp3" hinglish
"""

from __future__ import annotations

import sys

from core.config import load_environment

load_environment()

from main import print_progress  # noqa: E402
from core.pipeline import PipelineError, run_pipeline  # noqa: E402


def run(source: str, language: str) -> int:
    try:
        record = run_pipeline(source, language, progress=print_progress)
    except PipelineError as exc:
        print(f"\nFAILED at '{exc.stage_label}': {exc}")
        return 1

    transcript = record.transcript
    print("\nTRANSCRIPT (first 500 chars):\n", transcript[:500] + ("…" if len(transcript) > 500 else ""))
    for heading, content in [
        ("TITLE", record.title),
        ("SUMMARY", record.summary),
        ("ACTION ITEMS", record.action_items),
        ("KEY DECISIONS", record.key_decisions),
        ("OPEN QUESTIONS", record.open_questions),
    ]:
        print(f"\n{'=' * 60}\n{heading}\n{'=' * 60}\n{content}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "english"))
