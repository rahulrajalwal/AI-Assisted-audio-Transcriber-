"""Lightweight persistent meeting history: one JSON file per processed meeting.

Stores the transcript and the Gemini analysis so previous meetings can be
reopened (and re-processed meetings reuse their analysis) without new LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from core.config import meetings_dir

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[0-9a-f]{8,64}$")


@dataclass
class MeetingRecord:
    meeting_id: str
    title: str
    transcript: str
    summary: str
    action_items: str
    key_decisions: str
    open_questions: str
    source: str = ""
    language: str = ""
    processed_at: str = ""
    model: str = ""
    duration_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeetingRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def _record_path(meeting_id: str, directory: Path | None = None) -> Path:
    if not _ID_PATTERN.match(meeting_id or ""):
        raise ValueError(f"Invalid meeting id: {meeting_id!r}")
    return (directory or meetings_dir()) / f"{meeting_id}.json"


def save_meeting(record: MeetingRecord, directory: Path | None = None) -> Path:
    """Atomically write the meeting record to disk."""
    path = _record_path(record.meeting_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def load_meeting(meeting_id: str, directory: Path | None = None) -> MeetingRecord | None:
    try:
        path = _record_path(meeting_id, directory)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return MeetingRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read meeting record %s: %s", path.name, exc)
        return None


def list_meetings(directory: Path | None = None) -> list[MeetingRecord]:
    """All saved meetings, newest first."""
    folder = directory or meetings_dir()
    if not folder.exists():
        return []
    records = []
    for path in folder.glob("*.json"):
        record = load_meeting(path.stem, folder)
        if record:
            records.append(record)
    return sorted(records, key=lambda r: r.processed_at or "", reverse=True)
