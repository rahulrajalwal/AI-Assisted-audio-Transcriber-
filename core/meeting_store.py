"""Lightweight persistent meeting history: one JSON file per processed meeting.

Stores the transcript and the LLM analysis so previous meetings can be reopened
(and re-processed meetings reuse their analysis) without new LLM calls. Records
written by older versions load fine — missing fields get defaults.
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
    meeting_date: str = ""  # ISO date of the meeting (user-provided; defaults to processing date)
    source_type: str = ""  # "youtube" | "upload" | "local_file"
    # Timed transcript segments: {"start", "end", "audio_chunk", "char_start", "char_end"}
    segments: list[dict[str, Any]] = field(default_factory=list)
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


# ── Chat history (saved per meeting so Q&A survives refreshes and restarts) ─────


def _chat_path(meeting_id: str, directory: Path | None = None) -> Path:
    if not _ID_PATTERN.match(meeting_id or ""):
        raise ValueError(f"Invalid meeting id: {meeting_id!r}")
    return (directory or meetings_dir()) / "chats" / f"{meeting_id}.json"


def load_chats(meeting_id: str, directory: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Saved conversations for a meeting, keyed by scope ("This meeting" / "All meetings")."""
    try:
        path = _chat_path(meeting_id, directory)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): list(v) for k, v in data.items() if isinstance(v, list)}
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read chat history for %s: %s", meeting_id, exc)
    return {}


def save_chats(meeting_id: str, chats: dict[str, list[dict[str, Any]]], directory: Path | None = None) -> None:
    """Atomically save a meeting's conversations."""
    path = _chat_path(meeting_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(chats, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
