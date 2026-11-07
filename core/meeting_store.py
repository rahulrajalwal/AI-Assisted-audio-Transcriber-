"""Lightweight persistence for meetings — no database needed.

* Meeting history: one JSON file per processed meeting (transcript + analysis), so
  previous meetings can be reopened without new LLM calls. Records written by older
  versions load fine — missing fields get defaults.
* Chat history: one JSON file per meeting in ``chats/``.
* Processing checkpoints (:class:`MeetingCheckpoint`): the audio, chunk plan, finished
  chunk transcripts and finished LLM steps of a meeting that is still being processed,
  so a retry resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from core.config import meetings_dir, work_dir

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[0-9a-f]{8,64}$")


@dataclass
class MeetingRecord:
    meeting_id: str
    title: str  # AI-generated meeting title
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
    source_title: str = ""  # original YouTube title or uploaded filename (shown first)
    status: str = "complete"  # "indexing" while vectors are being stored, then "complete"
    vector_chunks: int = 0  # number of transcript chunks stored in ChromaDB for this meeting
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Original YouTube title / filename; older records fall back to source or AI title."""
        if self.source_title:
            return self.source_title
        if self.source and not self.source.lower().startswith("http"):
            return Path(self.source).name
        return self.title or self.source or self.meeting_id

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


# ── Processing checkpoints (resume long recordings and failed LLM steps) ───────


class MeetingCheckpoint:
    """Resumable processing state of one meeting, in ``<WORK_DIR>/jobs/<meeting_id>/``.

    ``checkpoint.json`` records the source fingerprint, the audio-chunk plan, every
    finished chunk's transcript segments and every finished LLM result. The folder also
    holds the downloaded/converted audio and the chunk files, so nothing finished is
    redone on retry. It is deleted once the meeting is fully processed.
    """

    FILE = "checkpoint.json"

    def __init__(self, meeting_id: str):
        if not _ID_PATTERN.match(meeting_id or ""):
            raise ValueError(f"Invalid meeting id: {meeting_id!r}")
        self.meeting_id = meeting_id
        self.dir = work_dir() / "jobs" / meeting_id
        self.path = self.dir / self.FILE
        self.data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable checkpoint %s: %s", self.path, exc)

    def save(self) -> None:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def start(self, fingerprint: str, **info: Any) -> bool:
        """Begin or continue processing. Returns True if earlier progress is reused.

        If the saved checkpoint belongs to a different source (fingerprint mismatch),
        its stale files are discarded and processing starts fresh.
        """
        reused = self.data.get("fingerprint") == fingerprint
        if not reused:
            if self.dir.exists():
                shutil.rmtree(self.dir, ignore_errors=True)
            self.data = {"meeting_id": self.meeting_id, "fingerprint": fingerprint, "chunks": {}, "llm": {}}
        self.data.update(info)
        self.save()
        return reused

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self.save()

    def completed_chunks(self) -> dict[int, list[dict[str, Any]]]:
        return {int(index): segments for index, segments in (self.data.get("chunks") or {}).items()}

    def save_chunk(self, index: int, segments: list[dict[str, Any]]) -> None:
        self.data.setdefault("chunks", {})[str(index)] = segments
        self.save()

    # Mapping interface used as the LLM-result cache (see core.llm.invoke_with_retry).
    def get(self, key: str, default: Any = None) -> Any:
        return (self.data.get("llm") or {}).get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data.setdefault("llm", {})[key] = value
        self.save()

    def discard_files(self) -> None:
        """Delete the audio kept for resuming (once the transcript is saved); keep checkpoint.json."""
        for path in self.dir.glob("*"):
            if path.name != self.FILE:
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)

    def remove(self) -> None:
        """Delete the checkpoint folder (audio files included) after successful processing."""
        jobs_root = (work_dir() / "jobs").resolve()
        if self.dir.exists() and jobs_root in self.dir.resolve().parents:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.data = {}


def list_checkpoints() -> list[dict[str, Any]]:
    """Unfinished processing jobs (newest first), e.g. to offer "Resume" after a restart."""
    root = work_dir() / "jobs"
    jobs = []
    for path in root.glob(f"*/{MeetingCheckpoint.FILE}") if root.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        chunk_plan = data.get("chunk_plan") or []
        jobs.append({
            "meeting_id": data.get("meeting_id", path.parent.name),
            "source_title": data.get("source_title") or data.get("source") or path.parent.name,
            "language": data.get("language", ""),
            "status": data.get("status", ""),
            "error": data.get("error", ""),
            "chunks_done": len(data.get("chunks") or {}),
            "chunks_total": len(chunk_plan),
            "transcribed": bool(data.get("transcript")),
            "audio_saved": bool(data.get("source_file")) and Path(data["source_file"]).exists(),
            "updated_at": data.get("updated_at", ""),
        })
    return sorted(jobs, key=lambda job: job["updated_at"], reverse=True)
