"""Persistent multi-meeting transcript storage: text chunks → MiniLM embeddings → ChromaDB.

Two different chunking steps exist in the project and stay separate:
* AUDIO chunking (utils/audio_processor.py): 10-minute WAV pieces for Whisper/Sarvam.
* TEXT chunking (here): the finished transcript is split into ~800-character,
  sentence-aware chunks, embedded with ``all-MiniLM-L6-v2`` and stored in ChromaDB.

Storage design
--------------
* One persistent collection (``meeting_transcript``) holds every meeting. Nothing
  is ever cleared or recreated; chunks written by older versions (no metadata)
  stay searchable in "all meetings" mode.
* Every chunk has a deterministic ID ``<meeting_id>-<chunk_index>`` and metadata:
  meeting_id, title (AI title), source_title (original YouTube title / filename),
  meeting_date / meeting_ts, processed_at, source, source_type,
  language, chunk_id, chunk_index, char_start/char_end, audio time range
  (start_seconds / end_seconds / audio_chunk when timestamps are available),
  embedding_model, chunk_size, schema_version.
* Re-processing a meeting never duplicates chunks: if its chunks are already all
  stored it is skipped; an incomplete index (e.g. interrupted) or an intentional
  re-index (``replace=True``) upserts that meeting's chunks under the same IDs and
  deletes only its stale chunks — other meetings are never touched.
* :func:`backup_vector_store` copies the database folder before risky operations.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import re
import shutil
import threading
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import PROJECT_ROOT, chroma_dir

logger = logging.getLogger(__name__)

COLLECTION_NAME = "meeting_transcript"  # unchanged, so existing data keeps working
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 embeds up to 256 word-pieces (~1,000 characters) and was trained on
# ~128-token passages. 800 characters (~170 tokens) keeps whole exchanges together
# without being truncated by the model; 120 characters of overlap (~1–2 sentences)
# preserves context across boundaries. See README "Text chunking" for the evaluation.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
SCHEMA_VERSION = 2
LEGACY_MEETING_ID = "legacy"

_store_lock = threading.Lock()
_embeddings = None
_embeddings_lock = threading.Lock()


# ── Embeddings ──────────────────────────────────────────────────────────────────


def get_embeddings():
    """Load the sentence-transformers embedding model once per process (thread-safe)."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = _load_embeddings()
    return _embeddings


def warm_up_embeddings() -> None:
    """Load the embedding model in a background thread (importing torch is slow on first use)."""
    threading.Thread(target=get_embeddings, name="embeddings-warmup", daemon=True).start()


def _load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    def build(local_only: bool):
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Prefer the local cache: avoids slow Hugging Face Hub network checks on every start.
    try:
        return build(local_only=True)
    except Exception:
        logger.info("Embedding model not cached locally; downloading %s…", EMBEDDING_MODEL)
        return build(local_only=False)


# ── Store access ────────────────────────────────────────────────────────────────


def compute_meeting_id(transcript: str) -> str:
    """Stable ID derived from the transcript content (whitespace/case-insensitive)."""
    normalized = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=8)
def _open_store(persist_directory: str, collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def load_vector_store(persist_directory: str | None = None) -> Chroma:
    """Open (or create) the persistent collection. Never deletes existing data."""
    directory = str(persist_directory or chroma_dir())
    return _open_store(directory, COLLECTION_NAME)


def meeting_exists(vector_store: Chroma, meeting_id: str) -> bool:
    result = vector_store.get(where={"meeting_id": meeting_id}, limit=1, include=[])
    return bool(result.get("ids"))


def count_meeting_chunks(meeting_id: str, vector_store: Chroma | None = None) -> int:
    store = vector_store or load_vector_store()
    return len(store.get(where={"meeting_id": meeting_id}, include=[]).get("ids") or [])


def backup_vector_store(persist_directory: str | Path | None = None) -> Path | None:
    """Copy the ChromaDB folder to ``backups/vector_db-<timestamp>``. Returns the backup path."""
    source = Path(persist_directory or chroma_dir())
    if not source.exists():
        return None
    target = PROJECT_ROOT / "backups" / f"{source.name}-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copytree(source, target)
    logger.info("Backed up %s to %s", source, target)
    return target


# ── Text chunking + metadata ────────────────────────────────────────────────────


def split_for_indexing(transcript: str) -> list[Document]:
    """Sentence-aware text chunks with their character offsets (``start_index``)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )
    return splitter.create_documents([transcript])


def _time_range(char_start: int, char_end: int, segments: list[dict] | None) -> dict[str, Any]:
    """Map a character span of the transcript to the audio time range it came from."""
    if not segments:
        return {}
    starts = [seg["char_start"] for seg in segments]
    first = max(0, bisect.bisect_right(starts, char_start) - 1)
    last = max(first, bisect.bisect_left(starts, char_end) - 1)
    covered = segments[first : last + 1]
    if not covered:
        return {}
    return {
        "start_seconds": float(covered[0]["start"]),
        "end_seconds": float(covered[-1]["end"]),
        "audio_chunk": int(covered[0].get("audio_chunk", 0)),
    }


def _meeting_timestamp(meeting_date: str, processed_at: str) -> int:
    for value in (meeting_date, processed_at):
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value) if "T" in value else datetime.combine(
                date.fromisoformat(value), datetime.min.time()
            )
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            return int(parsed.timestamp())
        except ValueError:
            continue
    return int(datetime.now(timezone.utc).timestamp())


def build_documents(
    transcript: str,
    *,
    meeting_id: str,
    title: str,
    source: str,
    source_type: str,
    language: str,
    processed_at: str,
    meeting_date: str,
    segments: list[dict] | None,
    source_title: str = "",
) -> tuple[list[Document], list[str]]:
    meeting_ts = _meeting_timestamp(meeting_date, processed_at)
    docs, ids = [], []
    for index, chunk in enumerate(split_for_indexing(transcript)):
        char_start = int(chunk.metadata.get("start_index", 0))
        char_end = char_start + len(chunk.page_content)
        chunk_id = f"{meeting_id}-{index:05d}"
        metadata = {
            "meeting_id": meeting_id,
            "title": title or "Untitled Meeting",
            "source_title": source_title or title or "Untitled Meeting",
            "meeting_date": meeting_date or processed_at[:10],
            "meeting_ts": meeting_ts,
            "processed_at": processed_at,
            "source": source or "",
            "source_type": source_type or "",
            "language": language or "",
            "chunk_id": chunk_id,
            "chunk_index": index,
            "char_start": char_start,
            "char_end": char_end,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "schema_version": SCHEMA_VERSION,
        }
        metadata.update(_time_range(char_start, char_end, segments))
        docs.append(Document(page_content=chunk.page_content, metadata=metadata))
        ids.append(chunk_id)
    return docs, ids


def build_vector_store(
    transcript: str,
    *,
    meeting_id: str | None = None,
    title: str = "Untitled Meeting",
    source_title: str = "",
    source: str = "",
    source_type: str = "",
    language: str = "",
    processed_at: str | None = None,
    meeting_date: str = "",
    segments: list[dict] | None = None,
    replace: bool = False,
    vector_store: Chroma | None = None,
) -> Chroma:
    """Chunk, embed and store one meeting's transcript; return the persistent store.

    * All of this meeting's chunks already stored and ``replace`` is False → skipped.
    * Otherwise (new, partially indexed, or ``replace=True``) its chunks are upserted
      under deterministic IDs and leftover chunks of *this meeting only* are deleted,
      so reprocessing never creates duplicates.
    """
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Cannot index an empty transcript.")

    store = vector_store or load_vector_store()
    meeting_id = meeting_id or compute_meeting_id(text)
    processed_at = processed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    docs, ids = build_documents(
        text,
        meeting_id=meeting_id,
        title=title,
        source_title=source_title,
        source=source,
        source_type=source_type,
        language=language,
        processed_at=processed_at,
        meeting_date=meeting_date,
        segments=segments,
    )
    with _store_lock:
        existing_ids = set(store.get(where={"meeting_id": meeting_id}, include=[]).get("ids") or [])
        if existing_ids and not replace and set(ids) <= existing_ids:
            logger.info("Meeting %s is already indexed (%d chunks); skipping.", meeting_id, len(existing_ids))
            return store
        store.add_documents(documents=docs, ids=ids)  # upsert by deterministic ID
        stale = sorted(existing_ids - set(ids))
        if stale:
            store.delete(ids=stale)
        logger.info(
            "Indexed meeting %s: %d chunks (%s, %d stale removed).",
            meeting_id, len(ids), "replaced" if existing_ids else "new", len(stale),
        )
    return store


def get_retriever(vector_store: Chroma, k: int = 4, meeting_id: str | None = None):
    """Similarity retriever, optionally restricted to a single meeting."""
    search_kwargs: dict = {"k": k}
    if meeting_id:
        search_kwargs["filter"] = {"meeting_id": meeting_id}
    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def meeting_filter(meeting_ids: list[str] | None) -> dict | None:
    if not meeting_ids:
        return None
    if len(meeting_ids) == 1:
        return {"meeting_id": meeting_ids[0]}
    return {"meeting_id": {"$in": list(meeting_ids)}}


def list_indexed_meetings(vector_store: Chroma | None = None) -> list[dict]:
    """What is stored in the collection, grouped by meeting (newest first)."""
    store = vector_store or load_vector_store()
    data = store.get(include=["metadatas"])
    meetings: dict[str, dict] = {}
    for metadata in data.get("metadatas") or []:
        metadata = metadata or {}
        meeting_id = metadata.get("meeting_id") or LEGACY_MEETING_ID
        entry = meetings.setdefault(
            meeting_id,
            {
                "meeting_id": meeting_id,
                "title": metadata.get("source_title") or metadata.get("title") or "Earlier meeting (no metadata)",
                "meeting_date": metadata.get("meeting_date", ""),
                "meeting_ts": metadata.get("meeting_ts", 0),
                "processed_at": metadata.get("processed_at", ""),
                "source": metadata.get("source", ""),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return sorted(meetings.values(), key=lambda m: (m["meeting_ts"] or 0, m["processed_at"]), reverse=True)


if __name__ == "__main__":  # python -m core.vector_store [backup|stats]
    import sys

    from core.config import load_environment

    load_environment()
    command = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if command == "backup":
        print("Backup written to:", backup_vector_store() or "(no database yet)")
    else:
        for meeting in list_indexed_meetings():
            print(f"{meeting['meeting_date'] or '?':10}  {meeting['chunks']:4} chunks  {meeting['title']}")
