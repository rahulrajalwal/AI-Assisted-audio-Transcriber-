"""Persistent ChromaDB storage of transcript chunks with Hugging Face embeddings.

* Embeddings: ``all-MiniLM-L6-v2`` (local, CPU) via ``langchain-huggingface``.
* All meetings share one persistent collection; every chunk carries metadata
  (``meeting_id``, ``title``, ``processed_at``, ``source``, ``language``,
  ``chunk_index``) so retrieval can be scoped to one meeting or span all of them.
* Chunk IDs are deterministic (``<meeting_id>-<chunk_index>``) and a meeting
  that is already indexed is skipped, so re-processing never duplicates records.
* Existing data is never cleared: chunks indexed by older versions (without
  ``meeting_id``) stay searchable in "all meetings" mode.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from datetime import datetime, timezone
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import chroma_dir

logger = logging.getLogger(__name__)

COLLECTION_NAME = "meeting_transcript"  # unchanged, so existing data keeps working
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LEGACY_MEETING_ID = "legacy"

_store_lock = threading.Lock()


_embeddings = None
_embeddings_lock = threading.Lock()


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


def split_for_indexing(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(transcript)


def build_vector_store(
    transcript: str,
    *,
    meeting_id: str | None = None,
    title: str = "Untitled Meeting",
    source: str = "",
    language: str = "",
    processed_at: str | None = None,
    vector_store: Chroma | None = None,
) -> Chroma:
    """Index a transcript (once) and return the persistent vector store."""
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Cannot index an empty transcript.")

    store = vector_store or load_vector_store()
    meeting_id = meeting_id or compute_meeting_id(text)
    processed_at = processed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _store_lock:
        if meeting_exists(store, meeting_id):
            logger.info("Meeting %s is already indexed; skipping re-indexing.", meeting_id)
            return store

        chunks = split_for_indexing(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "meeting_id": meeting_id,
                    "title": title or "Untitled Meeting",
                    "processed_at": processed_at,
                    "source": source or "",
                    "language": language or "",
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        ids = [f"{meeting_id}-{index:05d}" for index in range(len(docs))]
        store.add_documents(documents=docs, ids=ids)
        logger.info("Indexed %d chunks for meeting %s.", len(docs), meeting_id)
    return store


def get_retriever(vector_store: Chroma, k: int = 4, meeting_id: str | None = None):
    """Similarity retriever, optionally restricted to a single meeting."""
    search_kwargs: dict = {"k": k}
    if meeting_id:
        search_kwargs["filter"] = {"meeting_id": meeting_id}
    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def list_indexed_meetings(vector_store: Chroma | None = None) -> list[dict]:
    """Summarise what is stored in the collection, grouped by meeting."""
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
                "title": metadata.get("title") or "Earlier meeting (no metadata)",
                "processed_at": metadata.get("processed_at", ""),
                "source": metadata.get("source", ""),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return sorted(meetings.values(), key=lambda m: m["processed_at"], reverse=True)
