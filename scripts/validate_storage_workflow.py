"""Live end-to-end validation of transcript storage and historical RAG.

Runs in two separate processes so step 7 ("restart the application") is a real
restart. Uses a temporary ChromaDB + meeting-history folder — your real data is
never touched. Requires OPENROUTER_API_KEY. Uses about 15 OpenRouter requests
(of the 50/day free quota): 2 meetings x 5 analysis calls + 5 questions.

Usage:  python scripts/validate_storage_workflow.py [path\\to\\meeting_audio.mp4]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def phase_one(audio: str | None) -> dict:
    from core.pipeline import analyse_transcript, run_pipeline
    from core.vector_store import compute_meeting_id, load_vector_store

    results: dict = {"ok": True}
    print("\nPHASE 1 — process two meetings", flush=True)

    start = time.perf_counter()
    if audio:
        first = run_pipeline(audio, "english", source_label=Path(audio).name, meeting_date="2026-10-29")
        results["first_mode"] = "audio → Whisper → OpenRouter"
    else:  # no audio supplied: use the budget-review fixture transcript
        text = (FIXTURES / "meeting_budget_review.txt").read_text(encoding="utf-8")
        first = analyse_transcript(text, source="meeting_budget_review.txt", meeting_date="2026-10-29")
        results["first_mode"] = "fixture transcript → OpenRouter"
    results["first_seconds"] = round(time.perf_counter() - start, 1)
    ok = check("1–2. meeting processed and transcript generated", bool(first.transcript),
               f"{len(first.transcript.split())} words, title '{first.title}', {results['first_seconds']}s")

    store = load_vector_store()
    data = store.get(where={"meeting_id": first.meeting_id}, include=["metadatas", "embeddings"])
    ok &= check("3. transcript split into text chunks", len(data["ids"]) >= 1, f"{len(data['ids'])} chunks")
    dims = {len(e) for e in data["embeddings"]}
    ok &= check("4. embeddings generated (all-MiniLM-L6-v2)", dims == {384}, f"dimension {dims}")
    ok &= check("5. chunks stored in ChromaDB", all(i.startswith(first.meeting_id) for i in data["ids"]))
    meta = min(data["metadatas"], key=lambda m: m["chunk_index"])
    required = {"meeting_id", "title", "meeting_date", "meeting_ts", "processed_at", "source", "source_type",
                "chunk_id", "chunk_index", "char_start", "char_end", "embedding_model", "schema_version"}
    missing = required - set(meta)
    has_time = "start_seconds" in meta
    ok &= check("6. metadata attached", not missing,
                f"missing {missing}" if missing else f"date {meta['meeting_date']}, source_type {meta['source_type']}, "
                f"timestamps {'yes' if has_time else 'n/a (no audio)'}")

    product = (FIXTURES / "meeting_product_sync.txt").read_text(encoding="utf-8")
    start = time.perf_counter()
    second = analyse_transcript(product, source="meeting_product_sync.txt", meeting_date="2026-11-05")
    results["second_seconds"] = round(time.perf_counter() - start, 1)
    print(f"  (second meeting analysed in {results['second_seconds']}s: '{second.title}')", flush=True)

    results.update(
        ok=ok,
        first_id=first.meeting_id,
        second_id=second.meeting_id,
        first_chunks=len(data["ids"]),
        second_chunks=len(store.get(where={"meeting_id": second.meeting_id}, include=[])["ids"]),
        total_chunks=len(store.get(include=[])["ids"]),
        product_id=compute_meeting_id(product),
    )
    return results


def phase_two(state: dict) -> bool:
    from core.pipeline import analyse_transcript
    from core.rag_engine import ask_question, load_rag_chain
    from core.vector_store import list_indexed_meetings, load_vector_store

    print("\nPHASE 2 — after restarting the application (new process)", flush=True)
    store = load_vector_store()
    ok = check("7–9. ChromaDB reloaded; stored chunks still available",
               len(store.get(include=[])["ids"]) == state["total_chunks"], f"{state['total_chunks']} chunks")
    meetings = {m["meeting_id"]: m for m in list_indexed_meetings(store)}
    ok &= check("10–11. both meetings stored separately",
                {state["first_id"], state["second_id"]} <= set(meetings)
                and len(store.get(where={"meeting_id": state["second_id"]}, include=[])["ids"]) == state["second_chunks"],
                ", ".join(f"{m['title']} ({m['meeting_date']}, {m['chunks']} chunks)" for m in meetings.values()))

    hits = store.similarity_search("Who is responsible for fixing the image cache crash?", k=3)
    ok &= check("12. semantic search returns relevant stored chunks",
                bool(hits) and hits[0].metadata["meeting_id"] == state["second_id"] and "image cache" in hits[0].page_content,
                f"top hit from '{hits[0].metadata['title']}'")

    product = (FIXTURES / "meeting_product_sync.txt").read_text(encoding="utf-8")
    before = len(store.get(include=[])["ids"])
    started = time.perf_counter()
    analyse_transcript(product, source="meeting_product_sync.txt", meeting_date="2026-11-05")
    ok &= check("13. reprocessing the same meeting creates no duplicates",
                len(store.get(include=[])["ids"]) == before,
                f"{before} chunks before and after; {time.perf_counter() - started:.1f}s (saved analysis reused, no LLM calls)")

    print("\n  14. OpenRouter answers from stored transcript chunks:", flush=True)
    questions = [
        ("this meeting", state["second_id"], None, "Who owns the image cache fix and by when?"),
        ("all meetings", None, state["second_id"], "What was decided in the previous meeting?"),
        ("all meetings", None, None, "What did we decide last week?"),
        ("all meetings", None, None, "Which meeting discussed hiring, and what was decided?"),
        ("this meeting", state["second_id"], None, "What was the agreed budget for the Berlin office?"),
    ]
    for scope, meeting_id, current, question in questions:
        started = time.perf_counter()
        result = ask_question(load_rag_chain(meeting_id), question, current_meeting_id=current)
        sources = sorted({f"{s['title']} ({s['meeting_date']})" for s in result["sources"]})
        print(f"\n  Q [{scope}]: {question}\n  A ({time.perf_counter() - started:.1f}s): {result['answer']}\n"
              f"  sources: {sources or 'none'}", flush=True)
    return ok


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        from core.config import load_environment

        load_environment()
        state_file = Path(sys.argv[3])
        if sys.argv[2] == "1":
            audio = sys.argv[4] if len(sys.argv) > 4 else None
            state_file.write_text(json.dumps(phase_one(audio)), encoding="utf-8")
            return 0
        return 0 if phase_two(json.loads(state_file.read_text(encoding="utf-8"))) else 1

    tmp = Path(tempfile.mkdtemp(prefix="storage_validation_"))
    env = {**os.environ, "CHROMA_DIR": str(tmp / "vector_db"), "MEETINGS_DIR": str(tmp / "meetings"),
           "WORK_DIR": str(tmp / "work"), "PYTHONIOENCODING": "utf-8"}
    state = tmp / "state.json"
    audio = [sys.argv[1]] if len(sys.argv) > 1 else []
    print(f"Temporary storage: {tmp}")
    first = subprocess.run([sys.executable, __file__, "--phase", "1", str(state), *audio], env=env)
    if first.returncode != 0:
        return first.returncode
    if not json.loads(state.read_text(encoding="utf-8"))["ok"]:
        print("Phase 1 checks failed.")
    return subprocess.run([sys.executable, __file__, "--phase", "2", str(state)], env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
